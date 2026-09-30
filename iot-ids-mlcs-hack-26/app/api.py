"""
api.py — FastAPI Backend for IoT IDS
======================================
Endpoints:
  GET  /           — welcome message
  GET  /health     — health check
  POST /train      — trigger model training
  POST /predict    — single prediction
  POST /predict-batch — batch prediction
  GET  /metrics    — latest evaluation metrics
  GET  /model-info — info about saved models
"""

import os
import sys
import json
import logging
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_and_prepare, discover_datasets, _load_config
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_pipeline
from src.train import (
    train_random_forest,
    save_model, list_saved_models, load_model,
)
from src.predict import predict_single, predict_batch
from src.evaluation import compute_metrics
from src.database import store_prediction, store_batch, get_prediction_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="MLCS-HACK-26 — IoT Intrusion Detection API",
    description="ML-based IoT Intrusion Detection & Risk Analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ─────────────────────────────────────────

class PredictionRequest(BaseModel):
    features: dict = Field(..., description="Column-name → value mapping")
    model_name: str = Field(default="random_forest", description="Model to use")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_score: int
    risk_level: str
    explanation: list[str]


class TrainRequest(BaseModel):
    data_path: Optional[str] = Field(default=None, description="Path to CSV (uses demo data if not provided)")
    model: str = Field(default="rf", description="Model: rf")
    label_col: Optional[str] = Field(default=None, description="Label column name override")


class TrainResponse(BaseModel):
    status: str
    models_trained: list[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    models_available: list[str]
    database_ok: bool


class MetricsResponse(BaseModel):
    metrics: dict


class ModelInfoResponse(BaseModel):
    models: list[str]
    default_model: str
    details: dict


# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "project": "MLCS-HACK-26",
        "title": "IoT Intrusion Detection & Risk Analysis",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    models = list_saved_models()
    return HealthResponse(
        status="healthy",
        models_available=models,
        database_ok=True,
    )


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    try:
        config = _load_config()

        # Find dataset
        if req.data_path:
            csv_path = req.data_path
        else:
            datasets = discover_datasets()
            if not datasets:
                from generate_demo_data import main as gen_main
                gen_main()
                datasets = discover_datasets()
            if not datasets:
                raise HTTPException(status_code=400, detail="No dataset found.")
            csv_path = datasets[0]

        df, label_col = load_and_prepare(csv_path, label_col=req.label_col)
        df = engineer_features(df)

        result = preprocess_pipeline(
            df, label_col,
            test_size=config["ml"]["test_size"],
            random_state=config["ml"]["random_state"],
        )

        trained = []
        rf = train_random_forest(result["X_train"], result["y_train"], config)
        save_model(rf, "random_forest")
        trained.append("random_forest")

        return TrainResponse(
            status="success",
            models_trained=trained,
            message=f"Trained on {len(result['X_train'])} samples with {len(result['feature_names'])} features.",
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: PredictionRequest):
    try:
        result = predict_single(req.features, model_name=req.model_name)

        # Store in DB
        store_prediction(
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            model_name=req.model_name,
            explanation="; ".join(result["explanation"]),
        )

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            explanation=result["explanation"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
async def predict_batch_endpoint(
    file: UploadFile = File(...),
    model_name: str = "random_forest",
):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        results = predict_batch(df, model_name=model_name)

        # Store in DB
        records = []
        for _, row in results.iterrows():
            records.append({
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "model_name": model_name,
                "explanation": row.get("explanation", ""),
            })
        store_batch(records)

        return {
            "status": "success",
            "total": len(results),
            "predictions": results.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
def metrics_endpoint():
    """Return the latest evaluation metrics from reports/."""
    eval_dir = os.path.join(PROJECT_ROOT, "reports", "evaluation")
    if not os.path.isdir(eval_dir):
        raise HTTPException(status_code=404, detail="No evaluation reports found. Train a model first.")

    files = sorted(
        [f for f in os.listdir(eval_dir) if f.endswith(".json")],
        reverse=True,
    )
    if not files:
        raise HTTPException(status_code=404, detail="No evaluation reports found.")

    with open(os.path.join(eval_dir, files[0]), "r") as f:
        report = json.load(f)

    return MetricsResponse(metrics=report)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info_endpoint():
    models = list_saved_models()
    if not models:
        raise HTTPException(status_code=404, detail="No models found. Train a model first.")

    default = "random_forest" if "random_forest" in models else models[0]

    details = {}
    for m in models:
        mdl = load_model(m)
        details[m] = {
            "type": type(mdl).__name__,
            "params": {k: str(v) for k, v in mdl.get_params().items()} if hasattr(mdl, "get_params") else {},
        }

    return ModelInfoResponse(models=models, default_model=default, details=details)


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)

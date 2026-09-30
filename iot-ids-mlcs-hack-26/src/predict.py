"""
predict.py — Prediction Engine
================================
Loads a trained model and preprocessing artefacts, applies them
to new data, and returns predictions with confidence, risk scores,
and explanations.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from .preprocessing import load_preprocessing, encode_categorical, scale_features
from .risk_scoring import compute_risk_score, risk_level, risk_color, batch_risk_scores
from .explainability import explain_single, explain_feature_importance
from .feature_engineering import engineer_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_model(name: str):
    """Load a saved model."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}. Train a model first.")
    return joblib.load(path)


def prepare_input(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    """
    Transform a raw DataFrame into the same feature space used during training.

    Steps:
      1. Engineer features
      2. Align columns to training feature set
      3. Encode categoricals (reusing training encoders)
      4. Scale (reusing training scaler)
    """
    df = engineer_features(df.copy())

    feature_names = preproc["feature_names"]
    encoders = preproc["encoders"]
    scaler = preproc["scaler"]

    # Ensure all training columns exist (fill missing with 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Keep only the training columns, in order
    df = df[feature_names].copy()

    # Encode categoricals
    df, _ = encode_categorical(df, encoders=encoders, fit=False)

    # Replace inf / NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Scale
    X_scaled, _, _ = scale_features(df, scaler=scaler, fit=False)
    return X_scaled


def predict_single(features: dict, model_name: str = "random_forest") -> dict:
    """
    Predict a single record.

    Parameters
    ----------
    features : dict
        Column-name → value mapping.
    model_name : str

    Returns
    -------
    dict with prediction, confidence, risk_score, risk_level, explanation.
    """
    model = load_model(model_name)
    preproc = load_preprocessing()
    le_label = preproc.get("label_encoder")

    df = pd.DataFrame([features])
    X = prepare_input(df, preproc)

    pred_encoded = model.predict(X)[0]
    pred_label = le_label.inverse_transform([pred_encoded])[0] if le_label else str(pred_encoded)

    # Confidence / probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))
        attack_idx = _attack_class_index(le_label)
        attack_prob = float(proba[attack_idx]) if attack_idx < len(proba) else confidence
    else:
        confidence = 1.0
        attack_prob = 1.0 if pred_label == "ATTACK" else 0.0

    score = compute_risk_score(attack_prob)
    level = risk_level(score)
    colour = risk_color(level)

    # Explanation
    explanation = explain_single(model, X[0], preproc["feature_names"])

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "risk_score": score,
        "risk_level": level,
        "risk_color": colour,
        "explanation": explanation,
    }


def predict_batch(df: pd.DataFrame, model_name: str = "random_forest") -> pd.DataFrame:
    """
    Predict a batch of records.

    Returns a DataFrame with columns:
      prediction, confidence, risk_score, risk_level, risk_color, explanation
    """
    model = load_model(model_name)
    preproc = load_preprocessing()
    le_label = preproc.get("label_encoder")

    X = prepare_input(df, preproc)

    preds_encoded = model.predict(X)
    preds_labels = le_label.inverse_transform(preds_encoded) if le_label else preds_encoded.astype(str)

    # Probabilities
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        confidences = probas.max(axis=1)
        attack_idx = _attack_class_index(le_label)
        risks = batch_risk_scores(probas, attack_class_index=attack_idx)
    else:
        confidences = np.ones(len(preds_encoded))
        risks = [
            {
                "attack_probability": 1.0 if p == "ATTACK" else 0.0,
                "risk_score": 100 if p == "ATTACK" else 0,
                "risk_level": "CRITICAL" if p == "ATTACK" else "LOW",
                "risk_color": "#d50000" if p == "ATTACK" else "#00c853",
            }
            for p in preds_labels
        ]

    # Feature importance explanation (same for all rows — global)
    feat_imp = explain_feature_importance(model, preproc["feature_names"])

    results = pd.DataFrame({
        "prediction": preds_labels,
        "confidence": np.round(confidences, 4),
        "risk_score": [r["risk_score"] for r in risks],
        "risk_level": [r["risk_level"] for r in risks],
        "risk_color": [r["risk_color"] for r in risks],
        "explanation": [feat_imp.get("summary", "N/A")] * len(preds_labels),
    })
    return results


def _attack_class_index(le_label) -> int:
    """Find the index of 'ATTACK' in the label encoder classes."""
    if le_label is None:
        return 1
    classes = list(le_label.classes_)
    if "ATTACK" in classes:
        return classes.index("ATTACK")
    return min(1, len(classes) - 1)

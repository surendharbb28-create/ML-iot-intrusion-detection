"""
train.py — Model Training
===========================
Trains Random Forest.
Persists models with Joblib.
"""

import os
import logging
import time
import numpy as np
import joblib
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _load_config() -> dict:
    cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_random_forest(X_train, y_train, config: dict | None = None) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    if config is None:
        config = _load_config()
    params = config.get("ml", {}).get("models", {}).get("random_forest", {})

    logger.info("Training Random Forest …")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 20),
        min_samples_split=params.get("min_samples_split", 5),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=params.get("n_jobs", -1),
        random_state=config.get("ml", {}).get("random_state", 42),
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("Random Forest trained in %.2f s", elapsed)
    return model


# ── Persistence ──────────────────────────────────────────────

def save_model(model, name: str):
    """Save a trained model to models/<name>.joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str):
    """Load a model from models/<name>.joblib."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def list_saved_models() -> list[str]:
    """Return names of saved models (without .joblib extension)."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return [
        f.replace(".joblib", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".joblib") and f != "feature_config.joblib"
    ]


# ── Combined training convenience ────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Train RF, evaluate on test set, save model.

    Returns dict: {model_name: {"model": …, "accuracy": …, "path": …}}
    """
    config = _load_config()
    results = {}

    # Random Forest
    rf = train_random_forest(X_train, y_train, config)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    path_rf = save_model(rf, "random_forest")
    results["random_forest"] = {"model": rf, "accuracy": acc_rf, "path": path_rf}

    return results

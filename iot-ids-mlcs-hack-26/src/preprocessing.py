"""
preprocessing.py — Data Preprocessing Pipeline
================================================
Handles:
  • Numeric scaling (StandardScaler)
  • Categorical encoding (LabelEncoder + OneHotEncoder)
  • Feature/target separation
  • Train/test splitting with optional stratification
  • Saving/loading preprocessing artefacts via Joblib
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def separate_features_target(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix ``X`` and target vector ``y``.

    Parameters
    ----------
    df : DataFrame
    label_col : str
    drop_cols : list[str] | None
        Additional columns to drop (IDs, timestamps, etc.).

    Returns
    -------
    (X, y)
    """
    if drop_cols is None:
        drop_cols = []
    cols_to_drop = [c for c in [label_col] + drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[label_col]
    return X, y


def encode_categorical(
    X: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all ``object`` / ``category`` columns.

    If ``fit=True``, new encoders are created and returned.
    If ``fit=False``, existing ``encoders`` dict is used (handles unseen values).

    Returns
    -------
    (encoded_X, encoders_dict)
    """
    if encoders is None:
        encoders = {}
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            X[col] = le.transform(X[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                # Unknown column → drop it
                X.drop(columns=[col], inplace=True)
                continue
            X[col] = X[col].astype(str)
            # Handle unseen categories
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])

    return X, encoders


def scale_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Scale numeric features with StandardScaler.

    Returns
    -------
    (scaled_array, scaler, feature_names)
    """
    feature_names = list(X.columns)
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(np.float64))
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False")
        X_scaled = scaler.transform(X.values.astype(np.float64))
    return X_scaled, scaler, feature_names


def split_data(
    X, y,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    """
    Train/test split with optional stratification.
    """
    strat = y if stratify else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        # stratification may fail if a class has too few samples
        logger.warning("Stratification failed — splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )
    return X_train, X_test, y_train, y_test


# ── Persistence ──────────────────────────────────────────────

def save_preprocessing(
    scaler: StandardScaler,
    encoders: dict,
    feature_names: list[str],
    label_encoder: LabelEncoder | None = None,
):
    """Save all preprocessing artefacts to models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    config = {
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
    }
    path = os.path.join(MODELS_DIR, "feature_config.joblib")
    joblib.dump(config, path)
    logger.info("Saved preprocessing config → %s", path)


def load_preprocessing() -> dict:
    """Load preprocessing artefacts."""
    path = os.path.join(MODELS_DIR, "feature_config.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Preprocessing config not found. Train a model first."
        )
    return joblib.load(path)


def preprocess_pipeline(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict:
    """
    Full preprocessing pipeline: separate → encode label → encode features → scale → split.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        scaler, encoders, feature_names, label_encoder, class_names
    """
    from .data_loader import map_labels, _load_config

    config = _load_config()

    X, y_raw = separate_features_target(df, label_col, drop_cols)

    # Binary label encoding
    y_binary, y_original = map_labels(y_raw, config)

    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y_binary)

    # Remove any remaining ID / timestamp cols
    ds_cfg = config.get("dataset", {})
    id_cols = ds_cfg.get("id_columns", [])
    ts_cols = ds_cfg.get("timestamp_candidates", [])
    to_drop = [c for c in id_cols + ts_cols if c in X.columns]
    if to_drop:
        X = X.drop(columns=to_drop)

    # Encode categoricals
    X, encoders = encode_categorical(X, fit=True)

    # Replace any remaining inf/NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Scale
    X_scaled, scaler, feature_names = scale_features(X, fit=True)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Save artefacts
    save_preprocessing(scaler, encoders, feature_names, le_label)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "label_encoder": le_label,
        "class_names": list(le_label.classes_),
    }

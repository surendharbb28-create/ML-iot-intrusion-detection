"""
evaluation.py — Model Evaluation & Security Metrics
=====================================================
Computes Accuracy, Precision, Recall, F1, Confusion Matrix,
Detection Rate, FPR, FNR with safe division-by-zero handling.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when denominator is zero."""
    return float(numerator / denominator) if denominator != 0 else 0.0


def compute_metrics(y_true, y_pred, class_names: list[str] | None = None) -> dict:
    """
    Compute all required classification + cybersecurity metrics.

    Parameters
    ----------
    y_true : array-like  — ground-truth labels (encoded ints or strings)
    y_pred : array-like  — predicted labels
    class_names : list[str] | None

    Returns
    -------
    dict with all metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine averaging strategy
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = len(unique_classes) <= 2
    avg = "binary" if is_binary else "weighted"

    # Positional label for binary metrics
    pos_label = unique_classes[1] if is_binary and len(unique_classes) == 2 else None

    metrics: dict = {}
    metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    metrics["precision"] = round(float(precision_score(
        y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0
    )), 4)
    metrics["recall"] = round(float(recall_score(
        y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0
    )), 4)
    metrics["f1_score"] = round(float(f1_score(
        y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0
    )), 4)
    metrics["averaging_method"] = avg

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_labels"] = [str(c) for c in unique_classes]

    # Cybersecurity metrics (binary: class 0 = ATTACK, class 1 = NORMAL typically,
    # but we compute from the CM directly)
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["detection_rate"] = round(_safe_div(tp, tp + fn), 4)
        metrics["false_positive_rate"] = round(_safe_div(fp, fp + tn), 4)
        metrics["false_negative_rate"] = round(_safe_div(fn, fn + tp), 4)
    else:
        # Multiclass: per-class rates
        per_class = {}
        for i, cls in enumerate(unique_classes):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            per_class[str(cls)] = {
                "detection_rate": round(_safe_div(tp_i, tp_i + fn_i), 4),
                "false_positive_rate": round(_safe_div(fp_i, fp_i + tn_i), 4),
                "false_negative_rate": round(_safe_div(fn_i, fn_i + tp_i), 4),
            }
        metrics["per_class_rates"] = per_class
        # Aggregate (macro)
        drs = [v["detection_rate"] for v in per_class.values()]
        fprs = [v["false_positive_rate"] for v in per_class.values()]
        fnrs = [v["false_negative_rate"] for v in per_class.values()]
        metrics["detection_rate"] = round(float(np.mean(drs)), 4) if drs else 0.0
        metrics["false_positive_rate"] = round(float(np.mean(fprs)), 4) if fprs else 0.0
        metrics["false_negative_rate"] = round(float(np.mean(fnrs)), 4) if fnrs else 0.0

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=[str(c) for c in unique_classes],
        zero_division=0, output_dict=True
    )

    return metrics


def save_evaluation_report(
    metrics: dict,
    model_name: str,
    dataset_info: dict | None = None,
    feature_names: list[str] | None = None,
    feature_importances: list[float] | None = None,
):
    """
    Save a JSON evaluation report under reports/evaluation/.
    """
    report_dir = os.path.join(PROJECT_ROOT, "reports", "evaluation")
    os.makedirs(report_dir, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "metrics": metrics,
    }
    if dataset_info:
        report["dataset"] = dataset_info
    if feature_names:
        report["feature_names"] = feature_names
    if feature_importances is not None and feature_names:
        fi = sorted(
            zip(feature_names, [float(f) for f in feature_importances]),
            key=lambda x: x[1], reverse=True
        )
        report["feature_importance"] = [{"feature": n, "importance": v} for n, v in fi]

    report["limitations"] = [
        "Results may vary with different train/test splits.",
        "Model performance on synthetic/demo data does NOT represent real-world accuracy.",
        "Feature engineering is limited to columns present in the dataset.",
    ]

    filename = f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(report_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Evaluation report saved → %s", path)
    return path

"""
explainability.py — Explainable AI (XAI)
==========================================
Provides feature-importance explanations using:
  1. SHAP (if installed and compatible)
  2. Random Forest built-in feature importances (fallback)

For each prediction it returns:
  • top contributing features
  • human-readable explanation sentence
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try importing SHAP
_SHAP_AVAILABLE = False
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    logger.info("SHAP not installed — falling back to built-in feature importance.")


def explain_feature_importance(model, feature_names: list[str], top_n: int = 10) -> dict:
    """
    Extract global feature importance from a model.

    Works with:
      • RandomForestClassifier (``feature_importances_``)
      • LogisticRegression (``coef_``)

    Returns
    -------
    dict with keys: importances (list of dicts), summary (str)
    """
    importances = _get_importances(model, feature_names)
    if importances is None:
        return {"importances": [], "summary": "Feature importance not available for this model type."}

    # Sort descending
    sorted_imp = sorted(importances, key=lambda x: abs(x["importance"]), reverse=True)
    top = sorted_imp[:top_n]

    summary = _build_summary(top)

    return {
        "importances": sorted_imp,
        "top_features": top,
        "summary": summary,
    }


def explain_single(model, x_single: np.ndarray, feature_names: list[str], top_n: int = 5) -> list[str]:
    """
    Explain a single prediction.

    Tries SHAP first; falls back to global feature importance.

    Returns
    -------
    list[str] — human-readable explanation lines.
    """
    # Attempt SHAP
    if _SHAP_AVAILABLE:
        try:
            explanations = _shap_explain_single(model, x_single, feature_names, top_n)
            if explanations:
                return explanations
        except Exception as e:
            logger.debug("SHAP explanation failed: %s — using fallback", e)

    # Fallback: global importance weighted by feature value
    return _importance_explain_single(model, x_single, feature_names, top_n)


def get_shap_values(model, X, feature_names: list[str]) -> dict | None:
    """
    Compute SHAP values for a dataset (used in dashboard).

    Returns
    -------
    dict with shap_values, expected_value, feature_names  — or None if SHAP unavailable.
    """
    if not _SHAP_AVAILABLE:
        return None
    try:
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            # Use a KernelExplainer subsample for non-tree models
            bg = shap.sample(X, min(100, len(X))) if len(X) > 100 else X
            explainer = shap.KernelExplainer(model.predict_proba, bg)

        sv = explainer.shap_values(X[:min(200, len(X))])
        return {
            "shap_values": sv,
            "expected_value": explainer.expected_value,
            "feature_names": feature_names,
        }
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


# ── Internal helpers ─────────────────────────────────────────

def _get_importances(model, feature_names: list[str]) -> list[dict] | None:
    """Extract importances from model attributes."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
    else:
        return None

    if len(imp) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(imp))]

    return [{"feature": n, "importance": round(float(v), 6)} for n, v in zip(feature_names, imp)]


def _build_summary(top_features: list[dict]) -> str:
    """Build a one-line summary of the top features."""
    if not top_features:
        return "No feature importance data available."
    names = [f["feature"] for f in top_features[:3]]
    return (
        f"Top contributing features: {', '.join(names)}. "
        "These features contributed most strongly to the model's prediction."
    )


def _shap_explain_single(model, x_single, feature_names, top_n) -> list[str]:
    """SHAP-based single-instance explanation."""
    x_2d = x_single.reshape(1, -1)
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, x_2d)

    sv = explainer.shap_values(x_2d)

    # sv may be a list (one per class)
    if isinstance(sv, list):
        sv = sv[-1]  # use ATTACK class
    sv = sv.flatten()

    if len(sv) != len(feature_names):
        return []

    pairs = sorted(zip(feature_names, sv), key=lambda p: abs(p[1]), reverse=True)
    lines = []
    for name, val in pairs[:top_n]:
        direction = "increased" if val > 0 else "decreased"
        lines.append(f"'{name}' {direction} the attack prediction (SHAP value: {val:.4f}).")
    return lines


def _importance_explain_single(model, x_single, feature_names, top_n) -> list[str]:
    """Fallback importance-based explanation for a single instance."""
    importances = _get_importances(model, feature_names)
    if not importances:
        return ["Feature importance data is not available for this model."]

    sorted_imp = sorted(importances, key=lambda x: abs(x["importance"]), reverse=True)
    lines = []
    for item in sorted_imp[:top_n]:
        feat = item["feature"]
        imp = item["importance"]
        # Find feature index
        try:
            idx = feature_names.index(feat)
            val = x_single[idx]
        except (ValueError, IndexError):
            val = None

        if val is not None:
            lines.append(
                f"'{feat}' (value={val:.2f}, importance={imp:.4f}) "
                f"contributed to the model's prediction."
            )
        else:
            lines.append(
                f"'{feat}' (importance={imp:.4f}) contributed to the model's prediction."
            )
    return lines

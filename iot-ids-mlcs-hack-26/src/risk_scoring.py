"""
risk_scoring.py — ML-Derived Security Risk Scoring
====================================================

Mathematical Logic
------------------
For a supervised classifier that outputs ``predict_proba``:

    attack_probability = P(class = ATTACK | features)

    risk_score = round(attack_probability × 100)

Risk levels:
    0–29   → LOW
    30–59  → MEDIUM
    60–79  → HIGH
    80–100 → CRITICAL

IMPORTANT: This is the application's ML-derived risk score.
It is NOT an industry-standard security score.  It reflects the
model's confidence that a given traffic sample is malicious.
"""

import os
import logging
import yaml
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_config() -> dict:
    cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_risk_score(attack_probability: float) -> int:
    """
    Convert attack probability → integer risk score in [0, 100].

    Formula:
        risk_score = round(attack_probability × 100)
    """
    score = int(round(float(attack_probability) * 100))
    return max(0, min(100, score))


def risk_level(score: int, config: dict | None = None) -> str:
    """
    Map an integer risk score to a human-readable risk level.
    """
    if config is None:
        config = _load_config()

    levels = config.get("risk_scoring", {}).get("levels", {})
    for level_name, bounds in levels.items():
        if bounds["min"] <= score <= bounds["max"]:
            return level_name

    # Fallback
    if score < 30:
        return "LOW"
    elif score < 60:
        return "MEDIUM"
    elif score < 80:
        return "HIGH"
    else:
        return "CRITICAL"


def risk_color(level: str, config: dict | None = None) -> str:
    """Return the hex colour associated with a risk level."""
    if config is None:
        config = _load_config()
    levels = config.get("risk_scoring", {}).get("levels", {})
    return levels.get(level, {}).get("color", "#888888")


def batch_risk_scores(
    probabilities: np.ndarray,
    attack_class_index: int = 1,
) -> list[dict]:
    """
    Compute risk scores for a batch of predictions.

    Parameters
    ----------
    probabilities : ndarray of shape (n_samples, n_classes)
        Output of ``model.predict_proba(X)``.
    attack_class_index : int
        Index of the ATTACK class in the probability array.

    Returns
    -------
    list[dict] — each dict has: risk_score, risk_level, risk_color, attack_probability
    """
    config = _load_config()
    results = []
    for prob_row in probabilities:
        p_attack = float(prob_row[attack_class_index]) if len(prob_row) > attack_class_index else float(prob_row[0])
        score = compute_risk_score(p_attack)
        level = risk_level(score, config)
        colour = risk_color(level, config)
        results.append({
            "attack_probability": round(p_attack, 4),
            "risk_score": score,
            "risk_level": level,
            "risk_color": colour,
        })
    return results


def risk_distribution(scores: list[int]) -> dict:
    """Count how many scores fall in each risk band."""
    config = _load_config()
    levels = config.get("risk_scoring", {}).get("levels", {})
    dist = {name: 0 for name in levels}
    for s in scores:
        lvl = risk_level(s, config)
        dist[lvl] = dist.get(lvl, 0) + 1
    return dist

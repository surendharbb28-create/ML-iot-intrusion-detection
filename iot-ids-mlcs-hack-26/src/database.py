"""
database.py — SQLite Prediction History
=========================================
Stores prediction results in a lightweight SQLite database
for the dashboard history page.
"""

import os
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "predictions.db")


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the predictions database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            prediction  TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            risk_score  INTEGER NOT NULL,
            risk_level  TEXT    NOT NULL,
            model_name  TEXT    NOT NULL,
            explanation TEXT,
            input_hash  TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)


def store_prediction(
    prediction: str,
    confidence: float,
    risk_score: int,
    risk_level: str,
    model_name: str,
    explanation: str | None = None,
    input_hash: str | None = None,
):
    """Insert a single prediction record."""
    init_db()
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO predictions
            (timestamp, prediction, confidence, risk_score, risk_level, model_name, explanation, input_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            prediction,
            round(confidence, 4),
            risk_score,
            risk_level,
            model_name,
            explanation,
            input_hash,
        ),
    )
    conn.commit()
    conn.close()


def store_batch(records: list[dict]):
    """Insert multiple prediction records at once."""
    init_db()
    conn = _get_connection()
    for r in records:
        conn.execute(
            """
            INSERT INTO predictions
                (timestamp, prediction, confidence, risk_score, risk_level, model_name, explanation, input_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                r.get("prediction", ""),
                round(r.get("confidence", 0.0), 4),
                r.get("risk_score", 0),
                r.get("risk_level", "LOW"),
                r.get("model_name", "unknown"),
                r.get("explanation", None),
                r.get("input_hash", None),
            ),
        )
    conn.commit()
    conn.close()
    logger.info("Stored %d predictions in database.", len(records))


def get_predictions(
    limit: int = 500,
    risk_level: str | None = None,
    prediction: str | None = None,
    model_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """
    Retrieve prediction history with optional filters.
    """
    init_db()
    conn = _get_connection()
    query = "SELECT * FROM predictions WHERE 1=1"
    params: list = []

    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level)
    if prediction:
        query += " AND prediction = ?"
        params.append(prediction)
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prediction_stats() -> dict:
    """Return aggregate statistics from the prediction history."""
    init_db()
    conn = _get_connection()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    attacks = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'ATTACK'").fetchone()[0]
    normals = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'NORMAL'").fetchone()[0]

    levels = {}
    for row in conn.execute("SELECT risk_level, COUNT(*) as cnt FROM predictions GROUP BY risk_level"):
        levels[row["risk_level"]] = row["cnt"]

    conn.close()
    return {
        "total": total,
        "attacks": attacks,
        "normals": normals,
        "risk_levels": levels,
    }


def clear_history():
    """Delete all prediction records."""
    init_db()
    conn = _get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    logger.info("Prediction history cleared.")

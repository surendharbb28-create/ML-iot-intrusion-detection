"""
tests/test_core.py — Test Suite for MLCS-HACK-26
==================================================
Covers: data loading, preprocessing, feature engineering,
risk scoring, prediction, metrics, API health.

Run:  pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small sample DataFrame resembling IoT traffic."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "duration": rng.exponential(2.0, n).round(3),
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["http", "dns", "ssh", "-"], n),
        "conn_state": rng.choice(["SF", "S0", "REJ", "RSTO"], n),
        "orig_bytes": rng.integers(0, 5000, n),
        "resp_bytes": rng.integers(0, 5000, n),
        "orig_pkts": rng.integers(1, 100, n),
        "resp_pkts": rng.integers(0, 50, n),
        "id.resp_p": rng.choice([22, 80, 443, 53, 8080, 31337], n),
        "label": np.concatenate([
            np.full(n // 2, "Benign"),
            rng.choice(["DDoS", "DoS", "Reconnaissance"], n - n // 2),
        ]),
    })
    return df


@pytest.fixture
def config():
    from src.data_loader import _load_config
    return _load_config()


# ── Data Loading ─────────────────────────────────────────────

class TestDataLoader:
    def test_load_csv_missing(self):
        from src.data_loader import load_csv
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent_file.csv")

    def test_load_csv_bad_ext(self, tmp_path):
        from src.data_loader import load_csv
        f = tmp_path / "file.xyz"
        f.write_text("a,b\n1,2")
        with pytest.raises(ValueError):
            load_csv(str(f))

    def test_inspect_columns(self, sample_df):
        from src.data_loader import inspect_columns
        info = inspect_columns(sample_df)
        assert "columns" in info
        assert "dtypes" in info
        assert info["shape"] == sample_df.shape

    def test_detect_label_column(self, sample_df, config):
        from src.data_loader import detect_label_column
        lbl = detect_label_column(sample_df, config)
        assert lbl == "label"

    def test_clean_dataframe(self, sample_df):
        from src.data_loader import clean_dataframe
        sample_df.loc[0, "duration"] = np.inf
        sample_df.loc[1, "orig_bytes"] = np.nan
        cleaned = clean_dataframe(sample_df)
        assert not np.isinf(cleaned.select_dtypes(include=[np.number]).values).any()
        assert not cleaned.select_dtypes(include=[np.number]).isnull().any().any()

    def test_map_labels(self, sample_df, config):
        from src.data_loader import map_labels
        binary, original = map_labels(sample_df["label"], config)
        assert set(binary.unique()) == {"NORMAL", "ATTACK"}


# ── Preprocessing ────────────────────────────────────────────

class TestPreprocessing:
    def test_separate_features_target(self, sample_df):
        from src.preprocessing import separate_features_target
        X, y = separate_features_target(sample_df, "label")
        assert "label" not in X.columns
        assert len(y) == len(sample_df)

    def test_encode_categorical(self, sample_df):
        from src.preprocessing import encode_categorical
        X = sample_df[["proto", "service"]].copy()
        X_enc, encoders = encode_categorical(X, fit=True)
        assert X_enc["proto"].dtype in (np.int32, np.int64, np.intp)
        assert "proto" in encoders

    def test_scale_features(self):
        from src.preprocessing import scale_features
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        X_s, scaler, names = scale_features(X, fit=True)
        assert X_s.shape == (3, 2)
        assert abs(X_s.mean()) < 1e-10


# ── Feature Engineering ──────────────────────────────────────

class TestFeatureEngineering:
    def test_engineer_features(self, sample_df):
        from src.feature_engineering import engineer_features, get_feature_docs
        df = engineer_features(sample_df)
        docs = get_feature_docs()
        # Should have created at least some features
        assert len(docs) > 0
        # Original columns still present
        assert "orig_bytes" in df.columns

    def test_no_fake_features(self):
        """If required columns don't exist, no features should be added."""
        from src.feature_engineering import engineer_features, get_feature_docs
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df2 = engineer_features(df)
        docs = get_feature_docs()
        assert len(docs) == 0


# ── Risk Scoring ─────────────────────────────────────────────

class TestRiskScoring:
    def test_compute_risk_score(self):
        from src.risk_scoring import compute_risk_score
        assert compute_risk_score(0.0) == 0
        assert compute_risk_score(1.0) == 100
        assert compute_risk_score(0.5) == 50
        assert compute_risk_score(0.92) == 92

    def test_risk_level(self):
        from src.risk_scoring import risk_level
        assert risk_level(10) == "LOW"
        assert risk_level(45) == "MEDIUM"
        assert risk_level(70) == "HIGH"
        assert risk_level(95) == "CRITICAL"

    def test_risk_distribution(self):
        from src.risk_scoring import risk_distribution
        scores = [5, 15, 35, 55, 65, 75, 85, 95]
        dist = risk_distribution(scores)
        assert dist["LOW"] == 2
        assert dist["MEDIUM"] == 2
        assert dist["HIGH"] == 2
        assert dist["CRITICAL"] == 2


# ── Evaluation / Metrics ─────────────────────────────────────

class TestEvaluation:
    def test_compute_metrics_binary(self):
        from src.evaluation import compute_metrics
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        m = compute_metrics(y_true, y_pred)
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["detection_rate"] <= 1
        assert 0 <= m["false_positive_rate"] <= 1

    def test_safe_division(self):
        from src.evaluation import _safe_div
        assert _safe_div(1, 0) == 0.0
        assert _safe_div(3, 2) == 1.5


# ── Prediction ───────────────────────────────────────────────

class TestPrediction:
    def test_predict_single_no_model(self):
        from src.predict import predict_single
        with pytest.raises(FileNotFoundError):
            predict_single({"a": 1}, model_name="nonexistent_model_xyz")


# ── Database ─────────────────────────────────────────────────

class TestDatabase:
    def test_init_and_store(self, tmp_path, monkeypatch):
        import src.database as db
        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()
        db.store_prediction("ATTACK", 0.95, 95, "CRITICAL", "rf", "test")
        records = db.get_predictions(limit=10)
        assert len(records) == 1
        assert records[0]["prediction"] == "ATTACK"

    def test_stats(self, tmp_path, monkeypatch):
        import src.database as db
        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test2.db"))
        db.init_db()
        db.store_prediction("ATTACK", 0.9, 90, "CRITICAL", "rf")
        db.store_prediction("NORMAL", 0.8, 15, "LOW", "rf")
        stats = db.get_prediction_stats()
        assert stats["total"] == 2
        assert stats["attacks"] == 1
        assert stats["normals"] == 1


# ── API Health ───────────────────────────────────────────────

class TestAPI:
    def test_root(self):
        from fastapi.testclient import TestClient
        from app.api import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project"] == "MLCS-HACK-26"

    def test_health(self):
        from fastapi.testclient import TestClient
        from app.api import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

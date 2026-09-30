"""
train_model.py — CLI entry point for model training
=====================================================
Usage:
    python train_model.py                     # Train Random Forest on demo data
    python train_model.py --data path/to.csv  # Train on a specific CSV
"""

import os
import sys
import argparse
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("train_model")

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_and_prepare, discover_datasets, map_labels, clean_dataframe, _load_config
from src.feature_engineering import engineer_features, get_feature_docs
from src.preprocessing import preprocess_pipeline
from src.train import (
    train_random_forest,
    save_model,
)
from src.evaluation import compute_metrics, save_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Train IoT IDS models")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name")
    parser.add_argument("--nrows", type=int, default=None, help="Max rows to load")
    args = parser.parse_args()

    print("=" * 64)
    print("  MLCS-HACK-26 - IoT Intrusion Detection System")
    print("  Model Training Pipeline")
    print("=" * 64)

    # ── Find dataset ─────────────────────────────────────────
    if args.data:
        csv_path = args.data
    else:
        datasets = discover_datasets()
        if not datasets:
            print("\n[!] No dataset found in data/raw/")
            print("    Generating synthetic demo data...\n")
            from generate_demo_data import main as gen_main
            gen_main()
            datasets = discover_datasets()
        csv_path = datasets[0]
        print(f"\nUsing dataset: {csv_path}")

    # ── Load & prepare ───────────────────────────────────────
    df, label_col = load_and_prepare(csv_path, label_col=args.label_col, nrows=args.nrows)
    print(f"  Shape: {df.shape}")
    print(f"  Label column: {label_col}")

    # ── Feature engineering ──────────────────────────────────
    df = engineer_features(df)
    feat_docs = get_feature_docs()
    if feat_docs:
        print(f"  Engineered features: {list(feat_docs.keys())}")

    # ── Preprocessing ────────────────────────────────────────
    config = _load_config()
    result = preprocess_pipeline(
        df, label_col,
        test_size=config["ml"]["test_size"],
        random_state=config["ml"]["random_state"],
        stratify=config["ml"].get("stratify", True),
    )

    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    feature_names = result["feature_names"]
    class_names = result["class_names"]

    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"  Classes: {class_names}")
    print(f"  Features: {len(feature_names)}")

    # ── Dataset info for reports ─────────────────────────────
    dataset_info = {
        "file": csv_path,
        "total_rows": len(df),
        "num_features": len(feature_names),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": {str(c): int((y_train == i).sum() + (y_test == i).sum())
                               for i, c in enumerate(class_names)},
    }

    # ── Training ─────────────────────────────────────────────
    print("\n── Random Forest ───────────────────────────────")
    rf = train_random_forest(X_train, y_train, config)
    save_model(rf, "random_forest")
    preds = rf.predict(X_test)
    metrics = compute_metrics(y_test, preds, class_names)
    _print_metrics(metrics)
    save_evaluation_report(metrics, "random_forest", dataset_info, feature_names,
                           rf.feature_importances_)

    print("\n" + "=" * 64)
    print("  [OK] Training complete. Models saved in models/")
    print("=" * 64 + "\n")


def _print_metrics(m: dict):
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  Recall         : {m['recall']:.4f}")
    print(f"  F1 Score       : {m['f1_score']:.4f}")
    print(f"  Detection Rate : {m['detection_rate']:.4f}")
    print(f"  FPR            : {m['false_positive_rate']:.4f}")
    print(f"  FNR            : {m['false_negative_rate']:.4f}")


if __name__ == "__main__":
    main()

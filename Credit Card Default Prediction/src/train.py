"""
Model Training Pipeline
Real-Time Credit Risk & Fraud Detection Intelligence Platform

End-to-end training orchestration:
  1. Load data (real or synthetic)
  2. Feature engineering
  3. Train/val/test split (stratified)
  4. Train stacked XGBoost + LightGBM ensemble
  5. Evaluate on hold-out test set
  6. Save artifacts + metrics

Usage:
    python src/train.py [--data-dir data/] [--model-dir data/models/] [--quick]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import load_data
from src.features import build_feature_set
from src.models import (
    train_xgboost,
    train_lightgbm,
    train_stacked_ensemble,
    evaluate_model,
    find_optimal_threshold,
    save_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--model-dir", type=str, default="data/models", help="Model artifact output dir")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 50k rows, single XGB model (for testing pipeline)"
    )
    parser.add_argument(
        "--model-type", choices=["xgb", "lgb", "ensemble"], default="ensemble",
        help="Model type to train"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = _PROJECT_ROOT / args.data_dir
    model_dir = _PROJECT_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    logger.info("=" * 70)
    logger.info("Credit Risk & Fraud Detection — Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Mode: {'QUICK' if args.quick else 'FULL'} | Model: {args.model_type}")

    # ── 1. Load Data ──
    logger.info("Step 1/6: Loading data")
    df, source = load_data(data_dir)
    logger.info(f"Data source: {source} | Shape: {df.shape} | Fraud rate: {df['isFraud'].mean():.4%}")

    if args.quick:
        logger.info("Quick mode: sampling 50,000 rows (stratified)")
        df, _ = train_test_split(df, train_size=50_000, stratify=df["isFraud"], random_state=42)
        logger.info(f"Sampled shape: {df.shape}")

    # ── 2. Feature Engineering ──
    logger.info("Step 2/6: Feature engineering")
    t_fe = time.time()
    df_features, feature_cols = build_feature_set(df)
    logger.info(f"Feature engineering: {len(feature_cols)} features in {time.time() - t_fe:.1f}s")

    # ── 3. Train/Val/Test Split ──
    logger.info("Step 3/6: Splitting data (60/20/20 stratified)")
    X = df_features[feature_cols].fillna(-999).astype(np.float32)
    y = df_features["isFraud"].values.astype(np.int8)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    logger.info(
        f"Train: {len(X_train):,} ({y_train.mean():.3%} fraud) | "
        f"Val: {len(X_val):,} ({y_val.mean():.3%} fraud) | "
        f"Test: {len(X_test):,} ({y_test.mean():.3%} fraud)"
    )

    # ── 4. Train Model ──
    logger.info(f"Step 4/6: Training {args.model_type} model")
    t_train = time.time()

    if args.model_type == "xgb" or args.quick:
        model, train_results = train_xgboost(X_train, y_train, X_val, y_val)
        model_name = "fraud_model"
    elif args.model_type == "lgb":
        model, train_results = train_lightgbm(X_train, y_train, X_val, y_val)
        model_name = "fraud_model"
    else:
        n_folds = 3 if args.quick else 5
        model, train_results = train_stacked_ensemble(
            X_train, y_train, X_val, y_val, n_folds=n_folds
        )
        model_name = "fraud_model"

    logger.info(f"Training complete in {time.time() - t_train:.1f}s")

    # ── 5. Evaluate ──
    logger.info("Step 5/6: Evaluating on test set")
    eval_metrics = evaluate_model(model, X_test, y_test, threshold=0.5)

    # Find optimal threshold
    if isinstance(model, dict) and "xgb_models" in model:
        from src.models import ensemble_predict_proba
        y_proba_test = ensemble_predict_proba(model, X_test)
    else:
        y_proba_test = model.predict_proba(X_test)[:, 1]

    optimal_threshold = find_optimal_threshold(y_test, y_proba_test)
    eval_metrics["optimal_threshold"] = optimal_threshold

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:40s}: {v:.4f}")
        elif isinstance(v, int):
            logger.info(f"  {k:40s}: {v:,}")
    logger.info("=" * 50)

    # ── 6. Save Artifacts ──
    logger.info("Step 6/6: Saving artifacts")
    save_artifacts(model, optimal_threshold, feature_cols, model_dir, model_name=model_name)

    # Save eval metrics
    metrics_path = model_dir / "eval_metrics.json"
    metrics_to_save = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float, str, list))}
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

    # Save feature list
    features_path = model_dir / "feature_cols.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Feature list saved to {features_path}")

    total_time = time.time() - t_total
    logger.info("")
    logger.info(f"Training pipeline complete in {total_time:.1f}s")
    logger.info(f"AUC-ROC: {eval_metrics['auc_roc']:.4f} | AUC-PR: {eval_metrics['auc_pr']:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info("")
    logger.info("To serve the model:")
    logger.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    logger.info("To launch the dashboard:")
    logger.info("  streamlit run dashboard/app.py --server.port 8501")


if __name__ == "__main__":
    main()

"""
train.py — Parkinson's Disease Detection
==========================================
End-to-end training entrypoint.

Usage:
    py -3.11 src/train.py

Pipeline:
    1. Load UCI Parkinson's Telemonitoring (or synthetic fallback)
    2. Build multi-modal features (voice + gait + tremor)
    3. Create binary target: total_UPDRS > 20 → PD=1 (or use 'status' if present)
    4. Train binary classifier (XGBoost)
    5. Train UPDRS regression model
    6. Subject-level cross-validation
    7. Evaluate: AUC-ROC, sensitivity, specificity, UPDRS RMSE
    8. Save artifacts + metrics.json to data/models/
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Module imports ─────────────────────────────────────────────────────────────
from data_loader import load_data, create_subject_splits
from features import build_multimodal_features
from models import (
    train_binary_classifier,
    train_updrs_regressor,
    subject_level_cross_validate,
    evaluate_classifier,
    evaluate_regressor,
    save_artifacts,
)

MAX_ROWS = 100_000


def _print_box(lines: list[str], width: int = 64) -> None:
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    for line in lines:
        print(f"| {line:<{width - 4}} |")
    print(border)


def _build_binary_target(df: pd.DataFrame) -> pd.Series:
    """
    Binary target strategy:
    - Use 'status' if present (classic UCI dataset: 1=PD, 0=healthy).
    - Else use total_UPDRS > 20 as PD indicator.
    - Else use UPDRS_total > 20 (synthetic data column name).
    """
    if "status" in df.columns:
        return df["status"].astype(int)
    if "total_UPDRS" in df.columns:
        return (df["total_UPDRS"] > 20).astype(int)
    if "UPDRS_total" in df.columns:
        return (df["UPDRS_total"] > 20).astype(int)
    raise ValueError("Cannot determine binary PD target: no 'status', 'total_UPDRS', or 'UPDRS_total' column.")


def _get_updrs_target(df: pd.DataFrame) -> pd.Series:
    """Return the continuous UPDRS target column."""
    for col in ["total_UPDRS", "UPDRS_total", "motor_UPDRS"]:
        if col in df.columns:
            return df[col].astype(float)
    return pd.Series(np.zeros(len(df)), index=df.index)


def main() -> None:
    print("=" * 64)
    print("  PARKINSON'S DISEASE DETECTION — Training Pipeline")
    print("=" * 64)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    # The loader searches: data/parkinsons.csv, data/raw/parkinsons.csv,
    # then falls back to synthetic multi-modal generation.
    # For the telemonitoring file specifically, we also check data/raw directly.
    raw_dir = DATA_DIR / "raw"
    tele_candidates = [
        raw_dir / "parkinsons_telemonitoring.csv",
        raw_dir / "parkinsons_updrs.data",
        DATA_DIR / "parkinsons_telemonitoring.csv",
    ]

    df = None
    for cand in tele_candidates:
        if cand.exists():
            print(f"      Found telemonitoring file: {cand.name}")
            try:
                from data_loader import load_uci_telemonitoring
                df = load_uci_telemonitoring(cand)
                # Telemonitoring has no healthy controls; augment columns
                from data_loader import _augment_uci_with_synthetic
                df = _augment_uci_with_synthetic(df)
                print(f"      Telemonitoring loaded: {len(df):,} recordings")
                break
            except Exception as exc:
                print(f"      Telemonitoring load failed: {exc}")
                df = None

    if df is None:
        try:
            df = load_data(str(raw_dir))
            print(f"      Data loaded via load_data(): {len(df):,} rows")
        except Exception as exc:
            print(f"      ERROR loading data: {exc}")
            sys.exit(1)

    if len(df) > MAX_ROWS:
        print(f"      Sampling to {MAX_ROWS:,} rows for speed.")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    print(f"      Shape: {df.shape} | Columns (first 10): {list(df.columns[:10])}")

    # ── 2. Build features ──────────────────────────────────────────────────────
    print("\n[2/7] Building multi-modal features...")
    try:
        X_features = build_multimodal_features(df)
        print(f"      Feature matrix: {X_features.shape[0]:,} rows × {X_features.shape[1]} features")
    except Exception as exc:
        print(f"      ERROR in build_multimodal_features: {exc}")
        sys.exit(1)

    # ── 3. Build targets ───────────────────────────────────────────────────────
    print("\n[3/7] Building targets...")
    try:
        y_binary = _build_binary_target(df)
        print(f"      Binary target (PD=1): {y_binary.sum():,} positive / {len(y_binary):,} total "
              f"({y_binary.mean():.1%} PD rate)")
    except Exception as exc:
        print(f"      ERROR building binary target: {exc}")
        sys.exit(1)

    y_updrs = _get_updrs_target(df)
    print(f"      UPDRS target: mean={y_updrs.mean():.1f}, std={y_updrs.std():.1f}")

    # Feature column names
    feature_names = list(X_features.columns)

    # ── 4. Subject-level train/test split ──────────────────────────────────────
    print("\n[4/7] Subject-level train/test split...")
    subject_col = "subject_id"
    if subject_col in df.columns:
        try:
            from data_loader import create_subject_splits
            df_train_raw, df_test_raw = create_subject_splits(df, subject_col=subject_col, test_size=0.2)
            train_idx = df_train_raw.index
            test_idx = df_test_raw.index
            X_train = X_features.iloc[:len(df_train_raw)].copy()
            X_test = X_features.iloc[len(df_train_raw):].copy()
            y_train_bin = y_binary.iloc[:len(df_train_raw)]
            y_test_bin = y_binary.iloc[len(df_train_raw):]
            y_train_updrs = y_updrs.iloc[:len(df_train_raw)]
            y_test_updrs = y_updrs.iloc[len(df_train_raw):]
            print(f"      Subject split: {len(X_train):,} train / {len(X_test):,} test recordings")
        except Exception as exc:
            print(f"      Subject split failed ({exc}), using random split.")
            X_train, X_test, y_train_bin, y_test_bin, y_train_updrs, y_test_updrs = (
                *train_test_split(X_features, y_binary, test_size=0.2, random_state=42, stratify=y_binary),
                None, None,
            )
            y_train_updrs, y_test_updrs = train_test_split(y_updrs, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train_bin, y_test_bin = train_test_split(
            X_features, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        y_train_updrs, y_test_updrs = train_test_split(y_updrs, test_size=0.2, random_state=42)
        print(f"      Random split: {len(X_train):,} train / {len(X_test):,} test")

    # ── 5. Train binary classifier ─────────────────────────────────────────────
    print("\n[5/7] Training binary PD classifier (XGBoost)...")
    try:
        classifier = train_binary_classifier(X_train, y_train_bin, model_type="xgboost")
        print("      Classifier training complete.")
    except Exception as exc:
        print(f"      XGBoost failed ({exc}), trying random forest...")
        try:
            classifier = train_binary_classifier(X_train, y_train_bin, model_type="random_forest")
        except Exception as exc2:
            print(f"      All classifiers failed: {exc2}")
            sys.exit(1)

    # ── 6. Train UPDRS regressor ───────────────────────────────────────────────
    print("\n[6/7] Training UPDRS severity regressor...")
    try:
        regressor = train_updrs_regressor(X_train, y_train_updrs)
        print("      Regressor training complete.")
    except Exception as exc:
        print(f"      UPDRS regressor failed: {exc}")
        regressor = None

    # ── 7. Subject-level cross-validation ────────────────────────────────────
    print("\n[7/7] Subject-level cross-validation...")
    cv_results = {}
    if subject_col in df.columns:
        df_cv = df.copy()
        df_cv = df_cv.iloc[:len(X_features)].reset_index(drop=True)
        for col in feature_names:
            if col not in df_cv.columns:
                df_cv[col] = X_features[col].values
        df_cv["_binary_target"] = y_binary.values
        df_cv["_updrs_target"] = y_updrs.values

        print("      Cross-validating classifier (GroupKFold)...")
        try:
            cv_results = subject_level_cross_validate(
                model_type="xgboost",
                df=df_cv,
                subject_col=subject_col,
                target_col="_binary_target",
                feature_cols=feature_names,
                n_folds=5,
                task="classification",
            )
            auc_mean = cv_results.get("auc_roc_mean", 0)
            auc_std = cv_results.get("auc_roc_std", 0)
            print(f"      CV AUC-ROC: {auc_mean:.4f} ± {auc_std:.4f}")
        except Exception as exc:
            print(f"      Cross-validation failed: {exc}")
    else:
        print("      Skipped (no subject_id column).")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n      Evaluating classifier on held-out test set...")
    clf_metrics = {}
    try:
        clf_metrics = evaluate_classifier(classifier, X_test, y_test_bin)
        print(f"      AUC-ROC     : {clf_metrics.get('auc_roc', 0):.4f}")
        print(f"      Sensitivity : {clf_metrics.get('sensitivity', 0):.4f}")
        print(f"      Specificity : {clf_metrics.get('specificity', 0):.4f}")
        print(f"      F1          : {clf_metrics.get('f1', 0):.4f}")
    except Exception as exc:
        print(f"      Classifier evaluation failed: {exc}")

    reg_metrics = {}
    if regressor is not None:
        print("\n      Evaluating UPDRS regressor...")
        try:
            reg_metrics = evaluate_regressor(regressor, X_test, y_test_updrs)
            print(f"      RMSE        : {reg_metrics.get('rmse', 0):.3f}")
            print(f"      MAE         : {reg_metrics.get('mae', 0):.3f}")
            print(f"      R²          : {reg_metrics.get('r2', 0):.4f}")
        except Exception as exc:
            print(f"      Regressor evaluation failed: {exc}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n      Saving artifacts...")
    try:
        save_artifacts(
            classif_model=classifier,
            regress_model=regressor if regressor is not None else classifier,
            feature_cols=feature_names,
            model_dir=str(MODEL_DIR),
        )
        print(f"      Artifacts saved to {MODEL_DIR}")
    except Exception as exc:
        print(f"      save_artifacts failed: {exc}")

    # Write metrics.json
    metrics_payload = {
        "project": "Parkinson's Disease Detection",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "pd_rate_train": float(round(y_train_bin.mean(), 4)),
        "pd_rate_test": float(round(y_test_bin.mean(), 4)),
        "classifier": {k: float(v) for k, v in clf_metrics.items()},
        "regressor": {k: float(v) for k, v in reg_metrics.items()},
        "cv_auc_roc_mean": float(cv_results.get("auc_roc_mean", 0)),
        "cv_auc_roc_std": float(cv_results.get("auc_roc_std", 0)),
    }

    metrics_path = MODEL_DIR / "metrics.json"
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2, default=str)
        print(f"      metrics.json saved to {metrics_path}")
    except Exception as exc:
        print(f"      metrics.json save failed: {exc}")

    # ── Summary box ────────────────────────────────────────────────────────────
    print()
    _print_box(
        [
            "PARKINSON'S DETECTION — Training Summary",
            "",
            f"  Dataset      : {len(X_features):,} rows | {X_features.shape[1]} features",
            f"  PD rate      : {y_binary.mean():.1%}",
            f"  AUC-ROC      : {clf_metrics.get('auc_roc', 0):.4f}",
            f"  Sensitivity  : {clf_metrics.get('sensitivity', 0):.4f}",
            f"  Specificity  : {clf_metrics.get('specificity', 0):.4f}",
            f"  UPDRS RMSE   : {reg_metrics.get('rmse', 0):.3f}",
            f"  CV AUC-ROC   : {cv_results.get('auc_roc_mean', 0):.4f} "
            f"± {cv_results.get('auc_roc_std', 0):.4f}",
            f"  Artifacts    : {MODEL_DIR}",
        ]
    )


if __name__ == "__main__":
    main()

"""
train.py — Loan Approval Prediction
====================================
End-to-end training entrypoint.

Usage:
    py -3.11 src/train.py

Pipeline:
    1. Load HMDA data (real or synthetic fallback)
    2. Engineer features
    3. Train LightGBM + XGBoost + Logistic Regression
    4. Evaluate all models + fairness summary
    5. Save artifacts + metrics.json to data/models/
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
from data_loader import load_data
from features import build_feature_pipeline
from models import (
    train_lightgbm,
    train_xgboost,
    train_logistic_regression,
    evaluate_all_models,
    save_artifacts,
)

MAX_ROWS = 200_000


def _print_box(lines: list[str], width: int = 64) -> None:
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    for line in lines:
        print(f"| {line:<{width - 4}} |")
    print(border)


def _fairness_summary(
    models_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    protected_df: pd.DataFrame,
) -> dict:
    """Compute approval-rate parity across demographic groups (sex, race)."""
    results = {}
    for model_name, model in models_dict.items():
        try:
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            prot = protected_df.copy()
            prot["predicted"] = y_pred
            prot["actual"] = y_test.values

            row = {}
            for col in ["applicant_sex", "applicant_race_1", "applicant_age_group"]:
                if col in prot.columns:
                    group_rates = prot.groupby(col)["predicted"].mean()
                    row[f"{col}_approval_rates"] = group_rates.round(3).to_dict()
            results[model_name] = row
        except Exception as exc:
            results[model_name] = {"error": str(exc)}
    return results


def main() -> None:
    print("=" * 64)
    print("  LOAN APPROVAL PREDICTION — Training Pipeline")
    print("=" * 64)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    try:
        df = load_data(str(DATA_DIR))
        print(f"      Loaded {len(df):,} rows.")
    except Exception as exc:
        print(f"      ERROR loading data: {exc}")
        sys.exit(1)

    # Cap at MAX_ROWS
    if len(df) > MAX_ROWS:
        print(f"      Sampling down to {MAX_ROWS:,} rows for speed.")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n[2/5] Engineering features...")
    try:
        X, y, feature_names, protected_cols = build_feature_pipeline(df)
        print(f"      Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"      Approval rate  : {y.mean():.1%}")
        print(f"      Protected cols : {protected_cols}")
    except Exception as exc:
        print(f"      ERROR in feature pipeline: {exc}")
        sys.exit(1)

    # Preserve protected attribute columns from original df for fairness analysis
    protected_df = pd.DataFrame(index=X.index)
    for col in protected_cols:
        if col in df.columns:
            # Align by positional index
            protected_df[col] = df[col].values[: len(X)]

    # ── 3. Train/test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(
        f"\n[3/5] Train/test split: "
        f"{len(X_train):,} train / {len(X_test):,} test"
    )

    # Align protected columns to test set
    prot_test = protected_df.loc[X_test.index].reset_index(drop=True)

    # ── 4. Train models ────────────────────────────────────────────────────────
    print("\n[4/5] Training models...")
    trained_models = {}

    print("      Training LightGBM...")
    try:
        lgb_model = train_lightgbm(X_train, y_train)
        trained_models["LightGBM"] = lgb_model
        print("      LightGBM: done.")
    except Exception as exc:
        print(f"      LightGBM FAILED: {exc}")

    print("      Training XGBoost...")
    try:
        xgb_model = train_xgboost(X_train, y_train)
        trained_models["XGBoost"] = xgb_model
        print("      XGBoost: done.")
    except Exception as exc:
        print(f"      XGBoost FAILED: {exc}")

    print("      Training Logistic Regression...")
    try:
        lr_model = train_logistic_regression(X_train, y_train)
        trained_models["LogisticRegression"] = lr_model
        print("      Logistic Regression: done.")
    except Exception as exc:
        print(f"      Logistic Regression FAILED: {exc}")

    if not trained_models:
        print("      ERROR: no models trained successfully. Exiting.")
        sys.exit(1)

    # ── 5. Evaluate all models ─────────────────────────────────────────────────
    print("\n[5/5] Evaluating models...")
    try:
        eval_df = evaluate_all_models(trained_models, X_test, y_test)
        print(eval_df[["model", "roc_auc", "avg_precision", "f1", "approval_rate"]].to_string(index=False))
    except Exception as exc:
        print(f"      ERROR in evaluation: {exc}")
        eval_df = pd.DataFrame()

    # Fairness analysis
    print("\n      Computing fairness metrics...")
    try:
        fairness = _fairness_summary(trained_models, X_test, y_test, prot_test)
    except Exception as exc:
        print(f"      Fairness analysis failed: {exc}")
        fairness = {}

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n      Saving model artifacts...")
    best_model_name = "LightGBM"
    if not eval_df.empty and "roc_auc" in eval_df.columns:
        best_model_name = eval_df.iloc[0]["model"]

    best_model = trained_models.get(best_model_name, list(trained_models.values())[0])

    try:
        save_artifacts(
            model=best_model,
            feature_cols=feature_names,
            threshold=0.5,
            model_dir=str(MODEL_DIR),
            model_name="lightgbm",
        )
        print(f"      Best model ({best_model_name}) saved.")
    except Exception as exc:
        print(f"      save_artifacts failed: {exc}")

    # Write metrics.json
    best_row = {}
    if not eval_df.empty:
        best_row = eval_df[eval_df["model"] == best_model_name].to_dict("records")
        best_row = best_row[0] if best_row else eval_df.iloc[0].to_dict()

    metrics_payload = {
        "project": "Loan Approval Prediction",
        "best_model": best_model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "approval_rate_actual": float(round(y_test.mean(), 4)),
        "roc_auc": float(best_row.get("roc_auc", 0.0)),
        "avg_precision": float(best_row.get("avg_precision", 0.0)),
        "f1": float(best_row.get("f1", 0.0)),
        "brier_score": float(best_row.get("brier_score", 0.0)),
        "all_models": eval_df.to_dict("records") if not eval_df.empty else [],
        "fairness": fairness,
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
            "LOAN APPROVAL PREDICTION — Training Summary",
            "",
            f"  Dataset      : {len(X):,} rows | {X.shape[1]} features",
            f"  Best model   : {best_model_name}",
            f"  AUC-ROC      : {metrics_payload['roc_auc']:.4f}",
            f"  Avg Precision: {metrics_payload['avg_precision']:.4f}",
            f"  F1 Score     : {metrics_payload['f1']:.4f}",
            f"  Protected    : {protected_cols}",
            f"  Artifacts    : {MODEL_DIR}",
        ]
    )


if __name__ == "__main__":
    main()

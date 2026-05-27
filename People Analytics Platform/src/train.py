"""
train.py — Employee Attrition Prediction
==========================================
End-to-end training entrypoint.

Usage:
    py -3.11 src/train.py

Pipeline:
    1. Load IBM HR data (real or synthetic), extended to 50k
    2. Engineer features (satisfaction, tenure, compensation, ELV)
    3. Train XGBoost / LightGBM / RandomForest / LogisticRegression
    4. Select best model, evaluate
    5. Compute Employee Lifetime Value + segment employees
    6. Save model + ELV results + metrics.json to data/models/
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
    train_all_models,
    select_best_model,
    evaluate_model,
    save_artifacts,
)
from elv_model import compute_elv, segment_employees

MAX_ROWS = 100_000


def _print_box(lines: list[str], width: int = 64) -> None:
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    for line in lines:
        print(f"| {line:<{width - 4}} |")
    print(border)


def main() -> None:
    print("=" * 64)
    print("  EMPLOYEE ATTRITION PREDICTION — Training Pipeline")
    print("=" * 64)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading employee data...")
    try:
        df = load_data(str(DATA_DIR), target_n=50_000)
        print(f"      Loaded {len(df):,} employees.")
        print(f"      Attrition rate: {df['Attrition'].mean():.1%}")
    except Exception as exc:
        print(f"      ERROR loading data: {exc}")
        sys.exit(1)

    if len(df) > MAX_ROWS:
        print(f"      Sampling down to {MAX_ROWS:,} rows for speed.")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n[2/6] Engineering features...")
    try:
        X, y, feature_names, protected_cols = build_feature_pipeline(df)
        print(f"      Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"      Attrition rate : {y.mean():.1%}")
        print(f"      Protected cols : {protected_cols}")
    except Exception as exc:
        print(f"      ERROR in feature pipeline: {exc}")
        sys.exit(1)

    # ── 3. Train/test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )
    print(
        f"\n[3/6] Train/test split: "
        f"{len(X_train):,} train / {len(X_test):,} test"
    )

    # ── 4. Train models ────────────────────────────────────────────────────────
    print("\n[4/6] Training all models (XGBoost, LightGBM, RandomForest, LR)...")
    try:
        trained_models = train_all_models(X_train, y_train)
        print(f"      Trained: {list(trained_models.keys())}")
    except Exception as exc:
        print(f"      ERROR training models: {exc}")
        sys.exit(1)

    # ── 5. Select best model & evaluate ───────────────────────────────────────
    print("\n[5/6] Selecting best model and evaluating...")
    try:
        best_model_name, best_model = select_best_model(trained_models, X_test, y_test)
        print(f"      Best model: {best_model_name}")
    except Exception as exc:
        print(f"      select_best_model failed: {exc}")
        best_model_name = list(trained_models.keys())[0]
        best_model = trained_models[best_model_name]

    try:
        metrics = evaluate_model(best_model, X_test, y_test)
        print(f"      AUC-ROC     : {metrics.get('roc_auc', 0):.4f}")
        print(f"      F1 (binary) : {metrics.get('f1_binary', 0):.4f}")
        print(f"      Recall      : {metrics.get('recall', 0):.4f}")
    except Exception as exc:
        print(f"      evaluate_model failed: {exc}")
        metrics = {}

    # ── 6. Employee Lifetime Value + Segmentation ──────────────────────────────
    print("\n[6/6] Computing Employee Lifetime Value and segmentation...")
    df_elv = df.copy()
    try:
        df_elv = compute_elv(df_elv)
        print(
            f"      ELV computed: "
            f"mean=${df_elv['elv'].mean():,.0f}, "
            f"median=${df_elv['elv'].median():,.0f}"
        )
    except Exception as exc:
        print(f"      compute_elv failed: {exc}")

    attrition_proba = np.zeros(len(df_elv))
    try:
        X_all_clean = X.fillna(0).astype(float)
        attrition_proba = best_model.predict_proba(X_all_clean)[:, 1]
    except Exception as exc:
        print(f"      Could not generate full-dataset probabilities: {exc}")

    try:
        df_segmented = segment_employees(df_elv, attrition_proba)
        segment_counts = df_segmented["segment"].value_counts().to_dict()
        print(f"      Segments: {segment_counts}")
    except Exception as exc:
        print(f"      segment_employees failed: {exc}")
        df_segmented = df_elv.copy()
        df_segmented["attrition_probability"] = attrition_proba[: len(df_elv)]

    # Top 5 flight-risk employees
    flight_risk_top5 = []
    try:
        df_segmented["attrition_proba"] = attrition_proba[: len(df_segmented)]
        top5 = (
            df_segmented.sort_values("attrition_proba", ascending=False)
            .head(5)[["EmployeeNumber", "JobRole", "attrition_proba", "segment"]]
            if "EmployeeNumber" in df_segmented.columns
            else df_segmented.sort_values("attrition_proba", ascending=False).head(5)
        )
        flight_risk_top5 = top5.to_dict("records")
        print("\n      Top 5 Flight Risk Employees:")
        for row in flight_risk_top5:
            emp_id = row.get("EmployeeNumber", "?")
            role = row.get("JobRole", "?")
            prob = row.get("attrition_proba", 0)
            seg = row.get("segment", "?")
            print(f"        Emp #{emp_id:>5} | {role:<30} | P(attr)={prob:.3f} | {seg}")
    except Exception as exc:
        print(f"      Top-5 extraction failed: {exc}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n      Saving artifacts...")
    try:
        save_artifacts(
            model=best_model,
            feature_cols=feature_names,
            model_dir=str(MODEL_DIR),
            model_name="best_model",
        )
        print(f"      Model saved: {MODEL_DIR / 'best_model.pkl'}")
    except Exception as exc:
        print(f"      save_artifacts failed: {exc}")

    # Save ELV results CSV
    try:
        elv_path = MODEL_DIR / "elv_results.csv"
        save_cols = [
            c for c in ["EmployeeNumber", "JobRole", "MonthlyIncome", "attrition_proba",
                        "elv", "replacement_cost", "segment"]
            if c in df_segmented.columns
        ]
        df_segmented[save_cols].to_csv(elv_path, index=False)
        print(f"      ELV results saved: {elv_path}")
    except Exception as exc:
        print(f"      ELV CSV save failed: {exc}")

    # Write metrics.json
    metrics_serializable = {
        k: v for k, v in metrics.items() if not isinstance(v, str) or k != "classification_report"
    }
    metrics_payload = {
        "project": "Employee Attrition Prediction",
        "best_model": best_model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "attrition_rate_actual": float(round(y_test.mean(), 4)),
        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
           for k, v in metrics_serializable.items()},
        "elv_mean": float(round(df_elv["elv"].mean(), 2)) if "elv" in df_elv.columns else None,
        "elv_median": float(round(df_elv["elv"].median(), 2)) if "elv" in df_elv.columns else None,
        "segment_distribution": df_segmented["segment"].value_counts().to_dict() if "segment" in df_segmented.columns else {},
        "top_flight_risk": flight_risk_top5,
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
            "EMPLOYEE ATTRITION — Training Summary",
            "",
            f"  Dataset      : {len(X):,} rows | {X.shape[1]} features",
            f"  Best model   : {best_model_name}",
            f"  AUC-ROC      : {metrics.get('roc_auc', 0):.4f}",
            f"  F1 (binary)  : {metrics.get('f1_binary', 0):.4f}",
            f"  Recall       : {metrics.get('recall', 0):.4f}",
            f"  Avg Precision: {metrics.get('avg_precision', 0):.4f}",
            f"  ELV mean     : ${df_elv['elv'].mean():,.0f}" if "elv" in df_elv.columns else "  ELV mean     : N/A",
            f"  Artifacts    : {MODEL_DIR}",
        ]
    )


if __name__ == "__main__":
    main()

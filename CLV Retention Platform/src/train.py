"""
train.py — Customer Churn Prediction
======================================
End-to-end training entrypoint.

Usage:
    py -3.11 src/train.py

Pipeline:
    1. Load data (KKBox → Telco → CSV → synthetic fallback)
    2. Build RFM + engagement + payment + lifecycle features
    3. Train churn classifier (XGBoost / LightGBM / LR ensemble)
    4. Evaluate: AUC-PR, F1
    5. Compute simple CLV + identify at-risk revenue
    6. Fit Kaplan-Meier survival curve
    7. Save all artifacts + metrics.json to data/models/
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
    train_churn_classifier,
    evaluate_model,
    save_artifacts,
)
from clv_model import compute_simple_clv
from survival_analysis import prepare_survival_data, fit_kaplan_meier

MAX_ROWS = 100_000
AVG_MONTHLY_REVENUE = 15.0  # default if no plan price available


def _print_box(lines: list[str], width: int = 66) -> None:
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    for line in lines:
        print(f"| {line:<{width - 4}} |")
    print(border)


def main() -> None:
    print("=" * 66)
    print("  CUSTOMER CHURN PREDICTION — Training Pipeline")
    print("=" * 66)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    try:
        df = load_data(str(DATA_DIR / "raw"), str(PROJECT_ROOT))
        print(f"      Loaded {len(df):,} customers.")
        if "churn_flag" in df.columns:
            print(f"      Churn rate: {df['churn_flag'].mean():.1%}")
    except Exception as exc:
        print(f"      ERROR loading data: {exc}")
        sys.exit(1)

    if len(df) > MAX_ROWS:
        print(f"      Sampling to {MAX_ROWS:,} rows for speed.")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n[2/7] Engineering features...")
    try:
        X, y, feature_names = build_feature_pipeline(df)
        print(f"      Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"      Churn rate    : {y.mean():.1%}")
    except Exception as exc:
        print(f"      ERROR in feature pipeline: {exc}")
        sys.exit(1)

    # ── 3. Train/val/test split ────────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
    )
    print(
        f"\n[3/7] Train/val/test split: "
        f"{len(X_train):,} / {len(X_val):,} / {len(X_test):,}"
    )

    # ── 4. Train churn classifier ──────────────────────────────────────────────
    print("\n[4/7] Training churn classifier...")
    try:
        best_model, model_comparison = train_churn_classifier(
            X_train, y_train, eval_set=(X_val, y_val)
        )
        print(f"      Model comparison (AUC-PR on val set):")
        for mname, mstats in model_comparison.items():
            auc_pr = mstats.get("auc_pr", "?")
            print(f"        {mname:<25} AUC-PR={auc_pr}")
    except Exception as exc:
        print(f"      ERROR training classifier: {exc}")
        sys.exit(1)

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    print("\n[5/7] Evaluating on test set...")
    try:
        metrics = evaluate_model(best_model, X_test, y_test)
        print(f"      AUC-ROC  : {metrics.get('auc_roc', 0):.4f}")
        print(f"      AUC-PR   : {metrics.get('auc_pr', 0):.4f}")
        print(f"      F1       : {metrics.get('f1', 0):.4f}")
        print(f"      Precision: {metrics.get('precision', 0):.4f}")
        print(f"      Recall   : {metrics.get('recall', 0):.4f}")
    except Exception as exc:
        print(f"      Evaluation failed: {exc}")
        metrics = {}

    # ── 6. CLV computation ─────────────────────────────────────────────────────
    print("\n[6/7] Computing Customer Lifetime Value...")

    # Get churn probabilities for the full dataset
    X_all_clean = X.fillna(0).replace([float("inf"), float("-inf")], 0).astype(float)
    churn_proba_series = pd.Series(np.zeros(len(X_all_clean)), index=X_all_clean.index)
    try:
        churn_proba_arr = best_model.predict_proba(X_all_clean)[:, 1]
        churn_proba_series = pd.Series(churn_proba_arr, index=X_all_clean.index)
    except Exception as exc:
        print(f"      Could not compute full-dataset probabilities: {exc}")

    try:
        # Determine average monthly revenue from data if available
        if "plan_list_price" in df.columns:
            avg_rev = float(df["plan_list_price"].median() / 30 * 30)
        elif "monthly_charges" in df.columns:
            avg_rev = float(df["monthly_charges"].median())
        else:
            avg_rev = AVG_MONTHLY_REVENUE

        clv_series = compute_simple_clv(
            df=df,
            avg_monthly_revenue=avg_rev,
            churn_proba=churn_proba_series,
        )
        avg_clv = float(clv_series.mean())
        median_clv = float(clv_series.median())

        # At-risk revenue: customers in top 20% churn probability
        n_at_risk = max(int(len(df) * 0.20), 1)
        top_risk_idx = churn_proba_series.nlargest(n_at_risk).index
        at_risk_revenue = float(clv_series.loc[top_risk_idx].sum())

        print(f"      Avg CLV         : ${avg_clv:,.2f}")
        print(f"      Median CLV      : ${median_clv:,.2f}")
        print(f"      At-risk revenue : ${at_risk_revenue:,.2f} (top 20% churn risk)")
        print(f"      Avg rev/month   : ${avg_rev:.2f}")

        # Save CLV results
        clv_df = df[["user_id"]].copy() if "user_id" in df.columns else pd.DataFrame(index=df.index)
        clv_df["churn_proba"] = churn_proba_series.values
        clv_df["clv"] = clv_series.values
        clv_path = MODEL_DIR / "clv_results.csv"
        clv_df.to_csv(clv_path, index=False)
        print(f"      CLV results saved: {clv_path}")
    except Exception as exc:
        print(f"      CLV computation failed: {exc}")
        clv_series = pd.Series(dtype=float)
        avg_clv = 0.0
        at_risk_revenue = 0.0

    # ── 7. Survival analysis ───────────────────────────────────────────────────
    print("\n[7/7] Fitting Kaplan-Meier survival curve...")
    km_results = {}
    median_survival = 0.0
    try:
        surv_df = prepare_survival_data(df, churn_col="churn_flag")
        km_results = fit_kaplan_meier(surv_df["T"], surv_df["E"])
        median_survival = km_results["all"].get("median_survival", 0.0)
        n_events = km_results["all"].get("event_count", 0)
        print(f"      Median survival time : {median_survival:.1f} days")
        print(f"      Total churn events   : {n_events:,}")
    except Exception as exc:
        print(f"      Survival analysis failed: {exc}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n      Saving artifacts...")
    try:
        save_artifacts(
            churn_model=best_model,
            clv_artifacts={"clv_mean": avg_clv, "avg_monthly_revenue": AVG_MONTHLY_REVENUE},
            survival_model=km_results,
            model_dir=str(MODEL_DIR),
        )
        print(f"      Artifacts saved to {MODEL_DIR}")
    except Exception as exc:
        print(f"      save_artifacts failed: {exc}")

    # Write metrics.json
    metrics_payload = {
        "project": "Customer Churn Prediction",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "churn_rate_actual": float(round(y_test.mean(), 4)),
        "auc_roc": float(metrics.get("auc_roc", 0)),
        "auc_pr": float(metrics.get("auc_pr", 0)),
        "f1": float(metrics.get("f1", 0)),
        "precision": float(metrics.get("precision", 0)),
        "recall": float(metrics.get("recall", 0)),
        "avg_clv": round(avg_clv, 2),
        "at_risk_revenue_top20pct": round(at_risk_revenue, 2),
        "avg_monthly_revenue": AVG_MONTHLY_REVENUE,
        "median_survival_days": round(median_survival, 1),
        "model_comparison": model_comparison,
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
            "CUSTOMER CHURN — Training Summary",
            "",
            f"  Dataset       : {len(X):,} rows | {X.shape[1]} features",
            f"  AUC-PR        : {metrics.get('auc_pr', 0):.4f}",
            f"  AUC-ROC       : {metrics.get('auc_roc', 0):.4f}",
            f"  F1 Score      : {metrics.get('f1', 0):.4f}",
            f"  Avg CLV       : ${avg_clv:,.2f}",
            f"  At-Risk Rev   : ${at_risk_revenue:,.2f}",
            f"  Median Surv.  : {median_survival:.1f} days",
            f"  Artifacts     : {MODEL_DIR}",
        ]
    )


if __name__ == "__main__":
    main()

"""
train.py — Bankruptcy / Financial Distress Prediction
=======================================================
End-to-end training entrypoint.

Usage:
    py -3.11 src/train.py

Pipeline:
    1. Load Financial Distress dataset (real CSV or synthetic fallback)
    2. Build feature pipeline (ratios, Altman Z-score, trends, peer-relative)
    3. Train XGBoost + LightGBM distress classifiers
    4. Evaluate: AUC-ROC, Precision@50, best model selection
    5. Generate supply-chain network + contagion risk scores
    6. Save artifacts + metrics.json to data/models/
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
    train_xgboost_distress,
    train_lightgbm_distress,
    evaluate_model,
    save_artifacts,
)
from network_analysis import generate_supply_chain_network, compute_contagion_risk

MAX_ROWS = 100_000


def _print_box(lines: list[str], width: int = 64) -> None:
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    for line in lines:
        print(f"| {line:<{width - 4}} |")
    print(border)


def _select_best(
    models_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, object, dict]:
    """Select the best model by AUC-ROC on the test set."""
    best_name, best_model, best_metrics = None, None, {}
    best_auc = -1.0
    for name, model in models_dict.items():
        try:
            m = evaluate_model(model, X_test, y_test, k=50)
            print(f"        {name}: AUC={m['auc_roc']:.4f}, P@50={m.get('precision_at_50', 0):.4f}")
            if m["auc_roc"] > best_auc:
                best_auc = m["auc_roc"]
                best_name = name
                best_model = model
                best_metrics = m
        except Exception as exc:
            print(f"        {name} evaluation failed: {exc}")
    return best_name, best_model, best_metrics


def main() -> None:
    print("=" * 64)
    print("  BANKRUPTCY / FINANCIAL DISTRESS — Training Pipeline")
    print("=" * 64)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading financial distress data...")
    # data_loader.py searches: data/financial_distress.csv, data/bankruptcy.csv,
    # data/raw/financial_distress.csv, data/raw/bankruptcy.csv
    # Also check for the Kaggle "Financial Distress.csv" name
    raw_dir = DATA_DIR / "raw"
    extra_candidates = [
        raw_dir / "Financial Distress.csv",
        DATA_DIR / "Financial Distress.csv",
    ]
    for cand in extra_candidates:
        if cand.exists() and not (raw_dir / "financial_distress.csv").exists():
            import shutil
            dest = raw_dir / "financial_distress.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(cand), str(dest))
            print(f"      Copied '{cand.name}' -> financial_distress.csv for loader.")
            break

    try:
        df = load_data(str(DATA_DIR))
        print(f"      Loaded {len(df):,} rows.")
        if "distress_flag" in df.columns:
            print(f"      Distress rate: {df['distress_flag'].mean():.2%}")
    except Exception as exc:
        print(f"      ERROR loading data: {exc}")
        sys.exit(1)

    if len(df) > MAX_ROWS:
        print(f"      Sampling to {MAX_ROWS:,} rows for speed.")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    # ── 2. Feature pipeline ────────────────────────────────────────────────────
    print("\n[2/6] Building feature pipeline...")
    try:
        X, y_dict, feature_names = build_feature_pipeline(df)
        print(f"      Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"      Target horizons: {list(y_dict.keys())}")
    except Exception as exc:
        print(f"      ERROR in feature pipeline: {exc}")
        sys.exit(1)

    # Use the 12-month horizon as primary target (most useful for early warning)
    primary_target = "distress_12m"
    if primary_target not in y_dict:
        primary_target = list(y_dict.keys())[0]
    y = y_dict[primary_target]
    print(f"      Using target: {primary_target} | Distress rate: {y.mean():.2%}")

    # ── 3. Train/test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(
        f"\n[3/6] Train/test split: "
        f"{len(X_train):,} train / {len(X_test):,} test"
    )

    # ── 4. Train models ────────────────────────────────────────────────────────
    print("\n[4/6] Training distress models...")
    trained_models = {}

    print("      Training XGBoost...")
    try:
        xgb_model = train_xgboost_distress(X_train, y_train)
        trained_models["XGBoost"] = xgb_model
        print("      XGBoost: done.")
    except Exception as exc:
        print(f"      XGBoost FAILED: {exc}")

    print("      Training LightGBM...")
    try:
        lgb_model = train_lightgbm_distress(X_train, y_train)
        trained_models["LightGBM"] = lgb_model
        print("      LightGBM: done.")
    except Exception as exc:
        print(f"      LightGBM FAILED: {exc}")

    if not trained_models:
        print("      ERROR: no models trained. Exiting.")
        sys.exit(1)

    # ── 5. Evaluate & select best model ───────────────────────────────────────
    print("\n[5/6] Evaluating models...")
    best_name, best_model, best_metrics = _select_best(trained_models, X_test, y_test)
    print(f"\n      Best model   : {best_name}")
    print(f"      AUC-ROC      : {best_metrics.get('auc_roc', 0):.4f}")
    print(f"      Precision@50 : {best_metrics.get('precision_at_50', 0):.4f}")
    print(f"      F1           : {best_metrics.get('f1', 0):.4f}")

    # ── 6. Network analysis ────────────────────────────────────────────────────
    print("\n[6/6] Supply chain network analysis...")
    contagion_df = pd.DataFrame()
    try:
        G = generate_supply_chain_network(df)
        print(f"      Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as exc:
        print(f"      generate_supply_chain_network failed: {exc}")
        G = None

    if G is not None:
        try:
            # Build distress probability series indexed by company_id
            proba_arr = best_model.predict_proba(X.fillna(X.median(numeric_only=True)))[:, 1]
            if "company_id" in df.columns:
                # Use latest row per company for probability
                df_temp = df.copy()
                df_temp["_proba"] = proba_arr
                latest = df_temp.sort_values("year").groupby("company_id")["_proba"].last() \
                    if "year" in df_temp.columns else df_temp.groupby("company_id")["_proba"].mean()
                distress_proba = latest
            else:
                distress_proba = pd.Series(proba_arr, index=range(len(proba_arr)))

            contagion_df = compute_contagion_risk(G, df, distress_proba)
            mean_risk = contagion_df["total_contagion_risk"].mean()
            high_risk = (contagion_df["total_contagion_risk"] > 0.5).sum()
            print(f"      Contagion risk: mean={mean_risk:.3f}, high-risk nodes={high_risk}")
        except Exception as exc:
            print(f"      compute_contagion_risk failed: {exc}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n      Saving artifacts...")
    try:
        save_artifacts(
            models_dict=trained_models,
            feature_cols=feature_names,
            model_dir=str(MODEL_DIR),
        )
        print(f"      Models saved to {MODEL_DIR}")
    except Exception as exc:
        print(f"      save_artifacts failed: {exc}")

    if not contagion_df.empty:
        try:
            contagion_path = MODEL_DIR / "contagion_risk.csv"
            contagion_df.to_csv(contagion_path, index=False)
            print(f"      Contagion risk saved: {contagion_path}")
        except Exception as exc:
            print(f"      Contagion CSV save failed: {exc}")

    # Write metrics.json
    all_model_metrics = {}
    for name, model in trained_models.items():
        try:
            all_model_metrics[name] = evaluate_model(model, X_test, y_test, k=50)
        except Exception:
            pass

    metrics_payload = {
        "project": "Bankruptcy Prediction",
        "best_model": best_name,
        "primary_target": primary_target,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "distress_rate_actual": float(round(y_test.mean(), 4)),
        "auc_roc": float(best_metrics.get("auc_roc", 0)),
        "precision_at_50": float(best_metrics.get("precision_at_50", 0)),
        "average_precision": float(best_metrics.get("average_precision", 0)),
        "f1": float(best_metrics.get("f1", 0)),
        "all_models": {
            name: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in m.items()}
            for name, m in all_model_metrics.items()
        },
        "contagion_mean_risk": float(contagion_df["total_contagion_risk"].mean())
        if not contagion_df.empty else None,
        "contagion_high_risk_count": int((contagion_df["total_contagion_risk"] > 0.5).sum())
        if not contagion_df.empty else None,
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
            "BANKRUPTCY PREDICTION — Training Summary",
            "",
            f"  Dataset      : {len(X):,} rows | {X.shape[1]} features",
            f"  Best model   : {best_name}",
            f"  AUC-ROC      : {best_metrics.get('auc_roc', 0):.4f}",
            f"  Precision@50 : {best_metrics.get('precision_at_50', 0):.4f}",
            f"  F1 Score     : {best_metrics.get('f1', 0):.4f}",
            f"  Target       : {primary_target}",
            f"  Network nodes: {G.number_of_nodes() if G else 'N/A'}",
            f"  Artifacts    : {MODEL_DIR}",
        ]
    )


if __name__ == "__main__":
    main()

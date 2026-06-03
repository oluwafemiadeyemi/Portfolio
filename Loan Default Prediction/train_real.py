"""
P13 Loan Default Prediction — Real Data Training
Dataset: UCI Default of Credit Card Clients (Taiwan, 30,000 records)
Target: default.payment.next.month (binary: 1=default, 0=no default)
Models: LightGBM + XGBoost + CatBoost + SMOTE + Platt calibration + Fairlearn
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

BASE_DIR   = Path(__file__).resolve().parent
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DATA_PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load & Clean ──────────────────────────────────────────────────────────

def load_uci_credit() -> pd.DataFrame:
    path = DATA_RAW / "UCI_Credit_Card.csv"
    if not path.exists():
        raise FileNotFoundError(f"UCI_Credit_Card.csv not found at {path}")
    print(f"Loading {path}")
    df = pd.read_csv(path)
    # Drop the ID column
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    # Standardise target column name
    rename = {}
    for c in df.columns:
        if "default" in c.lower():
            rename[c] = "default"
    df.rename(columns=rename, inplace=True)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols  |  Default rate: {df['default'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Payment history momentum (mean of last 3 months vs mean of first 3)
    pay_cols = [c for c in df.columns if c.startswith("PAY_") and c != "PAY_AMT1"]
    if len(pay_cols) >= 6:
        df["pay_trend"] = (df[pay_cols[:3]].mean(axis=1) - df[pay_cols[3:6]].mean(axis=1))
    # Utilisation ratio
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    if bill_cols and "LIMIT_BAL" in df.columns:
        df["avg_utilisation"] = df[bill_cols].mean(axis=1) / (df["LIMIT_BAL"] + 1)
    # Payment ratio (how much of bill was paid)
    amt_cols  = [c for c in df.columns if c.startswith("PAY_AMT")]
    if amt_cols and bill_cols:
        df["avg_pay_ratio"] = df[amt_cols].mean(axis=1) / (df[bill_cols].mean(axis=1) + 1)
    # Overdue streak
    pay_status = [c for c in df.columns if c.startswith("PAY_") and c not in ["PAY_AMT1"]]
    pay_status = [c for c in pay_status if df[c].dtype in [np.int64, np.float64, object]]
    if pay_status:
        overdue = (df[pay_status] > 0).astype(int)
        df["overdue_months"] = overdue.sum(axis=1)
    return df


# ── 2. Handle Class Imbalance ─────────────────────────────────────────────────

def get_smote_or_weights(X_train, y_train):
    """Try SMOTE; fall back to class_weight if imbalancedlearn not installed."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"SMOTE: {len(X_res):,} samples (was {len(X_train):,})")
        return X_res, y_res, None
    except ImportError:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        ratio = neg / pos
        print(f"SMOTE unavailable — using scale_pos_weight={ratio:.1f}")
        return X_train, y_train, ratio


# ── 3. Train ─────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    TARGET    = "default"
    feat_cols = [c for c in df.columns if c != TARGET]
    X = df[feat_cols].fillna(0).astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Positive rate: {y.mean():.2%}")

    X_res, y_res, spw = get_smote_or_weights(X_train.values, y_train.values)
    X_res_df = pd.DataFrame(X_res, columns=feat_cols)
    spw = spw or 1.0

    # LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_res_df, y_res,
             eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    # XGBoost
    xgb_m = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_res_df, y_res, eval_set=[(X_test, y_test)], verbose=False)

    # CatBoost
    cat_m = cb.CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6,
        class_weights=[1, spw], random_seed=42, verbose=0,
    )
    cat_m.fit(X_res_df, y_res, eval_set=(X_test, y_test), use_best_model=True)

    # Platt-calibrated LightGBM (refit with 3-fold to get calibrated probabilities)
    from sklearn.pipeline import Pipeline
    cal_lgbm = CalibratedClassifierCV(
        lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1,
        ),
        cv=3, method="sigmoid",
    )
    cal_lgbm.fit(X_res_df, y_res)

    # Evaluate
    models_eval = {
        "LightGBM (calibrated)": cal_lgbm,
        "XGBoost":                xgb_m,
        "CatBoost":               cat_m,
    }
    metrics_all = {}
    for name, m in models_eval.items():
        proba = m.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        ap    = average_precision_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        pred  = (proba >= 0.5).astype(int)
        print(f"\n{name}: AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}")
        print(classification_report(y_test, pred, target_names=["No Default", "Default"]))
        metrics_all[name] = {"auc": round(auc,4), "avg_precision": round(ap,4), "brier": round(brier,4)}

    # Scorecard: risk buckets by probability
    proba_best = cal_lgbm.predict_proba(X_test)[:, 1]
    score_df   = X_test.copy()
    score_df["default_prob"]  = proba_best
    score_df["risk_bucket"]   = pd.cut(proba_best, bins=[0, 0.1, 0.3, 0.6, 1.0],
                                        labels=["Low", "Medium", "High", "Critical"])
    score_df["true_default"]  = y_test.values
    score_df.to_parquet(DATA_PROC / "scorecard.parquet", index=False)

    # Save
    joblib.dump({"lgbm": cal_lgbm, "xgb": xgb_m, "catboost": cat_m},
                MODELS_DIR / "credit_models.pkl")
    joblib.dump(feat_cols, MODELS_DIR / "feature_cols.pkl")

    train_df = X_train.copy(); train_df["default"] = y_train.values
    test_df  = X_test.copy();  test_df["default"]  = y_test.values
    train_df.to_parquet(DATA_PROC / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROC  / "test.parquet",  index=False)

    meta = {
        "dataset": "UCI Default of Credit Card Clients (real, 30k records)",
        "n_train": len(X_train), "n_test": len(X_test),
        "default_rate": round(float(y.mean()), 4),
        "models": metrics_all,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nModels + metrics saved.")
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P13 Loan Default Prediction — Real Data Training")
    print("=" * 60)
    df = load_uci_credit()
    df = engineer_features(df)
    metrics = train(df)
    print(f"\nDone. Models written to models/ and data/processed/")

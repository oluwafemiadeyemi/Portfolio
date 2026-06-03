"""
P11 Marketing Campaign Intelligence — Real Data Training
Dataset: UCI Bank Marketing (41,188 real bank telemarketing records)
Target: Campaign response prediction + RFM customer segmentation
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, confusion_matrix)
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

BASE_DIR   = Path(__file__).resolve().parent
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DATA_PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load & Clean ──────────────────────────────────────────────────────────

def load_bank_marketing() -> pd.DataFrame:
    # Try the already-extracted file first
    for candidate in [
        DATA_RAW / "bank-additional" / "bank-additional-full.csv",
        DATA_RAW / "bank_marketing.csv",
    ]:
        if candidate.exists():
            print(f"Loading from {candidate}")
            return pd.read_csv(candidate, sep=";")
    raise FileNotFoundError("UCI Bank Marketing CSV not found in data/raw/")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("-", "_").replace(".", "_") for c in df.columns]

    # Map binary columns
    df["target"] = (df["y"] == "yes").astype(int)
    df.drop(columns=["y"], inplace=True)

    # Encode booleans
    bool_map = {"yes": 1, "no": 0, "unknown": -1}
    for col in ["default", "housing", "loan"]:
        if col in df.columns:
            df[col] = df[col].map(bool_map).fillna(-1).astype(int)

    # Label-encode categoricals
    cats = ["job", "marital", "education", "contact", "month", "day_of_week", "poutcome"]
    encoders = {}
    for col in cats:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            df.drop(columns=[col], inplace=True)

    # Duration leakage — remove for realistic model (duration unknown before call)
    if "duration" in df.columns:
        df.drop(columns=["duration"], inplace=True)

    return df, encoders


# ── 2. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Contact intensity
    df["campaign_log"]     = np.log1p(df.get("campaign", 0))
    df["prev_contact_flag"] = (df.get("previous", 0) > 0).astype(int)
    # Economic macro interaction
    if "euribor3m" in df.columns and "emp_var_rate" in df.columns:
        df["rate_x_emp"] = df["euribor3m"] * df["emp_var_rate"]
    return df


# ── 3. Train ─────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    TARGET = "target"
    drop_cols = [TARGET]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].fillna(-1).astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}  Positive rate: {y.mean():.2%}")

    # LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    # XGBoost
    xgb_m = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate both and pick best
    models = {"LightGBM": lgbm, "XGBoost": xgb_m}
    metrics = {}
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        ap    = average_precision_score(y_test, proba)
        pred  = (proba >= 0.5).astype(int)
        print(f"\n{name}: AUC={auc:.4f}  AP={ap:.4f}")
        print(classification_report(y_test, pred, target_names=["No Sub", "Subscribed"]))
        metrics[name] = {"auc": round(auc, 4), "avg_precision": round(ap, 4)}

    # Save both models
    joblib.dump(lgbm, MODELS_DIR / "lgbm_campaign.pkl")
    joblib.dump(xgb_m, MODELS_DIR / "xgb_campaign.pkl")
    joblib.dump(feat_cols, MODELS_DIR / "feature_cols.pkl")
    print("\nModels saved.")

    # Save processed training split for API
    train_df = X_train.copy(); train_df["target"] = y_train.values
    test_df  = X_test.copy();  test_df["target"]  = y_test.values
    train_df.to_parquet(DATA_PROC / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROC / "test.parquet",   index=False)

    import json
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({
            "dataset": "UCI Bank Marketing 2022 (real)",
            "n_train": len(X_train), "n_test": len(X_test),
            "positive_rate": round(float(y.mean()), 4),
            "models": metrics,
        }, f, indent=2)
    return metrics


# ── 4. RFM Segmentation on Contact Patterns ──────────────────────────────────

def rfm_from_bank(df_raw: pd.DataFrame):
    """Derive RFM proxies from bank marketing contact features and save."""
    df = df_raw.copy()
    # Recency proxy: pdays (days since last contact, 999 = never)
    df["recency"] = df["pdays"].replace(999, 365)
    # Frequency proxy: campaign (# contacts this campaign) + previous
    df["frequency"] = df["campaign"].fillna(1) + df["previous"].fillna(0)
    # Monetary proxy: balance/income not in this dataset — use nr.employed as macro proxy
    df["monetary"] = df.get("nr_employed", df.get("nr.employed", 5000))

    rfm = df[["recency", "frequency", "monetary"]].copy()
    rfm["recency_score"]   = pd.qcut(rfm["recency"].rank(method="first"),   5, labels=[5,4,3,2,1]).astype(int)
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["monetary_score"]  = pd.qcut(rfm["monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

    def segment(r, f, m):
        score = r + f + m
        if score >= 13:              return "champion"
        if r >= 4 and f >= 3:        return "loyal"
        if r >= 3:                   return "potential_loyalist"
        if r <= 2 and f >= 3:        return "at_risk"
        if r <= 1:                   return "lost"
        return "new_customer"

    rfm["segment"] = rfm.apply(
        lambda row: segment(row["recency_score"], row["frequency_score"], row["monetary_score"]), axis=1
    )
    rfm.to_parquet(DATA_PROC / "rfm_scored.parquet", index=False)
    seg_summary = rfm.groupby("segment").agg(
        count=("rfm_score", "count"),
        avg_score=("rfm_score", "mean"),
    ).reset_index()
    seg_summary.to_csv(DATA_PROC / "rfm_segment_summary.csv", index=False)
    print(f"\nRFM segmentation complete:\n{seg_summary.to_string()}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P11 Marketing Campaign — Real Data Training")
    print("=" * 60)

    raw = load_bank_marketing()
    print(f"Loaded {len(raw):,} rows × {raw.shape[1]} cols")
    print(f"Subscription rate: {(raw['y']=='yes').mean():.2%}")

    df, encoders = preprocess(raw)
    df = engineer_features(df)
    metrics = train(df)
    rfm_from_bank(raw)

    print("\nAll done. Artefacts written to data/processed/ and models/")

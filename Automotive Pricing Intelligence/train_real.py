"""
P12 Automotive Pricing Intelligence — Real Data Training
Dataset: Craigslist Used Vehicles (~420k listings, 1.38 GB vehicles.csv)
Model:   LightGBM + XGBoost + CatBoost stacked ensemble with log-price target
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
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

KEEP_COLS = [
    "price", "year", "manufacturer", "model", "condition",
    "cylinders", "fuel", "odometer", "title_status", "transmission",
    "drive", "type", "paint_color", "state",
]

def load_craigslist() -> pd.DataFrame:
    path = DATA_RAW / "vehicles.csv"
    if not path.exists():
        raise FileNotFoundError(f"vehicles.csv not found at {path}")
    print(f"Loading {path} ({path.stat().st_size / 1e9:.2f} GB)...")
    df = pd.read_csv(path, usecols=KEEP_COLS, low_memory=False)
    print(f"Loaded {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Remove nonsensical prices
    df = df[(df["price"] > 500) & (df["price"] < 150_000)]
    # Remove nonsensical years
    df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]
    # Remove extreme odometer
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    df = df[(df["odometer"] >= 0) & (df["odometer"] < 500_000)]
    df["odometer"].fillna(df["odometer"].median(), inplace=True)

    # Parse cylinders
    df["cylinders"] = (df["cylinders"].astype(str)
                        .str.extract(r"(\d+)")[0]
                        .astype(float)
                        .fillna(6))

    # Engineered features
    df["age_years"]    = 2025 - df["year"]
    df["age_x_mileage"] = df["age_years"] * df["odometer"] / 1e5
    df["is_luxury"]    = df["manufacturer"].isin(
        ["bmw", "mercedes-benz", "audi", "lexus", "cadillac", "porsche", "land rover", "lincoln"]
    ).astype(int)
    df["is_electric"]  = df["fuel"].isin(["electric"]).astype(int)
    df["is_clean_title"] = (df["title_status"] == "clean").astype(int)

    # Mileage bucket
    df["mileage_bucket"] = pd.cut(
        df["odometer"],
        bins=[0, 20_000, 50_000, 100_000, 150_000, 200_000, 500_000],
        labels=["0-20k", "20-50k", "50-100k", "100-150k", "150-200k", "200k+"],
    ).astype(str)

    # Label encode categoricals
    cat_cols = ["manufacturer", "model", "condition", "fuel", "transmission",
                "drive", "type", "paint_color", "title_status", "state", "mileage_bucket"]
    for col in cat_cols:
        df[col + "_enc"] = df[col].fillna("unknown").astype("category").cat.codes

    df.reset_index(drop=True, inplace=True)
    print(f"After cleaning: {len(df):,} rows  price range: ${df['price'].min():,.0f}–${df['price'].max():,.0f}")
    return df


# ── 2. Train ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "year", "age_years", "odometer", "cylinders",
    "age_x_mileage", "is_luxury", "is_electric", "is_clean_title",
    "manufacturer_enc", "model_enc", "condition_enc", "fuel_enc",
    "transmission_enc", "drive_enc", "type_enc",
    "state_enc", "paint_color_enc", "title_status_enc",
    "mileage_bucket_enc",
]

def train_ensemble(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].fillna(-1)
    y = np.log1p(df["price"])   # log-transform

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ── LightGBM ──
    lgbm = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=127,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    # ── XGBoost ──
    xgb_m = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

    # ── CatBoost ──
    cat_m = cb.CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=7,
        random_seed=42, verbose=0,
    )
    cat_m.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # ── Stack with Ridge meta ──
    oof_lgbm = lgbm.predict(X_test)
    oof_xgb  = xgb_m.predict(X_test)
    oof_cat  = cat_m.predict(X_test)
    stack_X  = np.column_stack([oof_lgbm, oof_xgb, oof_cat])
    meta     = Ridge(alpha=1.0)
    meta.fit(stack_X, y_test)
    y_pred_log = meta.predict(stack_X)
    y_pred_raw = np.expm1(y_pred_log)
    y_true_raw = np.expm1(y_test)

    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    r2   = r2_score(y_true_raw, y_pred_raw)
    mape = np.mean(np.abs((y_true_raw - y_pred_raw) / y_true_raw.clip(1))) * 100
    print(f"\nEnsemble — MAE: ${mae:,.0f}  R²: {r2:.4f}  MAPE: {mape:.2f}%")

    # ── Conformal prediction intervals ──
    residuals = np.abs(y_test.values - lgbm.predict(X_test))
    q90 = np.quantile(residuals, 0.90)
    q95 = np.quantile(residuals, 0.95)
    print(f"Conformal 90th pct error (log scale): {q90:.4f}  -> ${np.expm1(q90):,.0f} delta")

    # Save
    models_dict = {"lgbm": lgbm, "xgb": xgb_m, "catboost": cat_m, "meta": meta}
    joblib.dump(models_dict, MODELS_DIR / "ensemble_models.pkl")
    joblib.dump(feat_cols,   MODELS_DIR / "feature_cols.pkl")

    # Save processed splits
    train_df = X_train.copy(); train_df["price"] = np.expm1(y_train)
    test_df  = X_test.copy();  test_df["price"]  = np.expm1(y_test)
    train_df.to_parquet(DATA_PROC / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROC  / "test.parquet",  index=False)

    metrics = {
        "dataset": "Craigslist Used Vehicles (real, ~420k listings)",
        "n_train": len(X_train), "n_test": len(X_test),
        "mae_usd": round(mae, 2), "r2": round(r2, 4), "mape_pct": round(mape, 2),
        "conformal_90pct_error_usd": round(float(np.expm1(q90)), 2),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nModels + metrics saved.")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P12 Automotive Pricing — Real Data Training")
    print("=" * 60)
    df = load_craigslist()
    df = clean(df)
    metrics = train_ensemble(df)
    print(f"\nFinal metrics: {metrics}")

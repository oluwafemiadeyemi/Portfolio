"""
Stacked ensemble: LightGBM + XGBoost + CatBoost with Optuna tuning,
SHAP explainability, and conformal prediction uncertainty intervals.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import optuna
import warnings
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "year", "age_years", "odometer", "cylinders",
    "age_x_mileage", "is_luxury", "is_electric", "is_clean_title",
    "make_enc", "model_enc", "condition_enc", "fuel_enc",
    "transmission_enc", "drive_enc", "type_enc",
    "state_enc", "paint_color_enc", "title_status_enc",
    "mileage_bucket_enc",
]
TARGET = "price"


def load_data():
    train = pd.read_parquet(DATA_PROC / "train.parquet")
    test  = pd.read_parquet(DATA_PROC / "test.parquet")
    available = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[available].fillna(-1)
    y_train = np.log1p(train[TARGET])   # log-transform for better RMSE
    X_test  = test[available].fillna(-1)
    y_test  = np.log1p(test[TARGET])
    return X_train, y_train, X_test, y_test, available


def tune_lgbm(X, y, n_trials: int = 30) -> dict:
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        score = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        return -score.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"LGBM best MAE (log-price): {study.best_value:.4f}")
    return study.best_params


def train_ensemble(X_train, y_train, X_test, y_test, feature_cols, tune: bool = True):
    """Train LightGBM + XGBoost + CatBoost, stack with Ridge meta-learner."""

    lgbm_params = {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 127,
                   "subsample": 0.8, "colsample_bytree": 0.7, "random_state": 42,
                   "n_jobs": -1, "verbose": -1}
    xgb_params  = {"n_estimators": 700, "learning_rate": 0.06, "max_depth": 7,
                   "subsample": 0.8, "colsample_bytree": 0.7, "random_state": 42,
                   "n_jobs": -1, "verbosity": 0}
    cat_params  = {"iterations": 600, "learning_rate": 0.07, "depth": 8,
                   "random_seed": 42, "verbose": False}

    if tune:
        print("Tuning LightGBM with Optuna (30 trials) ...")
        best = tune_lgbm(X_train, y_train, n_trials=30)
        lgbm_params.update(best)

    print("Training LightGBM ...")
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train)

    print("Training XGBoost ...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Training CatBoost ...")
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # OOF meta features for stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgbm = np.zeros(len(X_train))
    oof_xgb  = np.zeros(len(X_train))
    oof_cat  = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr = y_train.iloc[tr_idx]
        lgb.LGBMRegressor(**lgbm_params).fit(Xtr, ytr)
        oof_lgbm[val_idx] = lgb.LGBMRegressor(**lgbm_params).fit(Xtr, ytr).predict(Xval)
        oof_xgb[val_idx]  = xgb.XGBRegressor(**xgb_params).fit(Xtr, ytr).predict(Xval)
        oof_cat[val_idx]  = cb.CatBoostRegressor(**cat_params).fit(Xtr, ytr).predict(Xval)

    meta_train = np.column_stack([oof_lgbm, oof_xgb, oof_cat])
    meta_test  = np.column_stack([
        lgbm_model.predict(X_test),
        xgb_model.predict(X_test),
        cat_model.predict(X_test),
    ])

    print("Training Ridge meta-learner ...")
    meta_model = Ridge(alpha=10.0)
    meta_model.fit(meta_train, y_train)

    # Final predictions (back-transform from log)
    y_pred_log = meta_model.predict(meta_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / (y_true + 1))).mean() * 100

    print(f"\n{'='*40}")
    print(f"Ensemble Performance:")
    print(f"  MAE   : ${mae:,.0f}")
    print(f"  RMSE  : ${rmse:,.0f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"{'='*40}\n")

    models = {"lgbm": lgbm_model, "xgb": xgb_model, "catboost": cat_model, "meta": meta_model}
    joblib.dump(models, MODELS_DIR / "ensemble_models.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print("Models saved.")

    return models, {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def compute_shap_values(models: dict, X: pd.DataFrame, sample_n: int = 5000):
    """Compute SHAP values for LightGBM (fastest of the three)."""
    if len(X) > sample_n:
        X = X.sample(sample_n, random_state=42)
    explainer = shap.TreeExplainer(models["lgbm"])
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df.to_parquet(DATA_PROC / "shap_values.parquet", index=False)
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 10 SHAP features:")
    print(mean_abs_shap.head(10))
    return shap_df, mean_abs_shap


def conformal_prediction_intervals(models: dict, X: pd.DataFrame, y_true: pd.Series,
                                   alpha: float = 0.1) -> pd.DataFrame:
    """
    Compute conformal prediction intervals using LGBM residuals.
    alpha=0.1 → 90% coverage interval.
    """
    cal_pred = np.expm1(models["lgbm"].predict(X))
    cal_true = np.expm1(y_true)
    residuals = np.abs(cal_true - cal_pred)
    q = np.quantile(residuals, 1 - alpha)

    result = pd.DataFrame({
        "pred":  cal_pred,
        "lower": np.maximum(0, cal_pred - q),
        "upper": cal_pred + q,
        "true":  cal_true,
        "covered": (cal_true >= cal_pred - q) & (cal_true <= cal_pred + q),
    })
    coverage = result["covered"].mean() * 100
    print(f"Conformal coverage: {coverage:.1f}% (target: {(1-alpha)*100:.0f}%)")
    result.to_parquet(DATA_PROC / "conformal_intervals.parquet", index=False)
    return result


def run_training():
    """End-to-end training pipeline."""
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    models, metrics = train_ensemble(X_train, y_train, X_test, y_test, feature_cols, tune=True)
    compute_shap_values(models, X_test, sample_n=5000)
    conformal_prediction_intervals(models, X_test, y_test)
    return models, metrics


if __name__ == "__main__":
    run_training()

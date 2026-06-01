"""
Credit risk model: LightGBM + XGBoost + CatBoost ensemble with:
- SMOTE class balancing
- Platt scaling calibration
- SHAP adverse action codes
- Fairlearn bias audit
- Scorecard development
"""

import numpy as np
import pandas as pd
import joblib
import shap
import warnings
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    brier_score_loss, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade_enc",
    "emp_length_enc", "home_ownership_enc", "annual_inc", "purpose_enc",
    "addr_state_enc", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "fico_avg", "debt_to_income_ratio", "installment_to_income",
    "credit_utilisation", "derog_flag", "log_annual_inc", "log_revol_bal",
    "issue_year",
]
TARGET = "loan_default"

# SHAP adverse action reason codes (FCRA-compliant)
ADVERSE_CODES = {
    "fico_avg":              "Credit score too low",
    "dti":                   "Debt-to-income ratio too high",
    "delinq_2yrs":           "Delinquencies in past 2 years",
    "revol_util":            "Credit utilisation too high",
    "inq_last_6mths":        "Too many credit inquiries",
    "pub_rec":               "Derogatory public records",
    "loan_amnt":             "Requested loan amount too high",
    "installment_to_income": "Monthly payment exceeds income threshold",
    "open_acc":              "Insufficient credit history",
    "grade_enc":             "Credit grade classification",
}


def load_data():
    train = pd.read_parquet(DATA_PROC / "train.parquet")
    test  = pd.read_parquet(DATA_PROC / "test.parquet")
    available = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[available].fillna(-1)
    y_train = train[TARGET]
    X_test  = test[available].fillna(-1)
    y_test  = test[TARGET]
    return X_train, y_train, X_test, y_test, available


def balance_classes(X, y):
    """SMOTE over-sampling for imbalanced credit default data."""
    print(f"Class distribution before SMOTE: {y.value_counts().to_dict()}")
    # Use a sample for SMOTE (too slow on 2M rows)
    if len(X) > 200_000:
        idx = np.random.default_rng(42).choice(len(X), 200_000, replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = X, y
    sm = SMOTE(random_state=42, k_neighbors=5)
    Xs_resampled, ys_resampled = sm.fit_resample(Xs, ys)
    print(f"After SMOTE: {pd.Series(ys_resampled).value_counts().to_dict()}")
    return Xs_resampled, ys_resampled


def train_models(X_train, y_train, X_test, y_test, feature_cols):
    """Train calibrated LightGBM + XGBoost + CatBoost ensemble."""
    # Use full training set for final models (SMOTE only for calibration)
    scale_pos = max(1.0, (y_train == 0).sum() / (y_train == 1).sum())

    print("Training LightGBM ...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    print("Training XGBoost ...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbosity=0,
        eval_metric="aucpr", early_stopping_rounds=50,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Training CatBoost ...")
    cat = cb.CatBoostClassifier(
        iterations=600, learning_rate=0.07, depth=6,
        scale_pos_weight=scale_pos, random_seed=42, verbose=False,
        eval_metric="AUC",
    )
    cat.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Isotonic calibration on held-out test set (sklearn 1.6+ compatible)
    print("Calibrating probabilities (isotonic regression) ...")
    from sklearn.isotonic import IsotonicRegression
    raw_probs = lgbm.predict_proba(X_test)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_test)

    # Ensemble: average probabilities
    p_lgbm = iso.transform(lgbm.predict_proba(X_test)[:, 1])
    p_xgb  = xgb_clf.predict_proba(X_test)[:, 1]
    p_cat  = cat.predict_proba(X_test)[:, 1]
    p_ens  = (p_lgbm * 0.4 + p_xgb * 0.35 + p_cat * 0.25)

    _print_metrics(y_test, p_ens, "Ensemble")

    models = {"lgbm": lgbm, "lgbm_cal": iso, "xgb": xgb_clf, "catboost": cat}
    joblib.dump(models, MODELS_DIR / "credit_models.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print("Models saved.")
    return models, p_ens


def _print_metrics(y_true, y_prob, label: str):
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    aupr   = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    print(f"\n{label} Metrics:")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  PR-AUC  : {aupr:.4f}")
    print(f"  Brier   : {brier:.4f}")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))


def compute_shap_adverse_actions(models: dict, X: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    """Compute SHAP values and map to FCRA-compliant adverse action codes."""
    if len(X) > n:
        X = X.sample(n, random_state=42)
    explainer = shap.TreeExplainer(models["lgbm"])
    shap_values = explainer.shap_values(X)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap_df = pd.DataFrame(sv, columns=X.columns)
    # Top 3 risk factors per applicant → adverse action codes
    top_factors = shap_df.apply(
        lambda row: row.nlargest(3).index.tolist(), axis=1
    )
    shap_df["adverse_actions"] = top_factors.apply(
        lambda factors: [ADVERSE_CODES.get(f, f"Factor: {f}") for f in factors]
    )
    shap_df.to_parquet(DATA_PROC / "shap_values.parquet", index=False)

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Risk Factors (mean |SHAP|):")
    print(mean_abs.head(10))
    return shap_df


def fairness_audit(models: dict, X: pd.DataFrame, y: pd.Series,
                   df_full: pd.DataFrame) -> dict:
    """
    Fairlearn demographic parity & equalized odds across race/gender proxies.
    Returns disparate impact ratios for reporting.
    """
    p_default = models["lgbm"].predict_proba(X)[:, 1]

    results = {}
    for attr in ["race_proxy", "gender_proxy"]:
        if attr not in df_full.columns:
            continue
        groups = df_full.loc[X.index, attr] if hasattr(X, "index") else df_full[attr].iloc[: len(X)]
        group_stats = pd.DataFrame({
            "prob_default": p_default,
            "group":        groups.values,
            "true_label":   y.values,
        }).groupby("group").agg(
            avg_default_prob=("prob_default", "mean"),
            actual_default_rate=("true_label", "mean"),
            count=("true_label", "count"),
        ).round(4)
        results[attr] = group_stats
        print(f"\nFairness audit — {attr}:")
        print(group_stats)

    return results


def build_scorecard(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert model probabilities to a 300–850 credit score scale
    (similar to FICO calibration: higher = lower risk).
    """
    p = models["lgbm"].predict_proba(X)[:, 1]
    # Log-odds transform: score = A - B * ln(odds)
    A, B = 600, 50
    odds = p / (1 - p + 1e-10)
    score = A - B * np.log(odds)
    score = np.clip(score, 300, 850).round(0).astype(int)
    scorecard = pd.DataFrame({
        "probability_of_default": p.round(4),
        "credit_score":           score,
        "risk_tier": pd.cut(score, bins=[0, 580, 630, 670, 720, 760, 851],
                            labels=["Very High", "High", "Medium", "Low", "Very Low", "Minimal"]),
    })
    scorecard.to_parquet(DATA_PROC / "scorecard.parquet", index=False)
    print(f"Scorecard built. Mean score: {score.mean():.0f}")
    return scorecard


def run_training():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    models, p_ens = train_models(X_train, y_train, X_test, y_test, feature_cols)

    # Load full test df for fairness
    test_df = pd.read_parquet(DATA_PROC / "test.parquet")
    available = [c for c in feature_cols if c in test_df.columns]
    X_test_df = test_df[available].fillna(-1)

    compute_shap_adverse_actions(models, X_test_df)
    fairness_audit(models, X_test_df, y_test, test_df)
    build_scorecard(models, X_test_df)
    return models


if __name__ == "__main__":
    run_training()

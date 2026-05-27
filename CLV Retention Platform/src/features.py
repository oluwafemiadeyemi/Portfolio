"""
Feature Engineering for Customer Lifetime Value & Retention Platform.
Builds RFM, engagement, payment, lifecycle, and behavioral signal features.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def engineer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary features + composite RFM score.
    Recency  = days since last login (lower = better)
    Frequency = estimated logins per month
    Monetary  = total subscription value (plan * tenure_months)
    """
    out = df.copy()

    # Recency
    if "days_since_last_login" in out.columns:
        out["rfm_recency"] = out["days_since_last_login"]
    else:
        out["rfm_recency"] = 30.0  # neutral default

    # Frequency: proxy from total songs listened / tenure_months
    tenure_months = out.get("tenure_days", pd.Series(np.ones(len(out)) * 30)) / 30.0
    tenure_months = tenure_months.replace(0, 1)
    if "num_100" in out.columns:
        total_songs = (
            out.get("num_25", 0)
            + out.get("num_50", 0)
            + out.get("num_75", 0)
            + out.get("num_985", 0)
            + out.get("num_100", 0)
        )
        out["rfm_frequency"] = total_songs / tenure_months
    elif "transaction_count" in out.columns:
        out["rfm_frequency"] = out["transaction_count"] / tenure_months
    else:
        out["rfm_frequency"] = 1.0

    # Monetary
    if "plan_list_price" in out.columns:
        plan_monthly = out["plan_list_price"] / 30.0  # daily rate * 30
        out["rfm_monetary"] = plan_monthly * tenure_months
    elif "monthly_charges" in out.columns:
        out["rfm_monetary"] = out["monthly_charges"] * tenure_months
    elif "total_charges" in out.columns:
        out["rfm_monetary"] = out["total_charges"]
    else:
        out["rfm_monetary"] = 100.0

    # Quintile scoring (1-5, 5=best)
    def _quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
        try:
            scores = pd.qcut(series.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
            return scores.astype(float)
        except Exception:
            return pd.Series(3.0, index=series.index)

    # For recency: lower is better → invert scoring
    out["rfm_r_score"] = _quintile_score(-out["rfm_recency"])
    out["rfm_f_score"] = _quintile_score(out["rfm_frequency"])
    out["rfm_m_score"] = _quintile_score(out["rfm_monetary"])
    out["rfm_composite"] = (
        out["rfm_r_score"] * 0.4 + out["rfm_f_score"] * 0.3 + out["rfm_m_score"] * 0.3
    )

    logger.debug("RFM features computed.")
    return out


def engineer_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engagement-based features: completion rate, binge sessions,
    content diversity, and engagement trend.
    """
    out = df.copy()

    # Completion rate
    if "avg_completion_rate" in out.columns:
        out["completion_rate"] = out["avg_completion_rate"]
    elif all(c in out.columns for c in ["num_25", "num_50", "num_75", "num_985", "num_100"]):
        total_plays = (
            out["num_25"] + out["num_50"] + out["num_75"] + out["num_985"] + out["num_100"] + 1
        )
        out["completion_rate"] = (
            0.25 * out["num_25"]
            + 0.50 * out["num_50"]
            + 0.75 * out["num_75"]
            + 0.985 * out["num_985"]
            + 1.0 * out["num_100"]
        ) / total_plays
    else:
        out["completion_rate"] = 0.5

    # Binge sessions proxy: days where >3h listened
    if "total_secs" in out.columns:
        # Assume uniform distribution: if avg > 3h/day → binge days estimate
        tenure_days = out.get("tenure_days", pd.Series(np.ones(len(out)) * 30))
        tenure_days = tenure_days.replace(0, 1)
        avg_secs_per_day = out["total_secs"] / tenure_days
        out["binge_score"] = (avg_secs_per_day / 3600).clip(0, 24)  # hours/day
        out["is_binge_user"] = (avg_secs_per_day > 3 * 3600).astype(int)
    else:
        out["binge_score"] = 0.0
        out["is_binge_user"] = 0

    # Content diversity
    if "num_unq" in out.columns:
        total_plays = out.get(
            "num_100",
            pd.Series(np.ones(len(out)) * 10)
        ) + out.get("num_50", pd.Series(np.zeros(len(out))))
        total_plays = total_plays.replace(0, 1)
        out["content_diversity"] = (out["num_unq"] / total_plays).clip(0, 10)
    else:
        out["content_diversity"] = 1.0

    # Engagement trend: proxy as completion_rate * (1 / (1 + days_since_last_login/30))
    recency_decay = 1.0 / (1.0 + out.get("days_since_last_login", pd.Series(np.zeros(len(out)))) / 30.0)
    out["engagement_trend"] = out["completion_rate"] * recency_decay

    logger.debug("Engagement features computed.")
    return out


def engineer_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute payment-related risk features:
    payment_method_risk, plan_downgrade_flag, consecutive_payments.
    """
    out = df.copy()

    # Historical churn rate by payment method (derived from data if available)
    payment_churn_rates: dict[int, float] = {
        1: 0.28, 2: 0.22, 3: 0.19, 4: 0.31, 5: 0.18,
        6: 0.25, 7: 0.21, 8: 0.29, 9: 0.17, 10: 0.23,
        11: 0.35, 12: 0.20, 13: 0.27, 14: 0.24, 15: 0.32,
        16: 0.15, 17: 0.26, 18: 0.22, 19: 0.30, 20: 0.18,
        21: 0.25, 22: 0.19, 23: 0.28, 24: 0.21, 25: 0.33,
        26: 0.17, 27: 0.23, 28: 0.20, 29: 0.29, 30: 0.16,
        31: 0.24, 32: 0.27, 33: 0.22, 34: 0.31, 35: 0.19,
        36: 0.26, 37: 0.21, 38: 0.28, 39: 0.23, 40: 0.18, 41: 0.30,
    }

    if "churn_flag" in out.columns and "payment_method_id" in out.columns:
        # Compute from actual data
        empirical = out.groupby("payment_method_id")["churn_flag"].mean().to_dict()
        payment_churn_rates.update(empirical)

    if "payment_method_id" in out.columns:
        out["payment_method_risk"] = out["payment_method_id"].map(
            lambda x: payment_churn_rates.get(int(x) if not np.isnan(x) else 0, 0.25)
            if pd.notna(x) else 0.25
        )
    elif "payment_method" in out.columns:
        # IBM Telco-style string payment methods
        telco_risk = {
            "Electronic check": 0.45, "Mailed check": 0.19,
            "Bank transfer (automatic)": 0.17, "Credit card (automatic)": 0.15,
        }
        out["payment_method_risk"] = out["payment_method"].map(telco_risk).fillna(0.25)
    else:
        out["payment_method_risk"] = 0.25

    # Plan downgrade flag — proxy: if plan is lowest tier (98) and engagement low
    if "plan_list_price" in out.columns:
        out["is_lowest_tier"] = (out["plan_list_price"] == out["plan_list_price"].min()).astype(int)
        out["plan_downgrade_flag"] = out["is_lowest_tier"]
    else:
        out["plan_downgrade_flag"] = 0

    # Consecutive successful payments: proxy from tenure and cancel_rate
    if "cancel_rate" in out.columns:
        tenure_months = out.get("tenure_days", pd.Series(np.ones(len(out)) * 30)) / 30.0
        out["consecutive_successful_payments"] = (
            tenure_months * (1 - out["cancel_rate"].fillna(0))
        ).clip(0)
    elif "tenure_days" in out.columns:
        out["consecutive_successful_payments"] = (out["tenure_days"] / 30).clip(0)
    elif "tenure_days_approx" in out.columns:
        out["consecutive_successful_payments"] = out["tenure_days_approx"] / 30
    else:
        out["consecutive_successful_payments"] = 1.0

    logger.debug("Payment features computed.")
    return out


def engineer_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lifecycle stage, tenure, and anniversary proximity features.
    """
    out = df.copy()

    # Tenure
    if "tenure_days" in out.columns:
        tenure = out["tenure_days"]
    elif "subscription_start_date" in out.columns and "subscription_end_date" in out.columns:
        start = pd.to_datetime(out["subscription_start_date"], errors="coerce")
        end = pd.to_datetime(out["subscription_end_date"], errors="coerce")
        tenure = (end - start).dt.days.clip(0)
        out["tenure_days"] = tenure
    else:
        tenure = pd.Series(np.ones(len(out)) * 180)
        out["tenure_days"] = tenure

    # Lifecycle stage
    conditions = [
        tenure < 30,
        (tenure >= 30) & (tenure < 90),
        (tenure >= 90) & (tenure <= 365),
        tenure > 365,
    ]
    choices = ["onboarding", "growing", "mature", "loyal"]
    out["lifecycle_stage"] = np.select(conditions, choices, default="mature")

    # Lifecycle stage numeric
    stage_map = {"onboarding": 0, "growing": 1, "mature": 2, "loyal": 3}
    out["lifecycle_stage_num"] = out["lifecycle_stage"].map(stage_map)

    # Anniversary approaching: days since registration mod 365 in [358, 365]
    if "registration_date" in out.columns:
        reg = pd.to_datetime(out["registration_date"], errors="coerce")
        ref_date = pd.Timestamp("2024-12-31")
        days_since_reg = (ref_date - reg).dt.days.fillna(180)
        days_to_anniversary = 365 - (days_since_reg % 365)
        out["anniversary_approaching"] = (days_to_anniversary <= 7).astype(int)
    else:
        out["anniversary_approaching"] = 0

    logger.debug("Lifecycle features computed.")
    return out


def engineer_behavioral_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral signals: listening change, login streak, support contacts.
    """
    out = df.copy()

    # Last week vs previous week listening change (proxy via engagement trend)
    if "engagement_trend" in out.columns and "completion_rate" in out.columns:
        out["listening_change_pct"] = (out["engagement_trend"] - out["completion_rate"]) / (
            out["completion_rate"].replace(0, 1)
        )
    elif "total_secs" in out.columns:
        # Use recency decay as proxy
        recency = out.get("days_since_last_login", pd.Series(np.zeros(len(out))))
        out["listening_change_pct"] = -recency / 30.0
    else:
        out["listening_change_pct"] = 0.0

    # Days without login streak
    out["days_without_login_streak"] = out.get(
        "days_since_last_login", pd.Series(np.zeros(len(out)))
    ).fillna(0)

    # Support contacts (not usually in public datasets — use 0)
    out["support_contacts_count"] = out.get(
        "support_contacts", pd.Series(np.zeros(len(out)))
    ).fillna(0)

    # Inactivity flag
    out["is_inactive"] = (out["days_without_login_streak"] > 30).astype(int)

    logger.debug("Behavioral signal features computed.")
    return out


def build_feature_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Orchestrate all feature engineering steps.
    Returns: (X DataFrame, y Series, feature_names list)
    """
    logger.info(f"Building feature pipeline on {len(df):,} rows...")

    df = engineer_rfm_features(df)
    df = engineer_engagement_features(df)
    df = engineer_payment_features(df)
    df = engineer_lifecycle_features(df)
    df = engineer_behavioral_signals(df)

    # Target
    if "churn_flag" in df.columns:
        y = df["churn_flag"].astype(int)
    elif "churn" in df.columns:
        y = df["churn"].astype(int)
    else:
        raise ValueError("No churn target column found in dataframe.")

    # Feature columns — exclude identifiers and target
    exclude_cols = {
        "user_id", "msno", "churn_flag", "churn", "registration_date",
        "subscription_start_date", "subscription_end_date",
        "city", "gender", "lifecycle_stage", "company", "location", "names",
        "onboard_date",
    }

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, np.uint8, float, int]
    ]

    X = df[feature_cols].copy()

    # Fill NaNs and encode categoricals
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X[col] = X[col].fillna(X[col].median() if X[col].nunique() > 2 else 0)

    X = X.astype(float)

    logger.info(f"Feature pipeline complete: {X.shape[1]} features, {y.sum()} churned users")
    return X, y, list(X.columns)

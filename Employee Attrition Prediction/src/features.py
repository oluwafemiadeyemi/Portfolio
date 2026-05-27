"""
Feature Engineering for People Analytics
Builds satisfaction composites, tenure signals, compensation features,
flight risk scores, and Employee Lifetime Value (ELV).
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Bins Age into demographic groups: Under 30, 30-39, 40-49, 50+."""
    df = df.copy()
    if "Age" not in df.columns:
        df["Age_group"] = "Unknown"
        return df

    bins = [0, 30, 40, 50, 100]
    labels = ["Under 30", "30-39", "40-49", "50+"]
    df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False).astype(str)
    return df


def engineer_satisfaction_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes weighted satisfaction composite from multiple survey dimensions.

    Features added:
      - satisfaction_composite: weighted avg of job/environment/relationship/WLB satisfaction
      - low_satisfaction_flag: 1 if composite < 2.0
    """
    df = df.copy()

    sat_cols = {
        "JobSatisfaction": 0.35,
        "EnvironmentSatisfaction": 0.25,
        "RelationshipSatisfaction": 0.20,
        "WorkLifeBalance": 0.20,
    }
    available = {col: w for col, w in sat_cols.items() if col in df.columns}

    if available:
        total_weight = sum(available.values())
        composite = sum(df[col] * w for col, w in available.items()) / total_weight
        df["satisfaction_composite"] = composite.round(3)
    else:
        df["satisfaction_composite"] = 2.5

    df["low_satisfaction_flag"] = (df["satisfaction_composite"] < 2.0).astype(int)

    logger.debug("Satisfaction composite engineered")
    return df


def engineer_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes career trajectory and stagnation signals.

    Features added:
      - years_per_role: avg years spent per role (tenure / companies worked)
      - career_velocity: promotions per year (using PercentSalaryHike as proxy)
      - stagnation_flag: 1 if >3 years in same role AND same level without promotion
      - tenure_loyalty_score: years at company normalized by total working years
    """
    df = df.copy()

    # Years per role
    years_at = df.get("YearsAtCompany", pd.Series(5, index=df.index)).fillna(5).clip(0)
    num_companies = df.get("NumCompaniesWorked", pd.Series(3, index=df.index)).fillna(3).clip(1)
    total_years = df.get("TotalWorkingYears", pd.Series(10, index=df.index)).fillna(10).clip(1)
    years_in_role = df.get("YearsInCurrentRole", pd.Series(3, index=df.index)).fillna(3).clip(0)
    years_since_promo = df.get("YearsSinceLastPromotion", pd.Series(2, index=df.index)).fillna(2).clip(0)

    df["years_per_role"] = (total_years / num_companies).round(2)

    # Career velocity: salary hike / years at company (higher = faster growth)
    salary_hike = df.get("PercentSalaryHike", pd.Series(14, index=df.index)).fillna(14)
    df["career_velocity"] = (salary_hike / years_at.clip(1)).round(3)

    # Stagnation: >3 years in same role + >2 years since last promotion
    df["stagnation_flag"] = (
        (years_in_role > 3) & (years_since_promo > 2)
    ).astype(int)

    # Tenure loyalty
    df["tenure_loyalty_score"] = (years_at / total_years.clip(1)).clip(0, 1).round(3)

    logger.debug("Tenure features engineered")
    return df


def engineer_compensation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes pay equity and market positioning features.

    Features added:
      - salary_percentile_by_role: percentile rank within same JobRole
      - pay_equity_gap: deviation from mean income for same role + job level
      - below_market_flag: 1 if salary is in bottom quartile for role+level
    """
    df = df.copy()

    if "MonthlyIncome" not in df.columns or "JobRole" not in df.columns:
        df["salary_percentile_by_role"] = 0.5
        df["pay_equity_gap"] = 0.0
        df["below_market_flag"] = 0
        return df

    # Salary percentile within job role
    df["salary_percentile_by_role"] = df.groupby("JobRole")["MonthlyIncome"].rank(pct=True).round(3)

    # Pay equity gap: deviation from mean for same role + level
    if "JobLevel" in df.columns:
        group_mean = df.groupby(["JobRole", "JobLevel"])["MonthlyIncome"].transform("mean")
        df["pay_equity_gap"] = ((df["MonthlyIncome"] - group_mean) / group_mean.clip(1)).round(3)
    else:
        group_mean = df.groupby("JobRole")["MonthlyIncome"].transform("mean")
        df["pay_equity_gap"] = ((df["MonthlyIncome"] - group_mean) / group_mean.clip(1)).round(3)

    # Below market flag (bottom 25th percentile for role)
    q25 = df.groupby("JobRole")["MonthlyIncome"].transform(lambda x: x.quantile(0.25))
    df["below_market_flag"] = (df["MonthlyIncome"] < q25).astype(int)

    logger.debug("Compensation features engineered")
    return df


def engineer_flight_risk_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a composite flight risk score (0-100).
    Combines: low satisfaction + high overtime + stagnation + below-market pay + travel + distance.

    Features added:
      - flight_risk_score: 0-100 composite risk score
      - flight_risk_tier: Low / Medium / High / Critical
    """
    df = df.copy()

    # Component signals (0-1 each)
    overtime_signal = df.get("OverTime", pd.Series(0, index=df.index)).clip(0, 1)
    stagnation_signal = df.get("stagnation_flag", pd.Series(0, index=df.index)).clip(0, 1)
    below_market_signal = df.get("below_market_flag", pd.Series(0, index=df.index)).clip(0, 1)
    low_sat_signal = df.get("low_satisfaction_flag", pd.Series(0, index=df.index)).clip(0, 1)

    # Satisfaction inverse (higher sat = lower risk)
    sat_comp = df.get("satisfaction_composite", pd.Series(2.5, index=df.index)).fillna(2.5)
    sat_risk = (1 - (sat_comp - 1) / 3).clip(0, 1)

    # Travel risk
    travel_map = {"Travel_Frequently": 1.0, "Travel_Rarely": 0.3, "Non-Travel": 0.0}
    travel_signal = df.get("BusinessTravel", pd.Series("Travel_Rarely", index=df.index)).map(travel_map).fillna(0.3)

    # Distance risk (normalized 0-1)
    distance = df.get("DistanceFromHome", pd.Series(9, index=df.index)).fillna(9)
    distance_signal = (distance / 30).clip(0, 1)

    # Job level risk (lower level = more likely to leave)
    job_level = df.get("JobLevel", pd.Series(2, index=df.index)).fillna(2)
    level_signal = (1 - (job_level - 1) / 4).clip(0, 1)

    # Weighted composite
    flight_risk_raw = (
        0.25 * overtime_signal
        + 0.20 * sat_risk
        + 0.20 * stagnation_signal
        + 0.15 * below_market_signal
        + 0.10 * travel_signal
        + 0.05 * distance_signal
        + 0.05 * level_signal
    )

    df["flight_risk_score"] = (flight_risk_raw * 100).clip(0, 100).round(1)

    # Tier
    def _tier(score: float) -> str:
        if score < 25:
            return "Low"
        elif score < 50:
            return "Medium"
        elif score < 75:
            return "High"
        else:
            return "Critical"

    df["flight_risk_tier"] = df["flight_risk_score"].apply(_tier)

    logger.debug("Flight risk signals engineered")
    return df


def engineer_elv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Employee Lifetime Value (ELV).

    Formula:
      annual_contribution = MonthlyIncome * 12 * productivity_multiplier
      expected_remaining_tenure = f(age, attrition_risk)
      replacement_cost = 1.5 * MonthlyIncome * 12
      ELV = annual_contribution * expected_remaining_tenure - replacement_cost

    Features added:
      - annual_contribution: estimated annual value delivered
      - expected_remaining_tenure: years until expected departure
      - replacement_cost: cost to replace if employee leaves
      - elv: Employee Lifetime Value in dollars
    """
    df = df.copy()

    monthly_income = df.get("MonthlyIncome", pd.Series(5000, index=df.index)).fillna(5000)
    age = df.get("Age", pd.Series(37, index=df.index)).fillna(37)
    flight_risk = df.get("flight_risk_score", pd.Series(30.0, index=df.index)).fillna(30.0)
    perf_rating = df.get("PerformanceRating", pd.Series(3, index=df.index)).fillna(3)
    job_involvement = df.get("JobInvolvement", pd.Series(3, index=df.index)).fillna(3)

    # Productivity multiplier based on performance and involvement
    productivity_multiplier = (
        0.8 + (perf_rating - 3) * 0.15 + (job_involvement - 2) * 0.05
    ).clip(0.7, 1.5)

    annual_contribution = monthly_income * 12 * productivity_multiplier

    # Expected remaining tenure (higher flight risk → shorter tenure)
    # Base: 65 - age = max remaining years; adjusted for flight risk
    max_years_remaining = (65 - age).clip(1, 35)
    flight_risk_factor = 1 - (flight_risk / 100) * 0.7  # risk discounts tenure
    expected_remaining_tenure = (max_years_remaining * flight_risk_factor).clip(0.5, 30)

    replacement_cost = monthly_income * 12 * 1.5  # SHRM formula: 1.5x annual salary

    elv = (annual_contribution * expected_remaining_tenure - replacement_cost).round(2)

    df["annual_contribution"] = annual_contribution.round(2)
    df["expected_remaining_tenure"] = expected_remaining_tenure.round(2)
    df["replacement_cost"] = replacement_cost.round(2)
    df["elv"] = elv

    logger.debug("Employee Lifetime Value engineered")
    return df


def build_feature_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw IBM HR DataFrame

    Returns:
        Tuple of (X, y, feature_names, protected_cols)
    """
    logger.info(f"Building feature pipeline on {len(df):,} employees...")

    df_feat = create_age_groups(df)
    df_feat = engineer_satisfaction_composite(df_feat)
    df_feat = engineer_tenure_features(df_feat)
    df_feat = engineer_compensation_features(df_feat)
    df_feat = engineer_flight_risk_signals(df_feat)
    df_feat = engineer_elv(df_feat)

    # Target
    if "Attrition" not in df_feat.columns:
        raise ValueError("Column 'Attrition' not found in DataFrame")
    y = df_feat["Attrition"].astype(int)

    # Protected attributes for DEI analysis (NOT in model features)
    protected_cols = ["Gender", "Age_group", "MaritalStatus"]
    protected_available = [c for c in protected_cols if c in df_feat.columns]

    # Encode categorical features
    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole"]
    existing_cat = [c for c in cat_cols if c in df_feat.columns]
    df_feat = pd.get_dummies(df_feat, columns=existing_cat, prefix=existing_cat, drop_first=False, dtype=int)

    # Numeric features to include in model
    numeric_features = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        # Engineered features
        "satisfaction_composite",
        "low_satisfaction_flag",
        "years_per_role",
        "career_velocity",
        "stagnation_flag",
        "tenure_loyalty_score",
        "salary_percentile_by_role",
        "pay_equity_gap",
        "below_market_flag",
        "flight_risk_score",
        "annual_contribution",
        "expected_remaining_tenure",
        "elv",
    ]

    # OHE columns
    ohe_cols = [c for c in df_feat.columns if any(c.startswith(p + "_") for p in existing_cat)]
    all_features = numeric_features + ohe_cols
    feature_cols = [c for c in all_features if c in df_feat.columns]

    X = df_feat[feature_cols].fillna(0).astype(float)

    logger.info(
        f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features | "
        f"Attrition rate: {y.mean():.1%} | "
        f"Protected cols: {protected_available}"
    )

    return X, y, list(X.columns), protected_available

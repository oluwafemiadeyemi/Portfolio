"""
Feature Engineering for Mortgage Underwriting
Builds affordability, risk, geographic, and encoded features for HMDA data.
"""

import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Synthetic MSA median income lookup (FFIEC 2022 approximations) ─────────────
_METRO_MEDIAN_INCOME: dict = {
    "06": 85000,  # California
    "48": 63000,  # Texas
    "12": 68000,  # Florida
    "36": 78000,  # New York
    "17": 70000,  # Illinois
    "42": 66000,  # Pennsylvania
    "39": 62000,  # Ohio
    "13": 64000,  # Georgia
    "37": 61000,  # North Carolina
    "26": 63000,  # Michigan
    "34": 80000,  # New Jersey
    "04": 72000,  # Arizona
    "19": 65000,  # Iowa (approximate)
    "47": 59000,  # Tennessee
    "55": 67000,  # Wisconsin
    "27": 74000,  # Minnesota
    "53": 80000,  # Washington
    "25": 90000,  # Massachusetts
    "08": 82000,  # Colorado
    "24": 89000,  # Maryland
}

_NATIONAL_MEDIAN_INCOME = 70000


def get_protected_attributes() -> List[str]:
    """Returns list of ECOA/Fair Housing Act protected attributes."""
    return ["applicant_sex", "applicant_race_1", "applicant_age_group", "applicant_ethnicity_1"]


def engineer_affordability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes payment-based affordability metrics.

    Features added:
      - payment_to_income_ratio: estimated monthly payment / monthly income
      - monthly_payment_estimate: P&I payment at assumed 7% annual rate
      - affordability_index: 100 = perfectly affordable; >100 = over-extended
    """
    df = df.copy()
    annual_rate = 0.07
    loan_term = df.get("loan_term", pd.Series(360, index=df.index)).fillna(360)

    # Monthly P&I payment via standard mortgage formula
    monthly_rate = annual_rate / 12
    n_payments = loan_term.clip(1, 480)
    loan_amt = df["loan_amount"].clip(1000, None)
    df["monthly_payment_estimate"] = (
        loan_amt * (monthly_rate * (1 + monthly_rate) ** n_payments)
        / ((1 + monthly_rate) ** n_payments - 1)
    ).round(2)

    monthly_income = (df["income"] / 12).clip(1, None)
    df["payment_to_income_ratio"] = (df["monthly_payment_estimate"] / monthly_income).round(4)

    # Affordability index: 100 = affordable; higher = less affordable
    # Based on 28% front-end ratio guideline
    df["affordability_index"] = ((df["payment_to_income_ratio"] / 0.28) * 100).round(2)

    logger.debug("Affordability features engineered")
    return df


def engineer_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes credit-risk composite metrics.

    Features added:
      - combined_ltv_dti_score: weighted composite (higher = riskier)
      - risk_tier: categorical low/medium/high based on DTI + LTV
    """
    df = df.copy()

    ltv = df.get("loan_to_value_ratio", pd.Series(80.0, index=df.index)).fillna(80.0)
    dti = df.get("debt_to_income_ratio", pd.Series(36.0, index=df.index)).fillna(36.0)

    # Normalize to 0-1 range (LTV: 50-100; DTI: 5-65)
    ltv_norm = (ltv - 50) / 50
    dti_norm = (dti - 5) / 60
    df["combined_ltv_dti_score"] = (0.4 * ltv_norm + 0.6 * dti_norm).clip(0, 1).round(4)

    def _risk_tier(row: pd.Series) -> str:
        dti_val = row.get("debt_to_income_ratio", 36)
        ltv_val = row.get("loan_to_value_ratio", 80)
        if dti_val < 36 and ltv_val < 80:
            return "low"
        elif dti_val > 50 or ltv_val > 95:
            return "high"
        else:
            return "medium"

    df["risk_tier"] = df.apply(_risk_tier, axis=1)

    logger.debug("Risk features engineered")
    return df


def engineer_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds geographic context features.

    Features added:
      - state_fips: extracted from county_code
      - metro_area_flag: 1 if state is a major metro state
      - median_income_area: state-level median income from FFIEC lookup
      - income_to_area_median_ratio: applicant income relative to area median
    """
    df = df.copy()
    major_metro_states = {"06", "36", "17", "34", "25", "53", "24", "08", "27"}

    if "county_code" in df.columns:
        df["state_fips"] = df["county_code"].astype(str).str[:2]
    else:
        df["state_fips"] = "00"

    df["metro_area_flag"] = df["state_fips"].isin(major_metro_states).astype(int)
    df["median_income_area"] = df["state_fips"].map(_METRO_MEDIAN_INCOME).fillna(_NATIONAL_MEDIAN_INCOME)
    df["income_to_area_median_ratio"] = (df["income"] / df["median_income_area"]).round(4)

    logger.debug("Geographic features engineered")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns:
      - census_tract: frequency encoding (proxy for target encoding without leakage)
      - property_type, loan_purpose, occupancy_type, lien_status: one-hot encoding
      - applicant_age: ordinal encoding
      - risk_tier: ordinal encoding
    """
    df = df.copy()

    # Frequency encode census_tract (high cardinality)
    if "census_tract" in df.columns:
        freq = df["census_tract"].value_counts(normalize=True)
        df["census_tract_freq"] = df["census_tract"].map(freq).fillna(0.0)

    # One-hot encode low-cardinality categoricals
    ohe_cols = []
    for col in ["property_type", "loan_purpose", "occupancy_type", "lien_status",
                "construction_method", "preapproval"]:
        if col in df.columns:
            ohe_cols.append(col)

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols, drop_first=False, dtype=int)

    # Ordinal encode applicant_age
    age_order = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5, "nan": 2}
    if "applicant_age" in df.columns:
        df["applicant_age_ordinal"] = df["applicant_age"].astype(str).map(age_order).fillna(2).astype(int)
    elif "applicant_age_raw" in df.columns:
        bins = [18, 25, 35, 45, 55, 65, 100]
        df["applicant_age_ordinal"] = pd.cut(
            df["applicant_age_raw"], bins=bins, labels=[0, 1, 2, 3, 4, 5], right=False
        ).astype(float).fillna(2).astype(int)

    # Add age group label for fairness analysis
    age_label_map = {0: "18-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65+"}
    if "applicant_age_ordinal" in df.columns:
        df["applicant_age_group"] = df["applicant_age_ordinal"].map(age_label_map)

    # Ordinal encode risk tier
    tier_order = {"low": 0, "medium": 1, "high": 2}
    if "risk_tier" in df.columns:
        df["risk_tier_ordinal"] = df["risk_tier"].map(tier_order).fillna(1).astype(int)

    logger.debug("Categorical encoding complete")
    return df


def _build_target(df: pd.DataFrame) -> pd.Series:
    """Binary target: 1=approved (action_taken 1 or 2), 0=denied (action_taken 3)."""
    if "action_taken" not in df.columns:
        raise ValueError("Column 'action_taken' not found in DataFrame")

    # Keep only originated, approved-not-accepted, and denied
    mask = df["action_taken"].isin([1, 2, 3])
    if mask.sum() < len(df) * 0.5:
        logger.warning("Less than 50% of records have action_taken in {1,2,3}. Using all records.")
        mask = pd.Series(True, index=df.index)

    y = df["action_taken"].map({1: 1, 2: 1, 3: 0, 4: 0, 5: 0}).fillna(0).astype(int)
    return y


def build_feature_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw or partially processed HMDA DataFrame

    Returns:
        Tuple of (X, y, feature_names, protected_cols)
          X: Feature matrix
          y: Binary approval target
          feature_names: List of feature column names
          protected_cols: ECOA protected attribute column names present in X
    """
    logger.info(f"Building feature pipeline on {len(df):,} records...")

    # Build target
    y = _build_target(df)

    # Filter to actionable rows (no withdrawals/incomplete for training)
    actionable_mask = df["action_taken"].isin([1, 2, 3])
    df_filtered = df[actionable_mask].copy()
    y_filtered = y[actionable_mask].copy()

    logger.info(
        f"After filtering to actionable records: {len(df_filtered):,} rows "
        f"({y_filtered.mean():.1%} approval rate)"
    )

    # Pipeline
    df_feat = engineer_affordability_features(df_filtered)
    df_feat = engineer_risk_features(df_feat)
    df_feat = engineer_geographic_features(df_feat)
    df_feat = encode_categoricals(df_feat)

    # Identify protected columns (for fairness analysis, NOT for model training)
    protected_base = ["applicant_sex", "applicant_race_1", "applicant_age_group", "applicant_ethnicity_1"]
    protected_cols = [c for c in protected_base if c in df_feat.columns]

    # Drop columns not used as model features
    drop_cols = [
        "action_taken",
        "census_tract",
        "county_code",
        "applicant_age",
        "risk_tier",
        "state_fips",
    ] + protected_base

    # Numeric feature columns
    numeric_features = [
        "loan_amount",
        "income",
        "debt_to_income_ratio",
        "loan_to_value_ratio",
        "loan_term",
        "co_applicant_present",
        "monthly_payment_estimate",
        "payment_to_income_ratio",
        "affordability_index",
        "combined_ltv_dti_score",
        "risk_tier_ordinal",
        "applicant_age_ordinal",
        "census_tract_freq",
        "metro_area_flag",
        "median_income_area",
        "income_to_area_median_ratio",
        "negative_amortization",
        "interest_only_payments",
        "balloon_payment",
        "open_end_line_of_credit",
        "reverse_mortgage",
        "business_or_commercial_purpose",
        "hoepa_status",
    ]

    # Collect all feature columns (numeric + one-hot dummies)
    ohe_cols = [c for c in df_feat.columns if any(
        c.startswith(p + "_") for p in
        ["property_type", "loan_purpose", "occupancy_type", "lien_status",
         "construction_method", "preapproval"]
    )]

    all_features = numeric_features + ohe_cols
    feature_cols = [c for c in all_features if c in df_feat.columns]

    X = df_feat[feature_cols].fillna(0).astype(float)

    logger.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    logger.info(f"Protected attributes tracked: {protected_cols}")

    return X, y_filtered, list(X.columns), protected_cols

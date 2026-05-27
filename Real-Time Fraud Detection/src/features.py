"""
Feature Engineering Module
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Velocity features, time features, email features, amount features,
categorical encoding, missing value handling, and full feature pipeline.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Domain Risk Scores  (higher = riskier)
# ─────────────────────────────────────────────
EMAIL_DOMAIN_RISK: dict[str, float] = {
    "gmail.com": 0.05,
    "yahoo.com": 0.08,
    "hotmail.com": 0.08,
    "outlook.com": 0.06,
    "icloud.com": 0.05,
    "aol.com": 0.10,
    "protonmail.com": 0.40,
    "guerrillamail.com": 0.90,
    "10minutemail.com": 0.95,
    "anonymous.com": 0.85,
    "mailinator.com": 0.90,
    "trashmail.com": 0.88,
    "temp-mail.org": 0.85,
    "yopmail.com": 0.82,
    "dispostable.com": 0.80,
}
DEFAULT_DOMAIN_RISK = 0.25  # Unknown domains treated as moderately risky


# ─────────────────────────────────────────────
# Velocity Features
# ─────────────────────────────────────────────

def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity features: transaction count and amount deviation per card.

    Windows: 1h, 6h, 24h (based on TransactionDT in seconds).
    Also computes amount deviation from that card's historical average.

    Args:
        df: DataFrame with TransactionDT, TransactionAmt, card1.

    Returns:
        DataFrame with new velocity columns added in-place copy.
    """
    df = df.copy()

    if "TransactionDT" not in df.columns or "card1" not in df.columns:
        logger.warning("TransactionDT or card1 missing — skipping velocity features")
        for suffix in ["1h", "6h", "24h"]:
            df[f"tx_count_{suffix}"] = 0
            df[f"amt_sum_{suffix}"] = 0.0
        df["amt_deviation_from_card_avg"] = 0.0
        return df

    # Sort by card + time for rolling operations
    df_sorted = df.sort_values(["card1", "TransactionDT"]).copy()
    df_sorted["_row_idx"] = np.arange(len(df_sorted))

    windows_seconds = {"1h": 3_600, "6h": 21_600, "24h": 86_400}

    for label, window_sec in windows_seconds.items():
        counts = np.zeros(len(df_sorted), dtype=np.int32)
        amt_sums = np.zeros(len(df_sorted), dtype=np.float32)

        card1_vals = df_sorted["card1"].values
        dt_vals = df_sorted["TransactionDT"].values
        amt_vals = df_sorted["TransactionAmt"].fillna(0).values

        # Two-pointer approach per card group
        card_groups = df_sorted.groupby("card1", sort=False).indices

        for _card, idxs in card_groups.items():
            idxs_sorted = idxs[np.argsort(dt_vals[idxs])]
            left = 0
            for right in range(len(idxs_sorted)):
                r_idx = idxs_sorted[right]
                # Advance left pointer to maintain window
                while dt_vals[r_idx] - dt_vals[idxs_sorted[left]] > window_sec:
                    left += 1
                # Number of transactions in window (excluding current)
                counts[r_idx] = right - left
                amt_sums[r_idx] = float(np.sum(amt_vals[idxs_sorted[left:right]]))

        df_sorted[f"tx_count_{label}"] = counts
        df_sorted[f"amt_sum_{label}"] = amt_sums.astype(np.float32)

    # Amount deviation from card average
    card_avg_amt = df_sorted.groupby("card1")["TransactionAmt"].transform("mean")
    card_std_amt = df_sorted.groupby("card1")["TransactionAmt"].transform("std").fillna(1.0)
    card_std_amt = card_std_amt.replace(0, 1.0)
    df_sorted["amt_deviation_from_card_avg"] = (
        (df_sorted["TransactionAmt"].fillna(0) - card_avg_amt) / card_std_amt
    ).astype(np.float32)

    # Restore original order
    df_sorted = df_sorted.sort_values("_row_idx").drop(columns=["_row_idx"])
    df_sorted.index = df.index

    logger.debug(f"Velocity features added: tx_count_1h/6h/24h, amt_sum_1h/6h/24h, amt_deviation_from_card_avg")
    return df_sorted


# ─────────────────────────────────────────────
# Time Features
# ─────────────────────────────────────────────

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from TransactionDT.

    Features: hour_of_day, day_of_week, is_weekend, is_night (10 PM – 6 AM).

    Args:
        df: DataFrame with TransactionDT (seconds from epoch).

    Returns:
        DataFrame with new time columns.
    """
    df = df.copy()

    if "TransactionDT" not in df.columns:
        logger.warning("TransactionDT missing — adding zero-valued time features")
        for col in ["hour_of_day", "day_of_week", "is_weekend", "is_night"]:
            df[col] = 0
        return df

    dt = df["TransactionDT"].fillna(0).astype(np.int64)
    seconds_in_day = 86_400
    seconds_in_week = 604_800

    df["hour_of_day"] = ((dt % seconds_in_day) // 3_600).astype(np.int8)
    df["day_of_week"] = ((dt % seconds_in_week) // seconds_in_day).astype(np.int8)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_night"] = (
        ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6))
    ).astype(np.int8)

    logger.debug("Time features added: hour_of_day, day_of_week, is_weekend, is_night")
    return df


# ─────────────────────────────────────────────
# Email Features
# ─────────────────────────────────────────────

def engineer_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute email domain match and domain-level risk scores.

    Features:
    - email_domain_match: 1 if P_emaildomain == R_emaildomain else 0
    - p_email_risk_score: risk score for purchaser email domain
    - r_email_risk_score: risk score for recipient email domain

    Args:
        df: DataFrame with P_emaildomain, R_emaildomain columns.

    Returns:
        DataFrame with new email feature columns.
    """
    df = df.copy()

    p_col = "P_emaildomain"
    r_col = "R_emaildomain"

    if p_col not in df.columns:
        df[p_col] = "missing"
    if r_col not in df.columns:
        df[r_col] = "missing"

    # Cast to str first to avoid Categorical dtype setitem errors
    p_domain = df[p_col].astype(str).replace("nan", "missing")
    r_domain = df[r_col].astype(str).replace("nan", "missing")

    df["email_domain_match"] = (p_domain == r_domain).astype(np.int8)

    df["p_email_risk_score"] = (
        p_domain.map(EMAIL_DOMAIN_RISK).fillna(DEFAULT_DOMAIN_RISK).astype(np.float32)
    )
    df["r_email_risk_score"] = (
        r_domain.map(EMAIL_DOMAIN_RISK).fillna(DEFAULT_DOMAIN_RISK).astype(np.float32)
    )
    df["max_email_risk_score"] = df[["p_email_risk_score", "r_email_risk_score"]].max(axis=1).astype(np.float32)

    logger.debug("Email features added: email_domain_match, p/r/max_email_risk_score")
    return df


# ─────────────────────────────────────────────
# Amount Features
# ─────────────────────────────────────────────

def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer transaction amount features.

    Features:
    - log_amount: log1p(TransactionAmt)
    - amount_rounded: 1 if amount ends in .00 or .99
    - amount_zscore_per_card: z-score of amount relative to card1 distribution

    Args:
        df: DataFrame with TransactionAmt and card1.

    Returns:
        DataFrame with new amount feature columns.
    """
    df = df.copy()

    if "TransactionAmt" not in df.columns:
        df["TransactionAmt"] = 0.0

    amt = df["TransactionAmt"].fillna(0).astype(np.float64)

    df["log_amount"] = np.log1p(amt).astype(np.float32)

    # Amount ends in .00 or .99
    decimals = np.round(amt % 1, 2)
    df["amount_rounded"] = ((decimals == 0.00) | (decimals == 0.99)).astype(np.int8)

    # Z-score per card
    if "card1" in df.columns:
        card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
        card_std = df.groupby("card1")["TransactionAmt"].transform("std").fillna(1.0).replace(0, 1.0)
        df["amount_zscore_per_card"] = ((amt - card_mean) / card_std).astype(np.float32)
    else:
        global_mean = amt.mean()
        global_std = max(amt.std(), 1e-8)
        df["amount_zscore_per_card"] = ((amt - global_mean) / global_std).astype(np.float32)

    logger.debug("Amount features added: log_amount, amount_rounded, amount_zscore_per_card")
    return df


# ─────────────────────────────────────────────
# Categorical Encoding
# ─────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str = "isFraud",
    high_cardinality_threshold: int = 20,
) -> pd.DataFrame:
    """
    Encode categorical features.

    - High-cardinality (> threshold unique values): target encoding (mean fraud rate per category).
    - Low-cardinality: label encoding.

    Args:
        df: DataFrame containing cat_cols and target_col.
        cat_cols: List of categorical column names to encode.
        target_col: Target column for target encoding.
        high_cardinality_threshold: Cardinality cutoff.

    Returns:
        DataFrame with encoded categorical columns (suffixed _enc for clarity).
    """
    df = df.copy()

    for col in cat_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — skipping encoding")
            continue

        col_str = df[col].astype(str).fillna("missing")
        n_unique = col_str.nunique()

        enc_col = f"{col}_enc"

        if n_unique > high_cardinality_threshold and target_col in df.columns:
            # Target encoding: mean fraud rate per category
            target_mean = df.groupby(col_str)[target_col].transform("mean")
            df[enc_col] = target_mean.fillna(df[target_col].mean()).astype(np.float32)
            logger.debug(f"Target-encoded '{col}' ({n_unique} unique values)")
        else:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(col_str).astype(np.int16)
            logger.debug(f"Label-encoded '{col}' ({n_unique} unique values)")

    return df


# ─────────────────────────────────────────────
# Missing Value Handling
# ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Numeric columns: fill NaN with -999 (tree models handle sentinel values natively).
    - Categorical/object columns: fill NaN with 'missing'.

    Args:
        df: DataFrame with potential NaN values.

    Returns:
        DataFrame with no NaN values.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_null_counts = df[numeric_cols].isnull().sum()
    cat_null_counts = df[cat_cols].isnull().sum()

    n_numeric_missing = int(numeric_null_counts[numeric_null_counts > 0].sum())
    n_cat_missing = int(cat_null_counts[cat_null_counts > 0].sum())

    if n_numeric_missing > 0:
        df[numeric_cols] = df[numeric_cols].fillna(-999)
        logger.debug(f"Filled {n_numeric_missing:,} numeric NaN with -999")

    if n_cat_missing > 0:
        for col in cat_cols:
            if df[col].dtype.name == "category":
                if "missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("missing")
            df[col] = df[col].fillna("missing")
        logger.debug(f"Filled {n_cat_missing:,} categorical NaN with 'missing'")

    return df


# ─────────────────────────────────────────────
# Full Feature Pipeline
# ─────────────────────────────────────────────

def build_feature_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Orchestrate all feature engineering steps.

    Pipeline:
    1. engineer_velocity_features
    2. engineer_time_features
    3. engineer_email_features
    4. engineer_amount_features
    5. encode_categoricals
    6. handle_missing

    Args:
        df: Raw DataFrame from data_loader.

    Returns:
        Tuple of (processed_df, feature_cols) where feature_cols is the final
        list of numeric feature columns ready for model training.
    """
    logger.info("Starting feature engineering pipeline")

    # Step 1 – Velocity
    df = engineer_velocity_features(df)

    # Step 2 – Time
    df = engineer_time_features(df)

    # Step 3 – Email
    df = engineer_email_features(df)

    # Step 4 – Amount
    df = engineer_amount_features(df)

    # Step 5 – Categoricals
    cat_cols_to_encode = [
        "ProductCD", "card4", "card6",
        "P_emaildomain", "R_emaildomain",
    ]
    existing_cat_cols = [c for c in cat_cols_to_encode if c in df.columns]
    df = encode_categoricals(df, existing_cat_cols)

    # Step 6 – Missing values
    df = handle_missing(df)

    # ── Assemble final feature list ──
    # Exclude non-feature columns
    exclude_cols = {
        "TransactionID", "isFraud",
        "TransactionDT",           # keep derived time features, drop raw DT
        "ProductCD", "card4", "card6",
        "P_emaildomain", "R_emaildomain",
    }

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and df[col].dtype != object
        and df[col].dtype.name != "category"
    ]

    logger.info(f"Feature engineering complete: {len(feature_cols)} features built")
    return df, feature_cols


# ─────────────────────────────────────────────
# Top Feature Names
# ─────────────────────────────────────────────

def get_feature_importance_names() -> List[str]:
    """
    Return the 50 most predictive feature names based on domain knowledge
    and empirical importance from IEEE-CIS benchmarks.

    Returns:
        List of 50 feature name strings.
    """
    return [
        # Amount-based (top predictors)
        "TransactionAmt", "log_amount", "amount_zscore_per_card",
        "amount_rounded", "amt_deviation_from_card_avg",
        # Time-based
        "hour_of_day", "day_of_week", "is_night", "is_weekend",
        # Velocity
        "tx_count_1h", "tx_count_6h", "tx_count_24h",
        "amt_sum_1h", "amt_sum_6h", "amt_sum_24h",
        # Email
        "email_domain_match", "p_email_risk_score",
        "r_email_risk_score", "max_email_risk_score",
        # Card features
        "card1", "card2", "card3", "card5",
        "card4_enc", "card6_enc",
        # Address
        "addr1", "addr2",
        # Count features (highly predictive in IEEE-CIS)
        "C1", "C2", "C3", "C4", "C5", "C6",
        "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
        # Time delta features
        "D1", "D2", "D3", "D4", "D5",
        # Match features
        "M1_enc", "M2_enc", "M3_enc", "M4_enc", "M5_enc",
        # Vesta top features (empirically ranked)
        "V258", "V257",
    ]

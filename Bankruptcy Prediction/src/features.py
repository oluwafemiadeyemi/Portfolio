"""
features.py
===========
Feature engineering for Financial Distress / Bankruptcy Prediction.

Functions:
  - Ratio validation and imputation
  - Altman Z-score component engineering
  - Year-over-year trend features
  - Peer-relative percentile features (sector benchmarking)
  - News sentiment features
  - Multi-horizon target construction (3/6/12/18 months)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Financial ratio validation and engineering
# ---------------------------------------------------------------------------

# Physiologically / financially plausible bounds per ratio
RATIO_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ROA": (-0.50, 0.50),
    "ROE": (-1.0, 1.0),
    "net_profit_margin": (-0.50, 0.60),
    "EBITDA_margin": (-0.50, 0.60),
    "current_ratio": (0.01, 20.0),
    "quick_ratio": (0.01, 15.0),
    "cash_ratio": (0.0, 10.0),
    "debt_to_equity": (0.0, 50.0),
    "debt_to_assets": (0.0, 1.50),
    "interest_coverage": (-10.0, 100.0),
    "asset_turnover": (0.0, 10.0),
    "inventory_turnover": (0.0, 200.0),
    "receivables_turnover": (0.0, 200.0),
    "revenue_growth_yoy": (-1.0, 5.0),
    "asset_growth_yoy": (-1.0, 3.0),
    "PE_ratio": (-500.0, 500.0),
    "market_to_book": (0.0, 100.0),
    "news_sentiment_30d": (-1.0, 1.0),
}


def engineer_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clip outliers, and fill missing financial ratios.

    Steps:
      1. Clip values to plausible financial bounds (prevents model pollution)
      2. Fill NaN with sector median (if sector column exists), else global median
      3. Replace infinities with NaN before processing

    Parameters
    ----------
    df: Input DataFrame with financial ratio columns.

    Returns
    -------
    Cleaned DataFrame.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col, (low, high) in RATIO_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)

    # Fill NaN with sector median, or global median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"distress_flag", "year", "years_of_data"}
    fill_cols = [c for c in numeric_cols if c not in exclude]

    if "sector" in df.columns:
        for col in fill_cols:
            sector_medians = df.groupby("sector")[col].transform("median")
            df[col] = df[col].fillna(sector_medians)

    # Remaining NaN → global median
    for col in fill_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info("Financial ratios validated: %d columns processed.", len(fill_cols))
    return df


# ---------------------------------------------------------------------------
# Altman Z-Score component engineering
# ---------------------------------------------------------------------------

def engineer_altman_components(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all five Altman Z-Score components are present as separate features.

    Components:
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Value of Equity / Total Liabilities
      X5 = Sales / Total Assets

    Also adds: ``altman_safe_zone``, ``altman_grey_zone``, ``altman_distress_zone`` flags.
    """
    df = df.copy()

    # Map to standard names if present
    component_map = {
        "altman_wc_ta": "X1_wc_ta",
        "altman_re_ta": "X2_re_ta",
        "altman_ebit_ta": "X3_ebit_ta",
        "altman_mve_tl": "X4_mve_tl",
        "altman_sales_ta": "X5_sales_ta",
    }
    for src, dst in component_map.items():
        if src in df.columns:
            df[dst] = df[src]
        else:
            df[dst] = np.nan

    # Weighted contributions (for feature importance context)
    df["altman_X1_weighted"] = 1.2 * df["X1_wc_ta"].fillna(0)
    df["altman_X2_weighted"] = 1.4 * df["X2_re_ta"].fillna(0)
    df["altman_X3_weighted"] = 3.3 * df["X3_ebit_ta"].fillna(0)
    df["altman_X4_weighted"] = 0.6 * df["X4_mve_tl"].fillna(0)
    df["altman_X5_weighted"] = 1.0 * df["X5_sales_ta"].fillna(0)

    # Zone flags
    if "altman_zscore" in df.columns:
        df["altman_safe_zone"] = (df["altman_zscore"] > 2.99).astype(int)
        df["altman_grey_zone"] = ((df["altman_zscore"] >= 1.81) & (df["altman_zscore"] <= 2.99)).astype(int)
        df["altman_distress_zone"] = (df["altman_zscore"] < 1.81).astype(int)

    return df


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------

def engineer_trend_features(
    df: pd.DataFrame,
    company_col: str = "company_id",
    year_col: str = "year",
) -> pd.DataFrame:
    """Compute year-over-year changes in key financial ratios.

    Creates: ROA_delta, ROE_delta, leverage_delta, liquidity_delta,
             profitability_trend_3yr (linear slope over 3 years).
    """
    df = df.copy().sort_values([company_col, year_col])

    key_trend_cols = [
        "ROA", "ROE", "current_ratio", "debt_to_equity",
        "net_profit_margin", "revenue_growth_yoy", "altman_zscore",
    ]
    existing = [c for c in key_trend_cols if c in df.columns]

    for col in existing:
        df[f"{col}_delta"] = df.groupby(company_col)[col].diff(1)
        df[f"{col}_delta_2yr"] = df.groupby(company_col)[col].diff(2)
        # 3-year rolling slope
        df[f"{col}_trend_3yr"] = (
            df.groupby(company_col)[col]
            .transform(lambda x: x.rolling(3, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) >= 2 else np.nan,
                raw=True,
            ))
        )

    # High-level composite changes
    if "debt_to_equity" in df.columns and "current_ratio" in df.columns:
        df["leverage_delta"] = df.get("debt_to_equity_delta", pd.Series(np.zeros(len(df))))
        df["liquidity_delta"] = df.get("current_ratio_delta", pd.Series(np.zeros(len(df))))

    logger.info("Trend features engineered: %d delta/slope columns.", len(existing) * 3)
    return df


# ---------------------------------------------------------------------------
# Peer-relative features (sector benchmarking)
# ---------------------------------------------------------------------------

def engineer_peer_relative_features(
    df: pd.DataFrame,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Express each ratio as its percentile rank within its sector.

    Creates ``{col}_sector_pct`` columns — how the company compares to peers.
    A very low percentile in profitability metrics or very high percentile in
    leverage metrics are early warning signs.
    """
    df = df.copy()

    if sector_col not in df.columns:
        logger.warning("Sector column '%s' absent — skipping peer-relative features.", sector_col)
        return df

    peer_cols = [
        "ROA", "ROE", "EBITDA_margin", "current_ratio", "quick_ratio",
        "debt_to_equity", "debt_to_assets", "asset_turnover",
        "revenue_growth_yoy", "altman_zscore",
    ]
    existing = [c for c in peer_cols if c in df.columns]

    for col in existing:
        df[f"{col}_sector_pct"] = df.groupby(sector_col)[col].transform(
            lambda x: x.rank(pct=True, na_option="midpoint")
        )

    logger.info("Peer-relative features: %d percentile columns.", len(existing))
    return df


# ---------------------------------------------------------------------------
# News sentiment features
# ---------------------------------------------------------------------------

def engineer_news_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer news sentiment-based features.

    Input column: ``news_sentiment_30d`` (−1 to +1).

    Creates:
      - ``sentiment_ma30``         : same as input (already 30-day moving average)
      - ``sentiment_momentum``     : current vs. 90-day average (within-company)
      - ``negative_news_spike``    : flag if sentiment < −0.5
      - ``sentiment_volatility``   : rolling std of sentiment (if panel data)
    """
    df = df.copy()

    if "news_sentiment_30d" not in df.columns:
        logger.warning("news_sentiment_30d not found — creating neutral placeholders.")
        df["news_sentiment_30d"] = 0.0

    df["sentiment_ma30"] = df["news_sentiment_30d"]
    df["negative_news_spike"] = (df["news_sentiment_30d"] < -0.5).astype(int)
    df["positive_news_signal"] = (df["news_sentiment_30d"] > 0.5).astype(int)

    # Sentiment momentum (vs. company's own historical mean)
    if "company_id" in df.columns:
        df["sentiment_company_mean"] = df.groupby("company_id")["news_sentiment_30d"].transform("mean")
        df["sentiment_momentum"] = df["news_sentiment_30d"] - df["sentiment_company_mean"]
        df = df.drop(columns=["sentiment_company_mean"])
    else:
        df["sentiment_momentum"] = 0.0

    logger.info("News sentiment features engineered.")
    return df


# ---------------------------------------------------------------------------
# Multi-horizon target builder + full pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    df: pd.DataFrame,
    company_col: str = "company_id",
    year_col: str = "year",
    sector_col: str = "sector",
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str]]:
    """Build complete feature matrix with multi-horizon distress targets.

    Horizons: 3, 6, 12, 18 months ahead (approximated as 1/2 year lead
    for panel data with annual observations).

    Parameters
    ----------
    df: Raw financial panel DataFrame.

    Returns
    -------
    Tuple[X, y_dict, feature_names]:
      - X: Feature matrix DataFrame
      - y_dict: Dict of {``distress_3m``, ``distress_6m``, ``distress_12m``, ``distress_18m``}
      - feature_names: List of column names in X
    """
    df_proc = engineer_financial_ratios(df)
    df_proc = engineer_altman_components(df_proc)
    df_proc = engineer_trend_features(df_proc, company_col, year_col)
    df_proc = engineer_peer_relative_features(df_proc, sector_col)
    df_proc = engineer_news_sentiment_features(df_proc)

    # Multi-horizon targets
    # For annual data: 3m ≈ same year, 6m ≈ same year, 12m ≈ next year, 18m ≈ next year
    if "distress_flag" in df_proc.columns:
        df_proc["distress_3m"] = df_proc["distress_flag"]
        df_proc["distress_6m"] = df_proc["distress_flag"]

        if company_col in df_proc.columns and year_col in df_proc.columns:
            df_proc = df_proc.sort_values([company_col, year_col])
            df_proc["distress_12m"] = df_proc.groupby(company_col)["distress_flag"].shift(-1).fillna(0).astype(int)
            df_proc["distress_18m"] = df_proc.groupby(company_col)["distress_flag"].shift(-2).fillna(0).astype(int)
        else:
            df_proc["distress_12m"] = df_proc["distress_flag"]
            df_proc["distress_18m"] = df_proc["distress_flag"]

    # Feature columns — exclude identifiers and targets
    exclude_cols = {
        company_col, year_col, sector_col, "distress_flag",
        "distress_3m", "distress_6m", "distress_12m", "distress_18m",
        "altman_zone",
    }
    feature_names = [
        c for c in df_proc.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    X = df_proc[feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with medians
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y_dict: Dict[str, pd.Series] = {}
    for target in ["distress_3m", "distress_6m", "distress_12m", "distress_18m"]:
        if target in df_proc.columns:
            y_dict[target] = df_proc[target].reset_index(drop=True)

    logger.info(
        "Feature pipeline complete: %d features, %d samples, %d target horizons.",
        len(feature_names),
        len(X),
        len(y_dict),
    )
    return X.reset_index(drop=True), y_dict, feature_names

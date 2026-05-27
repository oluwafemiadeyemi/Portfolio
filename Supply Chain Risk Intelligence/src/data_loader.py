"""
data_loader.py
==============
Financial Distress / Bankruptcy Prediction — data loading and generation.

Supports:
  - Kaggle / UCI Taiwanese financial distress dataset (3,672 firms)
  - Synthetic US company financial panel data (5,000 firms × 8 years)
  - Altman Z-Score computation
  - Time-series panel feature creation with lagged variables
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load real Kaggle/Taiwanese financial distress dataset
# ---------------------------------------------------------------------------

BANKRUPT_COL_MAP: Dict[str, str] = {
    "Bankrupt": "distress_flag",
    "ROA(C) before interest and depreciation before interest": "roa_c",
    "ROA(A) before interest and % after tax": "roa_a",
    "ROA(B) before interest and depreciation after tax": "roa_b",
    "Operating Gross Margin": "gross_margin",
    "Current Ratio": "current_ratio",
    "Quick Ratio": "quick_ratio",
    "Interest Expense Ratio": "interest_expense_ratio",
    "Total debt/Total net worth": "debt_to_equity",
    "Debt ratio %": "debt_ratio",
    "Net worth/Assets": "equity_ratio",
    "Total Asset Turnover": "asset_turnover",
    "Accounts Receivable Turnover": "receivables_turnover",
    "Inventory Turnover Rate (times)": "inventory_turnover",
    "Cash Flow to Sales": "cfo_to_sales",
    "Cash Flow to Total Assets": "cfo_to_assets",
    "Net Income to Total Assets": "net_income_to_assets",
    "Retained Earnings to Total Assets": "retained_earnings_to_assets",
}


def load_financial_distress(filepath: str | Path) -> pd.DataFrame:
    """Load the Kaggle / UCI Taiwanese financial distress dataset.

    Maps key columns to standardised names used across the project.

    Parameters
    ----------
    filepath: Path to ``bankruptcy.csv`` or ``financial_distress.csv``.

    Returns
    -------
    pd.DataFrame with standardised columns and binary ``distress_flag``.
    """
    filepath = Path(filepath)
    logger.info("Loading financial distress dataset from %s", filepath)

    df = pd.read_csv(filepath, low_memory=False)

    # Standardise the target column — it may be 'Bankrupt', 'Bankrupt?', 'distress'
    target_candidates = ["Bankrupt?", "Bankrupt", "bankrupt", "distress", "Distress"]
    for cand in target_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: "distress_flag"})
            break

    # Convert Yes/No or 1/0 target to binary int
    if "distress_flag" in df.columns:
        col = df["distress_flag"]
        if col.dtype == object:
            df["distress_flag"] = col.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
        else:
            df["distress_flag"] = col.fillna(0).astype(int)

    # Rename known columns
    rename_map = {k: v for k, v in BANKRUPT_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Clean up: replace infinities with NaN, coerce all non-id columns to numeric
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in df.columns if c not in ("company_id", "sector", "distress_flag")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Add dummy identifiers if absent
    if "company_id" not in df.columns:
        df["company_id"] = [f"TW_{i:05d}" for i in range(len(df))]
    if "year" not in df.columns:
        df["year"] = 2020  # single-year snapshot

    logger.info(
        "Loaded %d rows, %d columns. Distress rate: %.1f%%",
        len(df),
        df.shape[1],
        df["distress_flag"].mean() * 100 if "distress_flag" in df.columns else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Synthetic US company financial panel data generator
# ---------------------------------------------------------------------------

def generate_synthetic_financial_data(
    n_companies: int = 5000,
    years: int = 8,
    start_year: int = 2016,
    distress_rate: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic US company financial panel dataset.

    Parameters
    ----------
    n_companies: Number of unique companies.
    years: Number of years in the panel (2016–2023 by default).
    start_year: First year of the panel.
    distress_rate: Target bankruptcy / distress prevalence (~8%).
    random_state: RNG seed.

    Returns
    -------
    pd.DataFrame with one row per (company, year) — n_companies × years rows.
    """
    rng = np.random.default_rng(random_state)

    sectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrial"]
    sector_weights = [0.20, 0.18, 0.15, 0.12, 0.20, 0.15]

    records: List[Dict] = []

    for comp_idx in range(n_companies):
        comp_id = f"US_{comp_idx + 1:05d}"
        sector = rng.choice(sectors, p=sector_weights)

        # Determine whether this company eventually goes into distress
        is_distress_company = rng.random() < distress_rate
        # Distress year (if applicable) — uniformly distributed across years
        distress_year = int(rng.integers(start_year + 2, start_year + years)) if is_distress_company else None

        # Company-level "true" financial health (stable underlying level)
        if is_distress_company:
            roa_base = rng.uniform(-0.10, 0.05)
            roe_base = rng.uniform(-0.20, 0.10)
            current_ratio_base = rng.uniform(0.5, 1.5)
            debt_to_equity_base = rng.uniform(2.0, 8.0)
            altman_wc_ta_base = rng.uniform(-0.10, 0.10)
            altman_re_ta_base = rng.uniform(-0.05, 0.15)
            sentiment_base = rng.uniform(-0.6, 0.0)
        else:
            roa_base = rng.uniform(0.02, 0.20)
            roe_base = rng.uniform(0.05, 0.35)
            current_ratio_base = rng.uniform(1.2, 4.5)
            debt_to_equity_base = rng.uniform(0.1, 3.0)
            altman_wc_ta_base = rng.uniform(0.10, 0.35)
            altman_re_ta_base = rng.uniform(0.10, 0.40)
            sentiment_base = rng.uniform(0.0, 0.8)

        pe_ratio = rng.uniform(5, 100) if rng.random() > 0.3 else np.nan
        market_to_book = rng.uniform(0.5, 10.0)

        for yr_idx in range(years):
            year = start_year + yr_idx

            # Is this year a distress year?
            is_distress_this_year = is_distress_company and distress_year is not None and year >= distress_year

            noise = 0.10  # ±10% year-over-year noise

            roa = float(np.clip(roa_base * rng.uniform(1 - noise, 1 + noise), -0.15, 0.25))
            roe = float(np.clip(roe_base * rng.uniform(1 - noise, 1 + noise), -0.30, 0.40))
            ebitda_margin = float(np.clip(roa + rng.uniform(0.02, 0.08), -0.10, 0.35))
            net_profit_margin = float(np.clip(roa * 0.8 + rng.normal(0, 0.02), -0.20, 0.30))
            current_ratio = float(np.clip(current_ratio_base * rng.uniform(1 - noise, 1 + noise), 0.3, 5.0))
            quick_ratio = float(np.clip(current_ratio * rng.uniform(0.6, 0.9), 0.2, 4.0))
            cash_ratio = float(np.clip(quick_ratio * rng.uniform(0.2, 0.6), 0.01, 2.0))
            d2e = float(np.clip(debt_to_equity_base * rng.uniform(1 - noise, 1 + noise), 0.1, 8.0))
            debt_to_assets = float(np.clip(d2e / (1 + d2e) + rng.normal(0, 0.03), 0.1, 0.9))
            interest_coverage = float(np.clip(rng.uniform(0.5, 20), 0.5, 20))

            asset_turnover = float(np.clip(rng.uniform(0.2, 3.0), 0.2, 3.0))
            inventory_turnover = float(np.clip(rng.uniform(1, 20), 1, 20))
            receivables_turnover = float(np.clip(rng.uniform(2, 30), 2, 30))

            revenue_growth = float(np.clip(rng.normal(0.05 if not is_distress_this_year else -0.10, 0.12), -0.30, 0.50))
            asset_growth = float(np.clip(rng.normal(0.03 if not is_distress_this_year else -0.05, 0.10), -0.20, 0.40))

            wc_ta = float(np.clip(altman_wc_ta_base + rng.normal(0, 0.03), -0.30, 0.60))
            re_ta = float(np.clip(altman_re_ta_base + rng.normal(0, 0.02), -0.30, 0.60))
            ebit_ta = float(np.clip(roa + 0.03 + rng.normal(0, 0.01), -0.20, 0.35))
            mve_tl = float(np.clip(market_to_book * (1 - debt_to_assets) / (debt_to_assets + 1e-9) + rng.normal(0, 0.5), 0.01, 20))
            sales_ta = float(np.clip(asset_turnover + rng.normal(0, 0.05), 0.1, 4.0))

            # Altman Z-Score
            z_score = compute_altman_zscore_single(wc_ta, re_ta, ebit_ta, mve_tl, sales_ta)

            sentiment = float(np.clip(
                sentiment_base * rng.uniform(1 - noise, 1 + noise),
                -1.0,
                1.0,
            ))
            if is_distress_this_year:
                sentiment -= rng.uniform(0.1, 0.4)
                sentiment = float(np.clip(sentiment, -1.0, 1.0))

            record = {
                "company_id": comp_id,
                "year": year,
                "sector": sector,
                "years_of_data": yr_idx + 1,
                # Profitability
                "ROA": roa,
                "ROE": roe,
                "net_profit_margin": net_profit_margin,
                "EBITDA_margin": ebitda_margin,
                # Leverage
                "debt_to_equity": d2e,
                "interest_coverage": interest_coverage,
                "debt_to_assets": debt_to_assets,
                # Liquidity
                "current_ratio": current_ratio,
                "quick_ratio": quick_ratio,
                "cash_ratio": cash_ratio,
                # Activity
                "asset_turnover": asset_turnover,
                "inventory_turnover": inventory_turnover,
                "receivables_turnover": receivables_turnover,
                # Growth
                "revenue_growth_yoy": revenue_growth,
                "asset_growth_yoy": asset_growth,
                # Market
                "PE_ratio": pe_ratio,
                "market_to_book": market_to_book,
                # Altman components
                "altman_wc_ta": wc_ta,
                "altman_re_ta": re_ta,
                "altman_ebit_ta": ebit_ta,
                "altman_mve_tl": mve_tl,
                "altman_sales_ta": sales_ta,
                "altman_zscore": z_score,
                # News
                "news_sentiment_30d": sentiment,
                # Target
                "distress_flag": int(is_distress_this_year),
            }
            records.append(record)

    df = pd.DataFrame(records)
    overall_rate = df["distress_flag"].mean() * 100
    logger.info(
        "Generated synthetic panel: %d rows (%d companies × %d years). Distress rate: %.1f%%",
        len(df),
        n_companies,
        years,
        overall_rate,
    )
    return df


def compute_altman_zscore_single(
    wc_ta: float,
    re_ta: float,
    ebit_ta: float,
    mve_tl: float,
    sales_ta: float,
) -> float:
    """Compute Altman Z-score from the five components.

    Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*Sales/TA
    """
    return float(
        1.2 * wc_ta
        + 1.4 * re_ta
        + 3.3 * ebit_ta
        + 0.6 * mve_tl
        + 1.0 * sales_ta
    )


# ---------------------------------------------------------------------------
# Altman Z-Score computation for a DataFrame
# ---------------------------------------------------------------------------

def compute_altman_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Altman Z-score and zone classification for each row.

    Uses the original 1968 manufacturing formula:
      Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*Sales/TA

    Zones:
      - Safe:     Z > 2.99
      - Grey:   1.81 ≤ Z ≤ 2.99
      - Distress: Z < 1.81

    Parameters
    ----------
    df: DataFrame that must contain the Altman component columns or
        equivalents (working_capital_to_assets, retained_earnings_to_assets,
        ebit_to_assets, market_cap_to_liabilities, sales_to_assets).

    Returns
    -------
    df with added columns: ``altman_zscore``, ``altman_zone``.
    """
    df = df.copy()

    # Map possible alternative column names
    wc_ta_col = next((c for c in ["altman_wc_ta", "working_capital_to_assets", "wc_ta"] if c in df.columns), None)
    re_ta_col = next((c for c in ["altman_re_ta", "retained_earnings_to_assets", "re_ta", "Retained_Earnings_to_Total_Assets", "retained_earnings_to_assets"] if c in df.columns), None)
    ebit_ta_col = next((c for c in ["altman_ebit_ta", "ebit_to_assets", "ebit_ta", "ROA"] if c in df.columns), None)
    mve_tl_col = next((c for c in ["altman_mve_tl", "market_cap_to_liabilities", "mve_tl", "market_to_book"] if c in df.columns), None)
    sales_ta_col = next((c for c in ["altman_sales_ta", "sales_to_assets", "sales_ta", "asset_turnover"] if c in df.columns), None)

    # Compute Z if all components available
    components_found = all(c is not None for c in [wc_ta_col, re_ta_col, ebit_ta_col, mve_tl_col, sales_ta_col])

    if components_found:
        df["altman_zscore"] = (
            1.2 * df[wc_ta_col].fillna(0)
            + 1.4 * df[re_ta_col].fillna(0)
            + 3.3 * df[ebit_ta_col].fillna(0)
            + 0.6 * df[mve_tl_col].fillna(0)
            + 1.0 * df[sales_ta_col].fillna(0)
        )
    elif "altman_zscore" not in df.columns:
        logger.warning("Altman Z-score components not found — approximating from available ratios.")
        # Approximate using what we have
        roa = df.get("ROA", pd.Series(np.zeros(len(df)))).fillna(0)
        cr = df.get("current_ratio", pd.Series(np.ones(len(df)))).fillna(1.0)
        d2e = df.get("debt_to_equity", pd.Series(np.ones(len(df)))).fillna(1.0)
        at = df.get("asset_turnover", pd.Series(np.ones(len(df)))).fillna(1.0)
        df["altman_zscore"] = 3.3 * roa + 1.2 * (cr - 1).clip(-0.5, 0.5) + 0.6 / (d2e + 1e-9) + at

    # Zone classification
    def _zone(z: float) -> str:
        if z > 2.99:
            return "Safe"
        elif z >= 1.81:
            return "Grey"
        else:
            return "Distress"

    df["altman_zone"] = df["altman_zscore"].apply(_zone)
    logger.info("Altman Z-score computed. Zone distribution:\n%s", df["altman_zone"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Try to load real financial distress data; fall back to synthetic generation.

    Search order:
      1. ``data_dir/financial_distress.csv``
      2. ``data_dir/bankruptcy.csv``
      3. ``data_dir/raw/financial_distress.csv``
      4. ``data_dir/raw/bankruptcy.csv``
      5. Synthetic generation
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "financial_distress.csv",
        data_dir / "bankruptcy.csv",
        data_dir / "raw" / "financial_distress.csv",
        data_dir / "raw" / "bankruptcy.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Found data file: %s", candidate)
            try:
                df = load_financial_distress(candidate)
                df = compute_altman_zscore(df)
                return df
            except Exception as exc:
                logger.warning("Failed to load %s: %s — falling back.", candidate, exc)

    logger.info("No real data found. Generating synthetic dataset.")
    df = generate_synthetic_financial_data()
    df = compute_altman_zscore(df)
    return df


# ---------------------------------------------------------------------------
# Panel / time-series features
# ---------------------------------------------------------------------------

def create_time_series_features(
    df: pd.DataFrame,
    company_col: str = "company_id",
    year_col: str = "year",
    lag_years: int = 3,
) -> pd.DataFrame:
    """Create lagged features for a panel dataset.

    For each numeric feature, adds columns:
      - ``{feat}_lag1``, ``{feat}_lag2``, ``{feat}_lag3`` (prior year values)
      - ``{feat}_yoy_change`` (year-over-year delta)
      - ``{feat}_2yr_change`` (two-year delta)

    Parameters
    ----------
    df: Panel DataFrame with one row per (company, year).
    company_col: Company identifier column.
    year_col: Year column.
    lag_years: Number of lag years to create.

    Returns
    -------
    df with added lag and change columns. Rows with insufficient history
    retain NaN for lag columns.
    """
    df = df.sort_values([company_col, year_col]).reset_index(drop=True)

    exclude = {company_col, year_col, "sector", "distress_flag", "altman_zone"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    for col in numeric_cols:
        for lag in range(1, lag_years + 1):
            df[f"{col}_lag{lag}"] = df.groupby(company_col)[col].shift(lag)
        df[f"{col}_yoy_change"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_2yr_change"] = df[col] - df.groupby(company_col)[col].shift(2)

    logger.info(
        "Created time-series features: %d new columns (lag %d).",
        len(numeric_cols) * (lag_years + 2),
        lag_years,
    )
    return df

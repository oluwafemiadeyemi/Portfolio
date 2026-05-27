"""
HMDA Mortgage Data Loader
Loads real HMDA 2022 data or generates realistic synthetic mortgage applications.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_hmda_column_mapping() -> dict:
    """Returns dict mapping HMDA code columns to human-readable labels."""
    return {
        "action_taken": {
            1: "loan_originated",
            2: "approved_not_accepted",
            3: "application_denied",
            4: "application_withdrawn",
            5: "file_closed_for_incompleteness",
            6: "purchased_loan",
            7: "preapproval_denied",
            8: "preapproval_approved_not_accepted",
        },
        "loan_purpose": {
            1: "home_purchase",
            2: "home_improvement",
            31: "refinancing",
            32: "cash_out_refinancing",
            4: "other_purpose",
            5: "not_applicable",
        },
        "preapproval": {
            1: "preapproval_requested",
            2: "preapproval_not_requested",
        },
        "construction_method": {
            1: "site_built",
            2: "manufactured_home",
        },
        "occupancy_type": {
            1: "principal_residence",
            2: "second_residence",
            3: "investment_property",
        },
        "lien_status": {
            1: "secured_by_first_lien",
            2: "secured_by_subordinate_lien",
        },
        "applicant_sex": {
            1: "male",
            2: "female",
            3: "information_not_provided",
            4: "not_applicable",
            6: "both_male_and_female",
        },
        "applicant_race_1": {
            1: "american_indian_alaska_native",
            2: "asian",
            3: "black_african_american",
            4: "native_hawaiian_pacific_islander",
            5: "white",
            6: "information_not_provided",
            7: "not_applicable",
        },
        "applicant_ethnicity_1": {
            1: "hispanic_or_latino",
            2: "not_hispanic_or_latino",
            3: "information_not_provided",
            4: "not_applicable",
        },
        "derived_loan_product_type": {
            "Conventional:First Lien": "conventional_first",
            "FHA:First Lien": "fha_first",
            "VA:First Lien": "va_first",
            "FSA/RHS:First Lien": "fsa_rhs_first",
            "Conventional:Subordinate Lien": "conventional_sub",
        },
    }


def filter_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the ~30 most predictive features for mortgage underwriting."""
    desired_cols = [
        "action_taken",
        "loan_amount",
        "income",
        "loan_purpose",
        "property_type",
        "occupancy_type",
        "lien_status",
        "applicant_sex",
        "applicant_race_1",
        "applicant_age",
        "co_applicant_present",
        "debt_to_income_ratio",
        "loan_to_value_ratio",
        "county_code",
        "census_tract",
        "preapproval",
        "construction_method",
        "applicant_ethnicity_1",
        "co_applicant_sex",
        "co_applicant_race_1",
        "loan_term",
        "intro_rate_period",
        "negative_amortization",
        "interest_only_payments",
        "balloon_payment",
        "other_nonamortizing_features",
        "business_or_commercial_purpose",
        "open_end_line_of_credit",
        "reverse_mortgage",
        "hoepa_status",
    ]
    available = [c for c in desired_cols if c in df.columns]
    logger.info(f"Filtering to {len(available)} relevant columns (of {len(df.columns)} total)")
    return df[available].copy()


def load_hmda(filepath: str, max_records: int = 500000) -> pd.DataFrame:
    """
    Loads HMDA 2022 Modified LAR CSV, selects relevant columns, maps codes to labels.

    Args:
        filepath: Path to HMDA CSV file
        max_records: Maximum number of records to load

    Returns:
        Cleaned DataFrame with mapped codes
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HMDA file not found: {filepath}")

    logger.info(f"Loading HMDA data from {filepath} (max {max_records:,} records)")

    # HMDA files are large; use chunked reading
    chunks = []
    chunk_size = 100000
    total_loaded = 0

    for chunk in pd.read_csv(
        filepath,
        chunksize=chunk_size,
        low_memory=False,
        na_values=["NA", "Exempt", "nan", "", " "],
    ):
        chunks.append(chunk)
        total_loaded += len(chunk)
        if total_loaded >= max_records:
            break

    df = pd.concat(chunks, ignore_index=True).head(max_records)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    # Standardize column names (HMDA uses various naming conventions)
    df.columns = [c.lower().strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Map common HMDA column aliases
    col_aliases = {
        "loan_amount_000s": "loan_amount",
        "applicant_income_000s": "income",
        "hud_median_family_income": "area_median_income",
        "rate_spread": "rate_spread",
        "tract_to_msamd_income": "tract_income_ratio",
        "number_of_owner_occupied_units": "owner_occupied_units",
        "number_of_1_to_4_family_units": "family_units",
    }
    df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})

    # Apply code mappings
    mapping = get_hmda_column_mapping()
    for col, code_map in mapping.items():
        if col in df.columns and isinstance(code_map, dict):
            df[col + "_label"] = df[col].map(code_map)

    # Parse numeric columns
    numeric_cols = ["loan_amount", "income", "debt_to_income_ratio", "loan_to_value_ratio", "loan_term"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Scale loan_amount if it's in thousands (HMDA 2017 and older)
    if "loan_amount" in df.columns:
        med = pd.to_numeric(df["loan_amount"], errors="coerce").median()
        if pd.notna(med) and med < 1000:
            df["loan_amount"] = df["loan_amount"] * 1000
            logger.info("Scaled loan_amount from thousands to dollars")

    # Scale income similarly
    if "income" in df.columns:
        med = pd.to_numeric(df["income"], errors="coerce").dropna().median()
        if pd.notna(med) and med < 1000:
            df["income"] = df["income"] * 1000
            logger.info("Scaled income from thousands to dollars")

    df = filter_relevant_columns(df)
    logger.info(f"After column filtering: {df.shape}")
    return df


def generate_synthetic_mortgage_data(n: int = 100000, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic mortgage applications matching HMDA 2022 distributions.

    Args:
        n: Number of synthetic applications to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic mortgage data
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating {n:,} synthetic mortgage applications...")

    # ── Geographic codes ──────────────────────────────────────────────────────
    state_fips = [
        "06", "48", "12", "36", "17", "42", "39", "13", "37", "26",
        "34", "04", "19", "47", "55", "27", "53", "25", "08", "24",
    ]
    county_codes = [f"{s}{str(rng.integers(1, 200)):>03}" for s in rng.choice(state_fips, n)]
    census_tracts = [f"{cc}{str(rng.integers(100000, 999999))}" for cc in county_codes]

    # ── Core financial features ────────────────────────────────────────────────
    # Income: log-normal ~ $90k median, moderate right skew
    log_income = rng.normal(loc=np.log(90000), scale=0.7, size=n)
    income = np.clip(np.exp(log_income), 20000, 800000).astype(int)

    # Loan amount: correlated with income
    loan_amount_multiplier = rng.uniform(2.5, 5.5, n)
    loan_amount_base = income * loan_amount_multiplier
    loan_amount = np.clip(
        (loan_amount_base * rng.lognormal(0, 0.15, n)).astype(int),
        50000,
        2000000,
    )

    # DTI: slightly correlated with income (higher income → slightly lower DTI)
    dti_base = rng.normal(loc=36, scale=10, size=n)
    income_effect = (income - income.mean()) / income.std() * (-3)
    debt_to_income_ratio = np.clip(dti_base + income_effect, 5, 65).round(1)

    # LTV: independent of demographics
    loan_to_value_ratio = np.clip(rng.normal(loc=82, scale=12, size=n), 50, 100).round(1)

    # ── Loan characteristics ───────────────────────────────────────────────────
    loan_purpose = rng.choice(
        [1, 2, 31, 32, 4],
        size=n,
        p=[0.42, 0.08, 0.28, 0.17, 0.05],
    )
    property_type = rng.choice([1, 2, 3], size=n, p=[0.72, 0.12, 0.16])
    occupancy_type = rng.choice([1, 2, 3], size=n, p=[0.82, 0.08, 0.10])
    lien_status = rng.choice([1, 2], size=n, p=[0.90, 0.10])
    loan_term = rng.choice([360, 180, 240, 120], size=n, p=[0.72, 0.14, 0.09, 0.05])

    # ── Protected attributes (ECOA-regulated) ─────────────────────────────────
    applicant_sex = rng.choice([1, 2, 3, 4], size=n, p=[0.52, 0.27, 0.18, 0.03])
    applicant_race_1 = rng.choice([1, 2, 3, 4, 5, 6, 7], size=n, p=[0.01, 0.06, 0.10, 0.01, 0.64, 0.15, 0.03])
    applicant_ethnicity_1 = rng.choice([1, 2, 3, 4], size=n, p=[0.10, 0.74, 0.13, 0.03])

    # Age: realistic distribution of mortgage applicants
    applicant_age_raw = np.clip(rng.normal(loc=42, scale=12, size=n), 18, 80).astype(int)
    age_bins = [18, 25, 35, 45, 55, 65, 80]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    applicant_age = pd.cut(applicant_age_raw, bins=age_bins, labels=age_labels, right=False).astype(str)

    co_applicant_present = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    co_applicant_sex = np.where(co_applicant_present == 1, rng.choice([1, 2, 5], size=n, p=[0.45, 0.45, 0.10]), 5)
    co_applicant_race_1 = np.where(
        co_applicant_present == 1,
        rng.choice([1, 2, 3, 4, 5, 6, 7], size=n, p=[0.01, 0.06, 0.10, 0.01, 0.64, 0.15, 0.03]),
        8,
    )

    # ── Approval outcome (fair: race/sex NOT direct determinants) ────────────
    # Base approval probability from creditworthiness factors only
    approval_logit = (
        2.5
        + 0.000008 * income
        - 0.08 * debt_to_income_ratio
        - 0.04 * (loan_to_value_ratio - 80)
        + 0.3 * (occupancy_type == 1).astype(float)
        + 0.2 * (lien_status == 1).astype(float)
        - 0.2 * (loan_purpose == 2).astype(float)
        + rng.normal(0, 0.3, n)
    )
    approval_prob = 1 / (1 + np.exp(-approval_logit))
    # Ensure overall ~75% approval
    target_mean = 0.75
    current_mean = approval_prob.mean()
    approval_prob = np.clip(approval_prob + (target_mean - current_mean), 0.05, 0.99)

    approved = (rng.uniform(0, 1, n) < approval_prob).astype(int)

    # Action taken: expand binary to realistic 5-way
    action_taken = np.where(
        approved == 1,
        rng.choice([1, 2], size=n, p=[0.85, 0.15]),  # originated vs approved-not-accepted
        rng.choice([3, 4, 5], size=n, p=[0.70, 0.20, 0.10]),  # denied vs withdrawn vs incomplete
    )

    # ── Feature flags ──────────────────────────────────────────────────────────
    preapproval = rng.choice([1, 2], size=n, p=[0.25, 0.75])
    construction_method = rng.choice([1, 2], size=n, p=[0.88, 0.12])
    negative_amortization = rng.choice([1, 2], size=n, p=[0.02, 0.98])
    interest_only_payments = rng.choice([1, 2], size=n, p=[0.05, 0.95])
    balloon_payment = rng.choice([1, 2], size=n, p=[0.03, 0.97])
    open_end_line_of_credit = rng.choice([1, 2], size=n, p=[0.08, 0.92])
    reverse_mortgage = rng.choice([1, 2], size=n, p=[0.02, 0.98])
    business_or_commercial_purpose = rng.choice([1, 2], size=n, p=[0.04, 0.96])
    hoepa_status = rng.choice([1, 2, 3], size=n, p=[0.02, 0.05, 0.93])

    df = pd.DataFrame(
        {
            "action_taken": action_taken,
            "loan_amount": loan_amount,
            "income": income,
            "loan_purpose": loan_purpose,
            "property_type": property_type,
            "occupancy_type": occupancy_type,
            "lien_status": lien_status,
            "applicant_sex": applicant_sex,
            "applicant_race_1": applicant_race_1,
            "applicant_ethnicity_1": applicant_ethnicity_1,
            "applicant_age": applicant_age,
            "applicant_age_raw": applicant_age_raw,
            "co_applicant_present": co_applicant_present,
            "co_applicant_sex": co_applicant_sex,
            "co_applicant_race_1": co_applicant_race_1,
            "debt_to_income_ratio": debt_to_income_ratio,
            "loan_to_value_ratio": loan_to_value_ratio,
            "county_code": county_codes,
            "census_tract": census_tracts,
            "loan_term": loan_term,
            "preapproval": preapproval,
            "construction_method": construction_method,
            "negative_amortization": negative_amortization,
            "interest_only_payments": interest_only_payments,
            "balloon_payment": balloon_payment,
            "open_end_line_of_credit": open_end_line_of_credit,
            "reverse_mortgage": reverse_mortgage,
            "business_or_commercial_purpose": business_or_commercial_purpose,
            "hoepa_status": hoepa_status,
        }
    )

    logger.info(
        f"Synthetic data generated: {len(df):,} rows, "
        f"approval rate = {approved.mean():.1%}, "
        f"median income = ${income.median():,.0f}, "
        f"median loan = ${loan_amount.median():,.0f}"
    )
    return df


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Attempts to load real HMDA data, falls back to synthetic generation.

    Args:
        data_dir: Directory to search for HMDA data files

    Returns:
        DataFrame with mortgage application data
    """
    data_path = Path(data_dir)
    candidates = [
        data_path / "raw" / "2022_public_lar.csv",
        data_path / "raw" / "hmda_2022.csv",
        data_path / "raw" / "hmda_2022_TX.csv",
        data_path / "raw" / "hmda_lar_2022.csv",
        data_path / "2022_public_lar.csv",
        data_path / "hmda_2022.csv",
        data_path / "hmda_2022_TX.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Found HMDA data at: {candidate}")
            try:
                df = load_hmda(str(candidate))
                logger.info(f"Successfully loaded real HMDA data: {len(df):,} records")
                return df
            except Exception as exc:
                logger.warning(f"Failed to load {candidate}: {exc}. Trying next...")

    logger.warning("No HMDA data file found. Generating synthetic mortgage data (100,000 applications).")
    df = generate_synthetic_mortgage_data(n=100000)

    # Save synthetic data for reproducibility
    raw_dir = data_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = raw_dir / "synthetic_hmda_2022.csv"
    df.to_csv(synthetic_path, index=False)
    logger.info(f"Synthetic data saved to: {synthetic_path}")

    return df

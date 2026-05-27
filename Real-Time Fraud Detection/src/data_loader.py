"""
Data Loader Module
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Handles loading IEEE-CIS, Credit Card 2023, and synthetic fraud data.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# IEEE-CIS Fraud Detection Dataset
# ─────────────────────────────────────────────

def load_ieee_cis(trans_path: str | Path, identity_path: str | Path) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS transaction + identity files on TransactionID.

    Args:
        trans_path: Path to train_transaction.csv
        identity_path: Path to train_identity.csv

    Returns:
        Merged DataFrame with all features and isFraud target.
    """
    trans_path = Path(trans_path)
    identity_path = Path(identity_path)

    logger.info(f"Loading IEEE-CIS transactions from {trans_path}")
    transactions = pd.read_csv(
        trans_path,
        dtype={
            "TransactionID": np.int32,
            "isFraud": np.int8,
            "TransactionDT": np.int32,
            "TransactionAmt": np.float32,
            "ProductCD": "category",
            "card1": np.float32,
            "card2": np.float32,
            "card3": np.float32,
            "card4": "category",
            "card5": np.float32,
            "card6": "category",
            "addr1": np.float32,
            "addr2": np.float32,
            "P_emaildomain": "category",
            "R_emaildomain": "category",
        },
    )
    logger.info(f"Loaded {len(transactions):,} transaction rows")

    logger.info(f"Loading IEEE-CIS identity data from {identity_path}")
    identity = pd.read_csv(identity_path, dtype={"TransactionID": np.int32})
    logger.info(f"Loaded {len(identity):,} identity rows")

    logger.info("Merging transactions and identity on TransactionID")
    merged = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(
        f"Merged dataset: {len(merged):,} rows × {merged.shape[1]} columns | "
        f"Fraud rate: {merged['isFraud'].mean():.4%}"
    )
    return merged


# ─────────────────────────────────────────────
# Credit Card Fraud 2023 Dataset
# ─────────────────────────────────────────────

def load_creditcard_2023(filepath: str | Path) -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection Dataset 2023.

    Expected columns: id, V1-V28, Amount, Class (0/1).
    Renames 'Class' → 'isFraud' and 'Amount' → 'TransactionAmt' for
    pipeline consistency.

    Args:
        filepath: Path to creditcard_2023.csv

    Returns:
        DataFrame with standardised column names.
    """
    filepath = Path(filepath)
    logger.info(f"Loading Credit Card 2023 dataset from {filepath}")

    df = pd.read_csv(filepath)
    rename_map: dict[str, str] = {}
    if "Class" in df.columns:
        rename_map["Class"] = "isFraud"
    if "Amount" in df.columns:
        rename_map["Amount"] = "TransactionAmt"
    if rename_map:
        df = df.rename(columns=rename_map)

    logger.info(
        f"Loaded {len(df):,} rows × {df.shape[1]} columns | "
        f"Fraud rate: {df['isFraud'].mean():.4%}"
    )
    return df


# ─────────────────────────────────────────────
# Synthetic Fraud Data Generator
# ─────────────────────────────────────────────

def generate_synthetic_fraud_data(
    n: int = 500_000,
    fraud_rate: float = 0.035,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic fraud data with the IEEE-CIS feature schema.

    Fraud signal is injected via:
    - Higher transaction amounts (log-normal with shifted mean)
    - Unusual hours (10 PM – 6 AM)
    - Mismatched P/R email domains
    - Elevated C/D features

    Args:
        n: Total number of transactions to generate.
        fraud_rate: Proportion of fraudulent transactions (default 3.5%).
        random_state: NumPy random seed for reproducibility.

    Returns:
        DataFrame with isFraud target and all feature columns.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    labels = np.concatenate([np.ones(n_fraud, dtype=np.int8), np.zeros(n_legit, dtype=np.int8)])

    logger.info(f"Generating {n:,} synthetic transactions ({n_fraud:,} fraud, {n_legit:,} legit)")

    # ── TransactionDT: seconds from a reference epoch (30-day window) ──
    # Legitimate: uniform over day
    # Fraud: skewed toward night hours
    legit_dt = rng.integers(0, 30 * 86_400, size=n_legit)
    # Fraud hours concentrated between 22:00–06:00 (mapped to seconds in day)
    fraud_hour = rng.choice(
        np.concatenate([np.arange(0, 6), np.arange(22, 24)]),
        size=n_fraud,
    )
    fraud_dt = (
        rng.integers(0, 30, size=n_fraud) * 86_400
        + fraud_hour * 3_600
        + rng.integers(0, 3_600, size=n_fraud)
    )
    TransactionDT = np.concatenate([fraud_dt, legit_dt]).astype(np.int32)

    # ── TransactionAmt ──
    legit_amt = rng.lognormal(mean=3.5, sigma=1.2, size=n_legit).clip(1, 25_000)
    fraud_amt = rng.lognormal(mean=5.0, sigma=1.5, size=n_fraud).clip(10, 25_000)
    TransactionAmt = np.concatenate([fraud_amt, legit_amt]).astype(np.float32)

    # ── Card features ──
    card1 = rng.integers(1_000, 19_000, size=n).astype(np.float32)
    card2 = rng.choice([111.0, 222.0, 321.0, 410.0, 555.0, np.nan], size=n, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]).astype(np.float32)
    card3 = rng.choice([150.0, 185.0, 187.0, 150.0], size=n, p=[0.4, 0.3, 0.2, 0.1]).astype(np.float32)
    card4_options = ["visa", "mastercard", "american express", "discover"]
    card4 = rng.choice(card4_options, size=n, p=[0.45, 0.35, 0.12, 0.08])
    card5 = rng.choice([102.0, 117.0, 201.0, 224.0, 226.0], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1]).astype(np.float32)
    card6_options = ["debit", "credit", "debit or credit", "charge card"]
    card6 = rng.choice(card6_options, size=n, p=[0.5, 0.35, 0.1, 0.05])

    # ── Address ──
    addr1 = rng.integers(100, 500, size=n).astype(np.float32)
    addr2 = rng.integers(10, 102, size=n).astype(np.float32)
    # Add some NaN
    addr1[rng.random(n) < 0.15] = np.nan
    addr2[rng.random(n) < 0.15] = np.nan

    # ── Email domains ──
    legit_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
    risky_domains = ["anonymous.com", "protonmail.com", "guerrillamail.com", "10minutemail.com"]
    all_domains = legit_domains + risky_domains

    p_legit_email = rng.choice(legit_domains, size=n_legit)
    p_fraud_email = rng.choice(all_domains, size=n_fraud, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.125, 0.125, 0.125, 0.125])
    P_emaildomain = np.concatenate([p_fraud_email, p_legit_email])

    # Legitimate: R matches P 90% of the time; Fraud: only 30% match
    r_legit = np.where(rng.random(n_legit) < 0.9, p_legit_email, rng.choice(legit_domains, size=n_legit))
    r_fraud = np.where(rng.random(n_fraud) < 0.3, p_fraud_email, rng.choice(all_domains, size=n_fraud))
    R_emaildomain = np.concatenate([r_fraud, r_legit])

    # ── C features (count features) ──
    C_cols: dict[str, np.ndarray] = {}
    for i in range(1, 15):
        base = rng.poisson(lam=1.5, size=n).astype(np.float32)
        fraud_boost = np.where(labels == 1, rng.poisson(lam=3, size=n), 0).astype(np.float32)
        C_cols[f"C{i}"] = (base + fraud_boost).astype(np.float32)

    # ── D features (time delta features in days) ──
    D_cols: dict[str, np.ndarray] = {}
    for i in range(1, 16):
        base = rng.exponential(scale=60, size=n).astype(np.float32)
        # Fraud has shorter time deltas (newer accounts / unusual patterns)
        fraud_d = np.where(labels == 1, rng.exponential(scale=5, size=n), base)
        D_cols[f"D{i}"] = fraud_d.astype(np.float32)
    # ~20% NaN for D features
    for key in D_cols:
        mask = rng.random(n) < 0.2
        D_cols[key][mask] = np.nan

    # ── M features (match features) ──
    M_cols: dict[str, np.ndarray] = {}
    for i in range(1, 10):
        # Legit: 85% T, 15% F; Fraud: 50% T, 50% F
        legit_m = rng.choice(["T", "F"], size=n_legit, p=[0.85, 0.15])
        fraud_m = rng.choice(["T", "F"], size=n_fraud, p=[0.5, 0.5])
        M_cols[f"M{i}"] = np.concatenate([fraud_m, legit_m])

    # ── V features (Vesta engineered, V1-V339) ──
    # Generate a reduced set of the most signal-bearing V features
    V_cols: dict[str, np.ndarray] = {}
    for i in range(1, 340):
        if i % 3 == 0:
            # Some V features are near-zero with heavy tails
            base = rng.exponential(scale=0.5, size=n).astype(np.float32)
            fraud_v = np.where(labels == 1, rng.exponential(scale=2.0, size=n), base)
        elif i % 3 == 1:
            # Binary-ish
            base = rng.binomial(1, 0.3, size=n).astype(np.float32)
            fraud_v = np.where(labels == 1, rng.binomial(1, 0.6, size=n), base)
        else:
            # Count-like
            base = rng.poisson(lam=2, size=n).astype(np.float32)
            fraud_v = np.where(labels == 1, rng.poisson(lam=5, size=n), base)
        fraud_v = fraud_v.astype(np.float32)
        # Sprinkle NaN (~30% for V features)
        fraud_v[rng.random(n) < 0.30] = np.nan
        V_cols[f"V{i}"] = fraud_v

    # ── Assemble DataFrame ──
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1, dtype=np.int32),
            "isFraud": labels,
            "TransactionDT": TransactionDT,
            "TransactionAmt": TransactionAmt,
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            "card1": card1,
            "card2": card2,
            "card3": card3,
            "card4": card4,
            "card5": card5,
            "card6": card6,
            "addr1": addr1,
            "addr2": addr2,
            "dist1": rng.exponential(scale=50, size=n).astype(np.float32),
            "dist2": rng.exponential(scale=100, size=n).astype(np.float32),
            "P_emaildomain": P_emaildomain,
            "R_emaildomain": R_emaildomain,
        }
    )

    for col_dict in [C_cols, D_cols, M_cols, V_cols]:
        for col_name, col_data in col_dict.items():
            df[col_name] = col_data

    # Shuffle to mix fraud/legit
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(
        f"Synthetic dataset generated: {len(df):,} rows × {df.shape[1]} columns | "
        f"Fraud rate: {df['isFraud'].mean():.4%}"
    )
    return df


# ─────────────────────────────────────────────
# Unified Data Loader
# ─────────────────────────────────────────────

def load_data(data_dir: str | Path) -> Tuple[pd.DataFrame, str]:
    """
    Unified loader: tries real datasets first, falls back to synthetic.

    Priority order:
    1. IEEE-CIS (train_transaction.csv + train_identity.csv)
    2. Credit Card 2023 (creditcard_2023.csv)
    3. Synthetic data (generated in-memory)

    Args:
        data_dir: Root data directory (expects 'raw/' sub-folder).

    Returns:
        Tuple of (DataFrame, source_name) where source_name describes the origin.
    """
    raw_dir = Path(data_dir) / "raw"

    # ── Attempt 1: IEEE-CIS ──
    ieee_trans = raw_dir / "train_transaction.csv"
    ieee_ident = raw_dir / "train_identity.csv"
    if ieee_trans.exists() and ieee_ident.exists():
        logger.info("Found IEEE-CIS dataset — loading real data")
        try:
            df = load_ieee_cis(ieee_trans, ieee_ident)
            logger.info("Successfully loaded IEEE-CIS dataset")
            return df, "ieee_cis"
        except Exception as exc:
            logger.warning(f"Failed to load IEEE-CIS data: {exc} — trying fallback")

    # ── Attempt 2: Credit Card 2023 ──
    cc_path = raw_dir / "creditcard_2023.csv"
    if cc_path.exists():
        logger.info("Found creditcard_2023.csv — loading real data")
        try:
            df = load_creditcard_2023(cc_path)
            logger.info("Successfully loaded Credit Card 2023 dataset")
            return df, "creditcard_2023"
        except Exception as exc:
            logger.warning(f"Failed to load Credit Card 2023 data: {exc} — falling back to synthetic")

    # ── Attempt 3: Synthetic ──
    logger.warning(
        "No real datasets found in %s. Generating 500k synthetic records. "
        "Run setup/download_data.py to obtain real data.",
        raw_dir,
    )
    df = generate_synthetic_fraud_data(n=500_000, fraud_rate=0.035)
    return df, "synthetic"

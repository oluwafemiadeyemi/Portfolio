"""
data_loader.py
==============
Parkinson's Disease Detection — data loading and synthetic generation.

Supports:
  - UCI Parkinson's (195 recordings)
  - UCI Parkinson's Telemonitoring (5,875 recordings)
  - Synthetic multi-modal dataset (voice + gait + tremor)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UCI Parkinson's (classic 195-recording dataset)
# ---------------------------------------------------------------------------

UCI_COLUMN_MAP: Dict[str, str] = {
    "Fo_Hz": "MDVP_Fo",
    "Fhi_Hz": "MDVP_Fhi",
    "Flo_Hz": "MDVP_Flo",
    "Jitter_pct": "MDVP_Jitter_pct",
    "Jitter_Abs": "MDVP_Jitter_Abs",
    "RAP": "MDVP_RAP",
    "PPQ": "MDVP_PPQ",
    "Jitter_DDP": "MDVP_Jitter_DDP",
    "Shimmer": "MDVP_Shimmer",
    "Shimmer_dB": "MDVP_Shimmer_dB",
    "Shimmer_APQ3": "MDVP_Shimmer_APQ3",
    "Shimmer_APQ5": "MDVP_Shimmer_APQ5",
    "APQ": "MDVP_APQ",
    "Shimmer_DDA": "MDVP_Shimmer_DDA",
    "NHR": "NHR",
    "HNR": "HNR",
    "status": "status",
    "RPDE": "RPDE",
    "DFA": "DFA",
    "spread1": "spread1",
    "spread2": "spread2",
    "D2": "D2",
    "PPE": "PPE",
}


def load_uci_parkinsons(filepath: str | Path) -> pd.DataFrame:
    """Load the original 195-recording UCI Parkinson's dataset.

    Parameters
    ----------
    filepath:
        Path to ``parkinsons.csv`` (or ``parkinsons_data.csv``).

    Returns
    -------
    pd.DataFrame with standardised column names and a derived ``subject_id``.
    """
    filepath = Path(filepath)
    logger.info("Loading UCI Parkinson's dataset from %s", filepath)

    df = pd.read_csv(filepath)

    # The first column is the recording name, e.g. "phon_R01_S01_1"
    name_col = df.columns[0]
    df = df.rename(columns={name_col: "recording_name"})

    # Rename columns to standardised names
    df = df.rename(columns=UCI_COLUMN_MAP)

    # Derive subject_id from recording name  (e.g. S01 → "S01")
    if "recording_name" in df.columns:
        df["subject_id"] = (
            df["recording_name"]
            .str.extract(r"(S\d+)", expand=False)
            .fillna("unknown")
        )

    # Ensure numeric
    numeric_cols = [c for c in df.columns if c not in ("recording_name", "subject_id")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    logger.info("Loaded %d recordings, %d subjects", len(df), df["subject_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# UCI Parkinson's Telemonitoring dataset
# ---------------------------------------------------------------------------

TELE_COLUMN_MAP: Dict[str, str] = {
    "subject#": "subject_id",
    "age": "age",
    "sex": "sex",
    "test_time": "test_time",
    "motor_UPDRS": "motor_UPDRS",
    "total_UPDRS": "total_UPDRS",
    "Jitter(%)": "MDVP_Jitter_pct",
    "Jitter(Abs)": "MDVP_Jitter_Abs",
    "Jitter:RAP": "MDVP_RAP",
    "Jitter:PPQ5": "MDVP_PPQ",
    "Jitter:DDP": "MDVP_Jitter_DDP",
    "Shimmer": "MDVP_Shimmer",
    "Shimmer(dB)": "MDVP_Shimmer_dB",
    "Shimmer:APQ3": "MDVP_Shimmer_APQ3",
    "Shimmer:APQ5": "MDVP_Shimmer_APQ5",
    "Shimmer:APQ11": "MDVP_APQ",
    "Shimmer:DDA": "MDVP_Shimmer_DDA",
    "NHR": "NHR",
    "HNR": "HNR",
    "RPDE": "RPDE",
    "DFA": "DFA",
    "PPE": "PPE",
}


def load_uci_telemonitoring(filepath: str | Path) -> pd.DataFrame:
    """Load the 5,875-recording UCI Parkinson's Telemonitoring dataset.

    Parameters
    ----------
    filepath:
        Path to ``parkinsons_updrs.data`` or ``parkinsons_telemonitoring.csv``.
    """
    filepath = Path(filepath)
    logger.info("Loading UCI Telemonitoring dataset from %s", filepath)

    # The file may be comma or whitespace-separated
    try:
        df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath, sep=r"\s+")

    df = df.rename(columns=TELE_COLUMN_MAP)

    # Ensure subject_id is a string
    if "subject_id" in df.columns:
        df["subject_id"] = "S" + df["subject_id"].astype(str).str.zfill(3)

    # Derive binary status: all recordings in this dataset are PD patients
    df["status"] = 1

    numeric_cols = [c for c in df.columns if c != "subject_id"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    logger.info("Loaded %d recordings from %d subjects", len(df), df["subject_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Synthetic multi-modal dataset generator
# ---------------------------------------------------------------------------

RNG_SEED = 42


def _sample_clipped(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    n: int,
) -> np.ndarray:
    """Sample from a truncated normal distribution."""
    raw = rng.normal(mean, std, n * 4)
    raw = raw[(raw >= low) & (raw <= high)][:n]
    if len(raw) < n:
        # Fallback: uniform fill
        extra = rng.uniform(low, high, n - len(raw))
        raw = np.concatenate([raw, extra])
    return raw[:n]


def generate_synthetic_multimodal(
    n_subjects: int = 1000,
    recordings_per_subject: int = 6,
    pd_prevalence: float = 0.75,
    random_state: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate a realistic synthetic multi-modal Parkinson's dataset.

    Preserves within-subject correlation: each subject has a stable
    "true" biomarker profile, and individual recordings deviate slightly.

    Parameters
    ----------
    n_subjects:
        Total number of unique subjects.
    recordings_per_subject:
        Number of recordings per subject.
    pd_prevalence:
        Fraction of subjects with Parkinson's Disease.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with one row per recording (n_subjects × recordings_per_subject rows).
    """
    rng = np.random.default_rng(random_state)
    n_pd = int(n_subjects * pd_prevalence)
    n_healthy = n_subjects - n_pd

    records = []

    for group, n_group, is_pd in [("PD", n_pd, True), ("HC", n_healthy, False)]:
        for subj_idx in range(n_group):
            subject_id = f"{group}_{subj_idx + 1:04d}"

            # ---- Demographics ----
            age_base = rng.uniform(55, 80) if is_pd else rng.uniform(50, 75)
            sex = rng.choice([0, 1])  # 0=female, 1=male
            years_dx = float(rng.uniform(0, 15)) if is_pd else 0.0

            # ---- Subject-level "true" voice biomarker profile ----
            if is_pd:
                fo_true = rng.uniform(150, 180)
                jitter_true = rng.uniform(0.003, 0.02)
                shimmer_true = rng.uniform(0.02, 0.10)
                hnr_true = rng.uniform(15, 22)
            else:
                fo_true = rng.uniform(165, 220)
                jitter_true = rng.uniform(0.001, 0.006)
                shimmer_true = rng.uniform(0.01, 0.04)
                hnr_true = rng.uniform(20, 28)

            rpde_true = rng.uniform(0.3, 0.7) if is_pd else rng.uniform(0.2, 0.55)
            dfa_true = rng.uniform(0.65, 0.90) if is_pd else rng.uniform(0.55, 0.78)
            spread1_true = rng.uniform(-6.0, -4.0) if is_pd else rng.uniform(-7.5, -5.0)
            spread2_true = rng.uniform(0.15, 0.45) if is_pd else rng.uniform(0.05, 0.25)
            d2_true = rng.uniform(2.0, 3.5) if is_pd else rng.uniform(1.5, 2.8)
            ppe_true = rng.uniform(0.20, 0.55) if is_pd else rng.uniform(0.05, 0.22)

            # ---- Subject-level "true" gait profile ----
            if is_pd:
                stride_true = rng.uniform(0.8, 1.2)
                cadence_true = rng.uniform(100, 120)
                stride_reg_true = rng.uniform(0.85, 0.95)
                gait_asym_true = rng.uniform(0.05, 0.20)
            else:
                stride_true = rng.uniform(1.1, 1.6)
                cadence_true = rng.uniform(95, 115)
                stride_reg_true = rng.uniform(0.92, 0.99)
                gait_asym_true = rng.uniform(0.01, 0.08)

            # ---- Subject-level "true" tremor profile ----
            if is_pd:
                rest_tremor_true = rng.uniform(0.5, 3.0)
                tremor_freq_true = rng.uniform(3, 6)
                tremor_power_true = rng.uniform(0.4, 0.9)
            else:
                rest_tremor_true = rng.uniform(0.0, 0.5)
                tremor_freq_true = rng.uniform(6, 12)
                tremor_power_true = rng.uniform(0.0, 0.25)

            # ---- UPDRS ----
            if is_pd:
                updrs_base = 15 + years_dx * 2.5 + rng.uniform(-5, 5)
                updrs_base = float(np.clip(updrs_base, 15, 60))
            else:
                updrs_base = float(rng.uniform(0, 10))

            # ---- Generate recordings for this subject ----
            for rec_idx in range(recordings_per_subject):
                noise = 0.05  # ±5% within-subject noise

                row: Dict[str, object] = {
                    "subject_id": subject_id,
                    "recording_id": f"{subject_id}_R{rec_idx + 1:02d}",
                    "recording_index": rec_idx,
                    "status": int(is_pd),
                    "age": float(np.clip(age_base + rng.uniform(-1, 1), 40, 90)),
                    "sex": sex,
                    "years_since_diagnosis": years_dx,
                    # Voice
                    "MDVP_Fo": float(np.clip(fo_true * rng.uniform(1 - noise, 1 + noise), 80, 260)),
                    "MDVP_Jitter_pct": float(np.clip(jitter_true * rng.uniform(1 - noise, 1 + noise), 0.0001, 0.05)),
                    "MDVP_Shimmer": float(np.clip(shimmer_true * rng.uniform(1 - noise, 1 + noise), 0.005, 0.20)),
                    "HNR": float(np.clip(hnr_true + rng.normal(0, 0.5), 5, 35)),
                    "RPDE": float(np.clip(rpde_true + rng.normal(0, 0.02), 0.1, 0.99)),
                    "DFA": float(np.clip(dfa_true + rng.normal(0, 0.01), 0.3, 1.0)),
                    "spread1": float(np.clip(spread1_true + rng.normal(0, 0.15), -9, -2)),
                    "spread2": float(np.clip(spread2_true + rng.normal(0, 0.02), 0.0, 0.6)),
                    "D2": float(np.clip(d2_true + rng.normal(0, 0.1), 1.0, 5.0)),
                    "PPE": float(np.clip(ppe_true + rng.normal(0, 0.02), 0.0, 0.9)),
                    # Gait
                    "stride_length": float(np.clip(stride_true * rng.uniform(1 - noise, 1 + noise), 0.4, 2.0)),
                    "cadence": float(np.clip(cadence_true + rng.normal(0, 1.5), 60, 150)),
                    "stride_regularity": float(np.clip(stride_reg_true + rng.normal(0, 0.01), 0.5, 1.0)),
                    "gait_asymmetry": float(np.clip(gait_asym_true * rng.uniform(1 - noise, 1 + noise), 0.0, 0.40)),
                    # Tremor
                    "rest_tremor_amplitude": float(np.clip(rest_tremor_true * rng.uniform(1 - noise, 1 + noise), 0.0, 5.0)),
                    "tremor_frequency": float(np.clip(tremor_freq_true + rng.normal(0, 0.3), 0.5, 20)),
                    "tremor_power_ratio": float(np.clip(tremor_power_true + rng.normal(0, 0.03), 0.0, 1.0)),
                    # UPDRS
                    "UPDRS_total": float(np.clip(updrs_base + rng.normal(0, 2), 0, 176)),
                }
                records.append(row)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info(
        "Generated synthetic dataset: %d recordings, %d subjects (%d PD / %d HC)",
        len(df),
        df["subject_id"].nunique(),
        (df["status"] == 1).sum() // recordings_per_subject,
        (df["status"] == 0).sum() // recordings_per_subject,
    )
    return df


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Try to load real data files; fall back to synthetic generation.

    Search order:
      1. ``data_dir/parkinsons.csv``
      2. ``data_dir/parkinsons_data.csv``
      3. ``data_dir/raw/parkinsons.csv``
      4. Synthetic generation
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "parkinsons.csv",
        data_dir / "parkinsons_data.csv",
        data_dir / "raw" / "parkinsons.csv",
        data_dir / "raw" / "parkinsons_data.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Found data file: %s", candidate)
            try:
                df = load_uci_parkinsons(candidate)
                # UCI dataset doesn't have gait/tremor; generate synthetic augmentation
                logger.info("UCI dataset loaded; appending synthetic multi-modal features.")
                return _augment_uci_with_synthetic(df)
            except Exception as exc:
                logger.warning("Failed to load %s: %s — falling back to synthetic.", candidate, exc)

    logger.info("No real data found in %s. Generating synthetic dataset.", data_dir)
    return generate_synthetic_multimodal()


def _augment_uci_with_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic gait and tremor columns to a UCI voice-only dataset."""
    rng = np.random.default_rng(0)
    n = len(df)
    status = df["status"].values

    df = df.copy()

    # Gait
    df["stride_length"] = np.where(
        status == 1,
        rng.uniform(0.8, 1.2, n),
        rng.uniform(1.1, 1.6, n),
    )
    df["cadence"] = np.where(
        status == 1,
        rng.uniform(100, 120, n),
        rng.uniform(95, 115, n),
    )
    df["stride_regularity"] = np.where(
        status == 1,
        rng.uniform(0.85, 0.95, n),
        rng.uniform(0.92, 0.99, n),
    )
    df["gait_asymmetry"] = np.where(
        status == 1,
        rng.uniform(0.05, 0.20, n),
        rng.uniform(0.01, 0.08, n),
    )

    # Tremor
    df["rest_tremor_amplitude"] = np.where(
        status == 1,
        rng.uniform(0.5, 3.0, n),
        rng.uniform(0.0, 0.5, n),
    )
    df["tremor_frequency"] = np.where(
        status == 1,
        rng.uniform(3, 6, n),
        rng.uniform(6, 12, n),
    )
    df["tremor_power_ratio"] = np.where(
        status == 1,
        rng.uniform(0.4, 0.9, n),
        rng.uniform(0.0, 0.25, n),
    )

    # Demographics
    if "age" not in df.columns:
        df["age"] = np.where(status == 1, rng.uniform(55, 80, n), rng.uniform(50, 75, n))
    if "sex" not in df.columns:
        df["sex"] = rng.integers(0, 2, n)
    df["years_since_diagnosis"] = np.where(status == 1, rng.uniform(0, 15, n), 0.0)

    # UPDRS
    df["UPDRS_total"] = np.where(
        status == 1,
        np.clip(rng.uniform(15, 60, n), 0, 176),
        rng.uniform(0, 10, n),
    )

    # subject_id / recording_id if absent
    if "subject_id" not in df.columns:
        df["subject_id"] = "S" + pd.Series(range(1, n + 1)).astype(str).str.zfill(3)
    if "recording_id" not in df.columns:
        df["recording_id"] = df["subject_id"] + "_R01"
    if "recording_index" not in df.columns:
        df["recording_index"] = 0

    return df


# ---------------------------------------------------------------------------
# Subject-level train/test split
# ---------------------------------------------------------------------------

def create_subject_splits(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    test_size: float = 0.2,
    random_state: int = RNG_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data at the **subject** level to prevent data leakage.

    All recordings from a given subject appear exclusively in train or test.

    Parameters
    ----------
    df:
        Input DataFrame with a subject identifier column.
    subject_col:
        Column name containing the subject identifier.
    test_size:
        Fraction of subjects to allocate to the test set.
    random_state:
        Random seed.

    Returns
    -------
    Tuple[train_df, test_df]
    """
    if subject_col not in df.columns:
        raise ValueError(f"Column '{subject_col}' not found in DataFrame.")

    subjects = df[subject_col].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_size))
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    train_df = df[df[subject_col].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df[subject_col].isin(test_subjects)].reset_index(drop=True)

    logger.info(
        "Subject split — train: %d recordings (%d subjects) | test: %d recordings (%d subjects)",
        len(train_df),
        train_df[subject_col].nunique(),
        len(test_df),
        test_df[subject_col].nunique(),
    )
    return train_df, test_df

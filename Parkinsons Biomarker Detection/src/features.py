"""
features.py
===========
Feature engineering for Parkinson's Disease Detection.

Three modalities:
  - Voice  (acoustic biomarkers)
  - Gait   (stride / cadence / regularity)
  - Tremor (accelerometer-derived)

Plus longitudinal aggregation across multiple recordings per subject.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice features
# ---------------------------------------------------------------------------

def extract_voice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and engineer voice biomarker features.

    Engineered features
    -------------------
    jitter_shimmer_ratio  : MDVP_Jitter_pct / (MDVP_Shimmer + eps) — captures
                            relative instability in pitch vs. amplitude
    noise_ratio           : 1 / (HNR + 1) — normalised noise-to-harmonics proxy
    vocal_stability_index : compound of RPDE, DFA, spread1 capturing non-linear
                            vocal dynamics (higher = less stable)
    """
    df = df.copy()
    eps = 1e-9

    # Raw voice columns that must be present
    voice_raw = ["MDVP_Fo", "MDVP_Jitter_pct", "MDVP_Shimmer", "HNR", "RPDE", "DFA",
                 "spread1", "spread2", "D2", "PPE"]
    missing = [c for c in voice_raw if c not in df.columns]
    if missing:
        logger.warning("Missing voice columns: %s — filling with column medians.", missing)
        for col in missing:
            df[col] = np.nan

    # Fill NaN with column medians
    for col in voice_raw:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Engineered
    df["jitter_shimmer_ratio"] = df["MDVP_Jitter_pct"] / (df["MDVP_Shimmer"] + eps)
    df["noise_ratio"] = 1.0 / (df["HNR"].clip(lower=0) + 1.0)

    # Vocal stability index: z-normalise each component then combine
    rpde_z = (df["RPDE"] - df["RPDE"].mean()) / (df["RPDE"].std() + eps)
    dfa_z = (df["DFA"] - df["DFA"].mean()) / (df["DFA"].std() + eps)
    # spread1 is negative in UCI data; higher absolute value = less stable
    spread1_z = (-df["spread1"] - (-df["spread1"]).mean()) / ((-df["spread1"]).std() + eps)
    df["vocal_stability_index"] = (rpde_z + dfa_z + spread1_z) / 3.0

    voice_feature_cols = voice_raw + ["jitter_shimmer_ratio", "noise_ratio", "vocal_stability_index"]
    return df[voice_feature_cols]


# ---------------------------------------------------------------------------
# Gait features
# ---------------------------------------------------------------------------

def extract_gait_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and engineer gait biomarker features.

    Engineered features
    -------------------
    stride_regularity_normalized : stride_regularity rescaled [0, 1]
    gait_variability_index       : compound measure of asymmetry + cadence spread
    bilateral_asymmetry_score    : sigmoid-scaled gait_asymmetry
    """
    df = df.copy()
    eps = 1e-9

    gait_raw = ["stride_length", "cadence", "stride_regularity", "gait_asymmetry"]
    missing = [c for c in gait_raw if c not in df.columns]
    if missing:
        logger.warning("Missing gait columns: %s — filling with medians.", missing)
        for col in missing:
            df[col] = np.nan

    for col in gait_raw:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Normalised stride regularity
    sr = df["stride_regularity"]
    df["stride_regularity_normalized"] = (sr - sr.min()) / (sr.max() - sr.min() + eps)

    # Gait variability index: low stride regularity + high asymmetry = high variability
    df["gait_variability_index"] = (
        (1.0 - df["stride_regularity"]) * 0.5
        + df["gait_asymmetry"] * 0.5
    )

    # Bilateral asymmetry score — sigmoid
    df["bilateral_asymmetry_score"] = 1.0 / (1.0 + np.exp(-10 * (df["gait_asymmetry"] - 0.10)))

    gait_feature_cols = gait_raw + [
        "stride_regularity_normalized",
        "gait_variability_index",
        "bilateral_asymmetry_score",
    ]
    return df[gait_feature_cols]


# ---------------------------------------------------------------------------
# Tremor features
# ---------------------------------------------------------------------------

def extract_tremor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and engineer tremor biomarker features.

    Engineered features
    -------------------
    tremor_severity_score : rest_tremor_amplitude × tremor_power_ratio
    frequency_band_power  : fraction of power in the PD-relevant 3-6 Hz band
                            (approximated from tremor_frequency and tremor_power_ratio)
    tremor_consistency    : inverse coefficient of variation (higher = more consistent tremor)
    """
    df = df.copy()
    eps = 1e-9

    tremor_raw = ["rest_tremor_amplitude", "tremor_frequency", "tremor_power_ratio"]
    missing = [c for c in tremor_raw if c not in df.columns]
    if missing:
        logger.warning("Missing tremor columns: %s — filling with medians.", missing)
        for col in missing:
            df[col] = np.nan

    for col in tremor_raw:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Tremor severity
    df["tremor_severity_score"] = df["rest_tremor_amplitude"] * df["tremor_power_ratio"]

    # Frequency band power: 3-6 Hz is the PD rest-tremor band.
    # Approximate: if tremor_frequency is in [3,6] → full power; else attenuated.
    in_band = ((df["tremor_frequency"] >= 3) & (df["tremor_frequency"] <= 6)).astype(float)
    df["frequency_band_power"] = df["tremor_power_ratio"] * in_band

    # Tremor consistency: ratio of amplitude to frequency (lower frequency + higher amplitude = PD)
    df["tremor_consistency"] = df["rest_tremor_amplitude"] / (df["tremor_frequency"] + eps)

    tremor_feature_cols = tremor_raw + [
        "tremor_severity_score",
        "frequency_band_power",
        "tremor_consistency",
    ]
    return df[tremor_feature_cols]


# ---------------------------------------------------------------------------
# Longitudinal features
# ---------------------------------------------------------------------------

def extract_longitudinal_features(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    feature_cols: Optional[List[str]] = None,
    time_col: str = "recording_index",
) -> pd.DataFrame:
    """Compute within-subject longitudinal statistics.

    For each subject and each feature column computes:
      - ``{feat}_slope``     : linear regression slope over recordings
      - ``{feat}_cv``        : coefficient of variation across recordings
      - ``{feat}_baseline_dev`` : deviation from the subject's first recording

    Returns a DataFrame with one row per subject and the longitudinal features.
    """
    if subject_col not in df.columns:
        raise ValueError(f"Column '{subject_col}' not found.")

    if feature_cols is None:
        # Use all numeric columns except identifiers and target
        exclude = {subject_col, "recording_id", "status", "UPDRS_total",
                   "age", "sex", "years_since_diagnosis", "recording_index"}
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    if time_col not in df.columns:
        df = df.copy()
        df[time_col] = df.groupby(subject_col).cumcount()

    rows = []
    for subj_id, grp in df.groupby(subject_col):
        grp = grp.sort_values(time_col)
        t = grp[time_col].values.astype(float)
        row: dict = {"subject_id": subj_id}

        # Subject-level metadata
        if "status" in grp.columns:
            row["status"] = grp["status"].iloc[0]
        if "UPDRS_total" in grp.columns:
            row["UPDRS_total_mean"] = float(grp["UPDRS_total"].mean())
        if "age" in grp.columns:
            row["age"] = float(grp["age"].iloc[0])
        if "sex" in grp.columns:
            row["sex"] = int(grp["sex"].iloc[0])
        if "years_since_diagnosis" in grp.columns:
            row["years_since_diagnosis"] = float(grp["years_since_diagnosis"].iloc[0])

        for feat in feature_cols:
            if feat not in grp.columns:
                continue
            vals = grp[feat].values.astype(float)
            if np.isnan(vals).all():
                row[f"{feat}_slope"] = np.nan
                row[f"{feat}_cv"] = np.nan
                row[f"{feat}_baseline_dev"] = np.nan
                continue

            # Slope
            if len(vals) >= 2 and len(t) >= 2:
                try:
                    slope, _, _, _, _ = linregress(t, vals)
                    row[f"{feat}_slope"] = float(slope)
                except Exception:
                    row[f"{feat}_slope"] = 0.0
            else:
                row[f"{feat}_slope"] = 0.0

            # CV
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            row[f"{feat}_cv"] = float(std_val / (abs(mean_val) + 1e-9))

            # Baseline deviation
            baseline = vals[0] if not np.isnan(vals[0]) else np.nanmean(vals)
            row[f"{feat}_baseline_dev"] = float(np.nanmean(vals) - baseline)

        rows.append(row)

    result_df = pd.DataFrame(rows)
    logger.info(
        "Longitudinal features computed: %d subjects, %d features",
        len(result_df),
        result_df.shape[1],
    )
    return result_df


# ---------------------------------------------------------------------------
# Multi-modal feature builder
# ---------------------------------------------------------------------------

def build_multimodal_features(
    df: pd.DataFrame,
    modalities: List[str] = None,
) -> pd.DataFrame:
    """Assemble feature matrix from requested modalities.

    Handles missing modalities gracefully — if gait columns are absent,
    the gait modality is skipped with a warning.

    Parameters
    ----------
    df:
        Source DataFrame (one row per recording).
    modalities:
        List from ``['voice', 'gait', 'tremor']``. Defaults to all three.

    Returns
    -------
    pd.DataFrame with selected and engineered features (no target column).
    """
    if modalities is None:
        modalities = ["voice", "gait", "tremor"]

    parts: List[pd.DataFrame] = []

    if "voice" in modalities:
        voice_cols_needed = ["MDVP_Fo", "MDVP_Jitter_pct", "MDVP_Shimmer", "HNR"]
        if all(c in df.columns for c in voice_cols_needed):
            try:
                parts.append(extract_voice_features(df))
                logger.info("Voice features extracted.")
            except Exception as exc:
                logger.warning("Voice feature extraction failed: %s", exc)
        else:
            logger.warning("Voice columns absent — skipping voice modality.")

    if "gait" in modalities:
        gait_cols_needed = ["stride_length", "cadence"]
        if all(c in df.columns for c in gait_cols_needed):
            try:
                parts.append(extract_gait_features(df))
                logger.info("Gait features extracted.")
            except Exception as exc:
                logger.warning("Gait feature extraction failed: %s", exc)
        else:
            logger.warning("Gait columns absent — skipping gait modality.")

    if "tremor" in modalities:
        tremor_cols_needed = ["rest_tremor_amplitude", "tremor_frequency"]
        if all(c in df.columns for c in tremor_cols_needed):
            try:
                parts.append(extract_tremor_features(df))
                logger.info("Tremor features extracted.")
            except Exception as exc:
                logger.warning("Tremor feature extraction failed: %s", exc)
        else:
            logger.warning("Tremor columns absent — skipping tremor modality.")

    if not parts:
        raise ValueError("No modality features could be extracted. Check input DataFrame.")

    feature_df = pd.concat(parts, axis=1)

    # Drop duplicate columns if any (e.g., overlapping raw columns)
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

    return feature_df

"""
longitudinal.py
===============
Longitudinal analysis for Parkinson's Disease detection platform.

Functions:
  - Progression trajectory computation per subject
  - Progression pattern clustering (fast/slow/stable progressors)
  - Future UPDRS extrapolation
  - Treatment response detection
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progression trajectories
# ---------------------------------------------------------------------------

def compute_progression_trajectory(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    time_col: str = "recording_index",
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute a linear regression slope per feature per subject.

    A positive slope indicates worsening over time for features where higher
    values correlate with more severe PD (e.g., tremor amplitude, jitter).
    A negative slope on stride_length or HNR indicates worsening.

    Parameters
    ----------
    df: Longitudinal dataset with multiple recordings per subject.
    subject_col: Column containing subject IDs.
    time_col: Column denoting recording time/order.
    feature_cols: Features to compute trajectories for. Defaults to all numerics.

    Returns
    -------
    pd.DataFrame with one row per subject, columns: subject_id, {feat}_slope,
    plus metadata (status, age, UPDRS_baseline, UPDRS_last, UPDRS_change).
    """
    if subject_col not in df.columns:
        raise ValueError(f"Column '{subject_col}' not found.")

    if time_col not in df.columns:
        df = df.copy()
        df[time_col] = df.groupby(subject_col).cumcount()

    if feature_cols is None:
        exclude = {subject_col, "recording_id", "status", "UPDRS_total",
                   "age", "sex", "years_since_diagnosis", "recording_index", time_col}
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    rows: List[Dict[str, Any]] = []
    for subj_id, grp in df.groupby(subject_col):
        grp = grp.sort_values(time_col)
        t = grp[time_col].values.astype(float)

        row: Dict[str, Any] = {subject_col: subj_id}

        # Metadata
        if "status" in grp.columns:
            row["status"] = int(grp["status"].iloc[0])
        if "age" in grp.columns:
            row["age"] = float(grp["age"].iloc[0])
        if "sex" in grp.columns:
            row["sex"] = int(grp["sex"].iloc[0])
        if "years_since_diagnosis" in grp.columns:
            row["years_since_diagnosis"] = float(grp["years_since_diagnosis"].iloc[0])

        if "UPDRS_total" in grp.columns:
            updrs_vals = grp["UPDRS_total"].dropna().values
            row["UPDRS_baseline"] = float(updrs_vals[0]) if len(updrs_vals) > 0 else np.nan
            row["UPDRS_last"] = float(updrs_vals[-1]) if len(updrs_vals) > 0 else np.nan
            row["UPDRS_change"] = float(updrs_vals[-1] - updrs_vals[0]) if len(updrs_vals) > 1 else 0.0
            if len(t) >= 2 and len(updrs_vals) >= 2:
                slope, intercept, r, p, _ = stats.linregress(t[:len(updrs_vals)], updrs_vals)
                row["UPDRS_slope"] = float(slope)
                row["UPDRS_r2"] = float(r ** 2)
                row["UPDRS_trend_pvalue"] = float(p)
            else:
                row["UPDRS_slope"] = 0.0
                row["UPDRS_r2"] = 0.0
                row["UPDRS_trend_pvalue"] = 1.0

        # Feature slopes
        for feat in feature_cols:
            if feat not in grp.columns:
                row[f"{feat}_slope"] = np.nan
                continue
            vals = grp[feat].dropna().values.astype(float)
            t_sub = t[:len(vals)]
            if len(vals) >= 2 and len(t_sub) >= 2:
                try:
                    slope, _, r_val, p_val, _ = stats.linregress(t_sub, vals)
                    row[f"{feat}_slope"] = float(slope)
                    row[f"{feat}_pvalue"] = float(p_val)
                except Exception:
                    row[f"{feat}_slope"] = 0.0
                    row[f"{feat}_pvalue"] = 1.0
            else:
                row[f"{feat}_slope"] = 0.0
                row[f"{feat}_pvalue"] = 1.0

        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info("Computed trajectories for %d subjects.", len(result))
    return result


# ---------------------------------------------------------------------------
# Progression clustering
# ---------------------------------------------------------------------------

def cluster_progression_patterns(
    trajectories_df: pd.DataFrame,
    n_clusters: int = 3,
    slope_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster subjects into progression patterns using K-Means on slope features.

    Cluster labels are mapped to human-readable progression types:
      - Cluster with highest mean UPDRS slope → "Fast Progressor"
      - Cluster with lowest mean UPDRS slope  → "Stable"
      - Middle cluster(s)                      → "Slow Progressor"

    Parameters
    ----------
    trajectories_df: Output of ``compute_progression_trajectory``.
    n_clusters: Number of K-Means clusters (default 3).
    slope_cols: Slope columns to use for clustering. Defaults to all ``*_slope`` columns.

    Returns
    -------
    trajectories_df with added columns: ``progression_cluster``, ``progression_label``
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = trajectories_df.copy()

    if slope_cols is None:
        slope_cols = [c for c in df.columns if c.endswith("_slope")]

    if not slope_cols:
        logger.warning("No slope columns found — assigning all subjects to 'Unknown' cluster.")
        df["progression_cluster"] = 0
        df["progression_label"] = "Unknown"
        return df

    X_slopes = df[slope_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_slopes)

    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["progression_cluster"] = cluster_labels

    # Map clusters to descriptive labels using UPDRS_slope if available
    if "UPDRS_slope" in df.columns:
        cluster_mean_updrs = (
            df.groupby("progression_cluster")["UPDRS_slope"].mean().sort_values()
        )
        sorted_clusters = cluster_mean_updrs.index.tolist()
        label_map: Dict[int, str] = {}
        labels = ["Stable", "Slow Progressor", "Fast Progressor"]
        # Distribute labels across however many clusters exist
        if n_clusters == 1:
            label_map[sorted_clusters[0]] = "Stable"
        elif n_clusters == 2:
            label_map[sorted_clusters[0]] = "Stable"
            label_map[sorted_clusters[1]] = "Fast Progressor"
        else:
            for i, cl in enumerate(sorted_clusters):
                idx = min(i, len(labels) - 1)
                label_map[cl] = labels[idx]
        df["progression_label"] = df["progression_cluster"].map(label_map).fillna("Unknown")
    else:
        df["progression_label"] = "Cluster_" + df["progression_cluster"].astype(str)

    counts = df["progression_label"].value_counts().to_dict()
    logger.info("Progression clusters: %s", counts)
    return df


# ---------------------------------------------------------------------------
# UPDRS extrapolation
# ---------------------------------------------------------------------------

def predict_future_updrs(
    model: Any,
    current_features: pd.DataFrame,
    updrs_slope: float,
    current_updrs: float,
    months_ahead: int = 6,
) -> Dict[str, float]:
    """Extrapolate UPDRS trajectory into the future.

    Uses the UPDRS regression slope estimated from past recordings to
    project the score forward. The model prediction anchors the current
    estimate and the slope provides the trajectory.

    Parameters
    ----------
    model: Fitted UPDRS regressor.
    current_features: Current-visit feature vector (single row).
    updrs_slope: Estimated slope (UPDRS units per recording interval).
    current_updrs: Current UPDRS score.
    months_ahead: Number of months to project ahead.

    Returns
    -------
    Dict with: predicted_updrs, lower_bound, upper_bound, months_ahead,
               clinical_category
    """
    X = current_features.copy().fillna(0)
    try:
        model_pred = float(model.predict(X)[0])
    except Exception:
        model_pred = current_updrs

    # Blend model prediction with linear extrapolation
    # Assume roughly one recording per month
    extrapolated = current_updrs + updrs_slope * months_ahead
    blended = 0.6 * extrapolated + 0.4 * model_pred
    predicted = float(np.clip(blended, 0, 176))

    # Uncertainty: ±15% of predicted score
    uncertainty = predicted * 0.15
    lower = float(max(0, predicted - uncertainty))
    upper = float(min(176, predicted + uncertainty))

    # Clinical category
    if predicted < 20:
        category = "Mild"
    elif predicted < 40:
        category = "Moderate"
    elif predicted < 80:
        category = "Moderate-Severe"
    else:
        category = "Severe"

    result = {
        "predicted_updrs": round(predicted, 1),
        "lower_bound": round(lower, 1),
        "upper_bound": round(upper, 1),
        "months_ahead": months_ahead,
        "clinical_category": category,
        "updrs_change_from_current": round(predicted - current_updrs, 1),
    }
    logger.info("Future UPDRS prediction: %s", result)
    return result


# ---------------------------------------------------------------------------
# Treatment response detection
# ---------------------------------------------------------------------------

def detect_treatment_response(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    subject_col: str = "subject_id",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Detect significant biomarker changes in response to treatment.

    Uses a paired t-test per biomarker across matched subjects.

    Parameters
    ----------
    before_df: Pre-treatment recordings (one per subject — use mean if multiple).
    after_df: Post-treatment recordings (one per subject).
    subject_col: Column with subject IDs.
    alpha: Significance threshold.

    Returns
    -------
    pd.DataFrame with columns: biomarker, mean_before, mean_after,
    mean_change, pct_change, t_statistic, p_value, significant, responder_direction
    """
    # Compute per-subject means if multiple recordings per subject
    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if subject_col in df.columns:
            return df.groupby(subject_col)[numeric_cols].mean().reset_index()
        return df[numeric_cols]

    before_agg = aggregate(before_df)
    after_agg = aggregate(after_df)

    # Align by subject
    if subject_col in before_agg.columns and subject_col in after_agg.columns:
        merged = before_agg.merge(
            after_agg, on=subject_col, suffixes=("_before", "_after")
        )
    else:
        # If no subject column, treat as matched order
        merged = pd.DataFrame()
        for col in before_agg.select_dtypes(include=[np.number]).columns:
            merged[f"{col}_before"] = before_agg[col].values[:len(after_agg)]
            merged[f"{col}_after"] = after_agg[col].values[:len(before_agg)]

    numeric_features = [
        c.replace("_before", "") for c in merged.columns if c.endswith("_before")
    ]

    results: List[Dict[str, Any]] = []
    for feat in numeric_features:
        before_col = f"{feat}_before"
        after_col = f"{feat}_after"
        if before_col not in merged.columns or after_col not in merged.columns:
            continue

        b_vals = merged[before_col].dropna().values
        a_vals = merged[after_col].dropna().values

        min_len = min(len(b_vals), len(a_vals))
        if min_len < 3:
            continue

        b_vals = b_vals[:min_len]
        a_vals = a_vals[:min_len]

        t_stat, p_val = ttest_rel(a_vals, b_vals)
        mean_before = float(np.mean(b_vals))
        mean_after = float(np.mean(a_vals))
        mean_change = mean_after - mean_before
        pct_change = (mean_change / (abs(mean_before) + 1e-9)) * 100

        results.append({
            "biomarker": feat,
            "mean_before": round(mean_before, 4),
            "mean_after": round(mean_after, 4),
            "mean_change": round(mean_change, 4),
            "pct_change": round(pct_change, 2),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < alpha),
            "responder_direction": "improved" if mean_change < 0 else "worsened",
        })

    result_df = pd.DataFrame(results)
    sig_count = result_df["significant"].sum() if len(result_df) > 0 else 0
    logger.info(
        "Treatment response: %d significant biomarker changes out of %d tested.",
        sig_count,
        len(result_df),
    )
    return result_df

"""
Model Monitoring & Drift Detection Module
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Population Stability Index (PSI), feature drift detection, and
model performance drift analysis for production monitoring.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PSI Thresholds
# ─────────────────────────────────────────────
PSI_STABLE = 0.10       # PSI < 0.10: no significant change
PSI_MODERATE = 0.20     # 0.10 <= PSI < 0.20: moderate change, monitor
PSI_CRITICAL = 0.20     # PSI >= 0.20: significant population shift, investigate


def _psi_severity(psi: float) -> str:
    """Return human-readable severity label for a PSI value."""
    if psi < PSI_STABLE:
        return "stable"
    elif psi < PSI_MODERATE:
        return "moderate"
    else:
        return "critical"


# ─────────────────────────────────────────────
# Population Stability Index (PSI)
# ─────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    buckets: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))

    Interpretation:
      PSI < 0.10: No significant change
      0.10 <= PSI < 0.20: Moderate change — monitor closely
      PSI >= 0.20: Significant shift — model retraining recommended

    Args:
        expected: Reference (training) distribution.
        actual: Current (production) distribution.
        buckets: Number of equal-width bins.

    Returns:
        PSI value (float).
    """
    expected_arr = np.asarray(expected, dtype=np.float64)
    actual_arr = np.asarray(actual, dtype=np.float64)

    # Remove NaN
    expected_arr = expected_arr[~np.isnan(expected_arr)]
    actual_arr = actual_arr[~np.isnan(actual_arr)]

    if len(expected_arr) == 0 or len(actual_arr) == 0:
        logger.warning("Empty array passed to compute_psi — returning 0.0")
        return 0.0

    # Bin edges from expected distribution
    min_val = min(expected_arr.min(), actual_arr.min())
    max_val = max(expected_arr.max(), actual_arr.max())

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, buckets + 1)
    bins[0] -= 1e-6   # Include min
    bins[-1] += 1e-6  # Include max

    expected_counts, _ = np.histogram(expected_arr, bins=bins)
    actual_counts, _ = np.histogram(actual_arr, bins=bins)

    # Convert to proportions with Laplace smoothing (avoid log(0))
    expected_pct = (expected_counts + 1e-6) / (expected_counts.sum() + 1e-6 * buckets)
    actual_pct = (actual_counts + 1e-6) / (actual_counts.sum() + 1e-6 * buckets)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 6)


# ─────────────────────────────────────────────
# Feature-Level Drift
# ─────────────────────────────────────────────

def compute_feature_drift(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    psi_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute PSI for each numeric feature between training and current data.

    Flags features with PSI > psi_threshold as drifted.

    Args:
        train_df: Reference (training) DataFrame.
        current_df: Current (production) DataFrame.
        feature_cols: List of feature columns to analyze.
        psi_threshold: PSI threshold above which a feature is flagged.

    Returns:
        Dict with:
          - psi_per_feature: {feature: psi_value}
          - drifted_features: list of feature names with PSI > threshold
          - summary_stats: {feature: {psi, severity, train_mean, current_mean}}
    """
    psi_per_feature: Dict[str, float] = {}
    summary_stats: Dict[str, Any] = {}
    drifted_features: List[str] = []

    for col in feature_cols:
        if col not in train_df.columns or col not in current_df.columns:
            logger.debug(f"Column '{col}' not found in both DataFrames — skipping")
            continue

        train_vals = train_df[col].dropna()
        curr_vals = current_df[col].dropna()

        if len(train_vals) == 0 or len(curr_vals) == 0:
            psi_per_feature[col] = 0.0
            continue

        # Only compute PSI for numeric columns
        if not pd.api.types.is_numeric_dtype(train_vals):
            continue

        psi = compute_psi(train_vals.values, curr_vals.values)
        psi_per_feature[col] = psi

        severity = _psi_severity(psi)
        summary_stats[col] = {
            "psi": psi,
            "severity": severity,
            "train_mean": round(float(train_vals.mean()), 4),
            "current_mean": round(float(curr_vals.mean()), 4),
            "train_std": round(float(train_vals.std()), 4),
            "current_std": round(float(curr_vals.std()), 4),
            "flagged": psi > psi_threshold,
        }

        if psi > psi_threshold:
            drifted_features.append(col)

    n_drifted = len(drifted_features)
    n_total = len(psi_per_feature)
    logger.info(
        f"Feature drift analysis | {n_drifted}/{n_total} features drifted "
        f"(PSI threshold={psi_threshold})"
    )
    if drifted_features:
        logger.warning(f"Drifted features: {drifted_features}")

    return {
        "psi_per_feature": psi_per_feature,
        "drifted_features": drifted_features,
        "summary_stats": summary_stats,
        "n_features_analyzed": n_total,
        "n_drifted": n_drifted,
        "psi_threshold": psi_threshold,
    }


# ─────────────────────────────────────────────
# Model Performance Drift
# ─────────────────────────────────────────────

def compute_model_performance_drift(
    y_true_hist: np.ndarray | pd.Series,
    y_pred_hist: np.ndarray | pd.Series,
    y_true_curr: np.ndarray | pd.Series,
    y_pred_curr: np.ndarray | pd.Series,
) -> Dict[str, Any]:
    """
    Compare AUC-PR (Average Precision) between historical and current periods.

    Returns performance metrics for both periods and the delta.

    Args:
        y_true_hist: Historical true labels.
        y_pred_hist: Historical predicted probabilities.
        y_true_curr: Current true labels.
        y_pred_curr: Current predicted probabilities.

    Returns:
        Dict with historical/current AUC metrics and drift indicators.
    """
    y_true_hist_arr = np.asarray(y_true_hist)
    y_pred_hist_arr = np.asarray(y_pred_hist)
    y_true_curr_arr = np.asarray(y_true_curr)
    y_pred_curr_arr = np.asarray(y_pred_curr)

    # Historical metrics
    hist_auc_pr = float(average_precision_score(y_true_hist_arr, y_pred_hist_arr))
    hist_auc_roc = float(roc_auc_score(y_true_hist_arr, y_pred_hist_arr))
    hist_fraud_rate = float(np.mean(y_true_hist_arr))

    # Current metrics
    curr_auc_pr = float(average_precision_score(y_true_curr_arr, y_pred_curr_arr))
    curr_auc_roc = float(roc_auc_score(y_true_curr_arr, y_pred_curr_arr))
    curr_fraud_rate = float(np.mean(y_true_curr_arr))

    # Deltas
    delta_auc_pr = round(curr_auc_pr - hist_auc_pr, 4)
    delta_auc_roc = round(curr_auc_roc - hist_auc_roc, 4)

    # Performance degradation flag: >5% relative drop in AUC-PR
    degradation_threshold = 0.05
    relative_drop = (hist_auc_pr - curr_auc_pr) / max(hist_auc_pr, 1e-8)
    performance_degraded = bool(relative_drop > degradation_threshold)

    logger.info(
        f"Performance drift | "
        f"hist_AUC-PR={hist_auc_pr:.4f} → curr_AUC-PR={curr_auc_pr:.4f} "
        f"(delta={delta_auc_pr:+.4f}) | degraded={performance_degraded}"
    )

    return {
        "historical": {
            "auc_pr": round(hist_auc_pr, 4),
            "auc_roc": round(hist_auc_roc, 4),
            "fraud_rate": round(hist_fraud_rate, 4),
            "n_samples": len(y_true_hist_arr),
        },
        "current": {
            "auc_pr": round(curr_auc_pr, 4),
            "auc_roc": round(curr_auc_roc, 4),
            "fraud_rate": round(curr_fraud_rate, 4),
            "n_samples": len(y_true_curr_arr),
        },
        "delta": {
            "auc_pr": delta_auc_pr,
            "auc_roc": delta_auc_roc,
            "fraud_rate": round(curr_fraud_rate - hist_fraud_rate, 4),
        },
        "performance_degraded": performance_degraded,
        "relative_auc_pr_drop": round(relative_drop, 4),
        "degradation_threshold": degradation_threshold,
    }


# ─────────────────────────────────────────────
# Full Drift Report
# ─────────────────────────────────────────────

def generate_drift_report(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    y_true_train: Optional[np.ndarray] = None,
    y_true_current: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Generate a complete drift analysis report.

    Combines feature drift (PSI) and model performance drift metrics.

    Args:
        train_df: Reference training DataFrame.
        current_df: Current production DataFrame.
        model: Trained model with predict_proba method.
        feature_cols: Feature column names for PSI analysis.
        y_true_train: Historical true labels (optional, for performance drift).
        y_true_current: Current true labels (optional, for performance drift).

    Returns:
        Dict containing feature drift analysis, performance drift (if labels available),
        and an overall health status.
    """
    logger.info("Generating full drift report")

    # Feature drift
    feature_drift = compute_feature_drift(train_df, current_df, feature_cols)

    # Performance drift (only if labels available)
    performance_drift: Optional[Dict[str, Any]] = None
    if y_true_train is not None and y_true_current is not None:
        try:
            # Prepare feature matrices
            X_train_feat = train_df[feature_cols].fillna(-999)
            X_curr_feat = current_df[feature_cols].fillna(-999)

            from src.models import ensemble_predict_proba
            if isinstance(model, dict) and "xgb_models" in model:
                y_pred_train = ensemble_predict_proba(model, X_train_feat)
                y_pred_curr = ensemble_predict_proba(model, X_curr_feat)
            else:
                y_pred_train = model.predict_proba(X_train_feat)[:, 1]
                y_pred_curr = model.predict_proba(X_curr_feat)[:, 1]

            performance_drift = compute_model_performance_drift(
                y_true_train, y_pred_train,
                y_true_current, y_pred_curr,
            )
        except Exception as exc:
            logger.warning(f"Could not compute performance drift: {exc}")

    # Score distribution drift (PSI on model output scores)
    score_psi: Optional[float] = None
    try:
        X_train_feat = train_df[feature_cols].fillna(-999)
        X_curr_feat = current_df[feature_cols].fillna(-999)

        from src.models import ensemble_predict_proba
        if isinstance(model, dict) and "xgb_models" in model:
            train_scores = ensemble_predict_proba(model, X_train_feat)
            curr_scores = ensemble_predict_proba(model, X_curr_feat)
        else:
            train_scores = model.predict_proba(X_train_feat)[:, 1]
            curr_scores = model.predict_proba(X_curr_feat)[:, 1]

        score_psi = compute_psi(train_scores, curr_scores)
        logger.info(f"Score distribution PSI: {score_psi:.4f}")
    except Exception as exc:
        logger.warning(f"Could not compute score PSI: {exc}")

    # Overall health
    n_critical = sum(
        1 for stat in feature_drift["summary_stats"].values()
        if stat.get("severity") == "critical"
    )
    perf_degraded = (
        performance_drift.get("performance_degraded", False)
        if performance_drift else False
    )
    score_drift_critical = score_psi is not None and score_psi >= PSI_CRITICAL

    if n_critical > 0 or perf_degraded or score_drift_critical:
        health_status = "degraded"
    elif feature_drift["n_drifted"] > 0:
        health_status = "warning"
    else:
        health_status = "healthy"

    report: Dict[str, Any] = {
        "health_status": health_status,
        "feature_drift": feature_drift,
        "performance_drift": performance_drift,
        "score_distribution_psi": score_psi,
        "score_severity": _psi_severity(score_psi) if score_psi is not None else "unknown",
        "n_critical_features": n_critical,
        "top_drifted_features": sorted(
            feature_drift["psi_per_feature"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10],
        "recommendation": _get_drift_recommendation(
            n_critical, perf_degraded, score_drift_critical
        ),
    }

    logger.info(f"Drift report complete | health_status={health_status}")
    return report


def _get_drift_recommendation(
    n_critical_features: int,
    performance_degraded: bool,
    score_drift_critical: bool,
) -> str:
    """Return actionable recommendation based on drift severity."""
    if performance_degraded:
        return (
            "URGENT: Model performance has degraded significantly. "
            "Initiate model retraining with recent data immediately."
        )
    if score_drift_critical:
        return (
            "WARNING: Score distribution has shifted significantly. "
            "Review recent transaction patterns and consider retraining."
        )
    if n_critical_features > 0:
        return (
            f"CAUTION: {n_critical_features} feature(s) show critical drift. "
            "Investigate data pipeline for issues and schedule model refresh."
        )
    return "Model is operating within normal parameters. Continue routine monitoring."

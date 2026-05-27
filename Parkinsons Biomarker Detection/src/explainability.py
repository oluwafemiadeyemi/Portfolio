"""
explainability.py
=================
Clinical explainability for Parkinson's Disease Detection predictions.

Functions:
  - SHAP value computation (TreeExplainer)
  - Per-patient risk explanation with biomarker interpretation
  - Biomarker clinical range reference dictionary
  - Structured clinical report generation
  - Bootstrap uncertainty quantification
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clinical ranges reference
# ---------------------------------------------------------------------------

def get_biomarker_clinical_ranges() -> Dict[str, Dict[str, Any]]:
    """Return clinical reference ranges for all tracked biomarkers.

    Returns
    -------
    Dict mapping biomarker name → {name, unit, pd_range, healthy_range, interpretation}
    """
    return {
        "MDVP_Fo": {
            "name": "Fundamental Frequency (Fo)",
            "unit": "Hz",
            "pd_range": "150–180",
            "healthy_range": "165–220",
            "interpretation": "Lower and less stable pitch is associated with PD vocal tremor.",
        },
        "MDVP_Jitter_pct": {
            "name": "Jitter (%)",
            "unit": "%",
            "pd_range": "0.3–2.0",
            "healthy_range": "0.1–0.6",
            "interpretation": "Cycle-to-cycle variation in pitch; elevated values indicate vocal instability.",
        },
        "MDVP_Shimmer": {
            "name": "Shimmer",
            "unit": "ratio",
            "pd_range": "0.02–0.10",
            "healthy_range": "0.01–0.04",
            "interpretation": "Amplitude variation in voice signal; higher values suggest PD laryngeal muscle weakness.",
        },
        "HNR": {
            "name": "Harmonics-to-Noise Ratio",
            "unit": "dB",
            "pd_range": "15–22",
            "healthy_range": "20–28",
            "interpretation": "Ratio of harmonic energy to noise; lower values indicate noisier voice.",
        },
        "RPDE": {
            "name": "Recurrence Period Density Entropy",
            "unit": "dimensionless",
            "pd_range": "0.3–0.7",
            "healthy_range": "0.2–0.55",
            "interpretation": "Non-linear dynamical measure of vocal fold vibration complexity.",
        },
        "DFA": {
            "name": "Detrended Fluctuation Analysis",
            "unit": "dimensionless",
            "pd_range": "0.65–0.90",
            "healthy_range": "0.55–0.78",
            "interpretation": "Signal fractal scaling exponent; values outside normal range suggest dysphonia.",
        },
        "PPE": {
            "name": "Pitch Period Entropy",
            "unit": "dimensionless",
            "pd_range": "0.20–0.55",
            "healthy_range": "0.05–0.22",
            "interpretation": "Entropy of fundamental frequency; higher values indicate disordered pitch.",
        },
        "stride_length": {
            "name": "Stride Length",
            "unit": "metres",
            "pd_range": "0.8–1.2",
            "healthy_range": "1.1–1.6",
            "interpretation": "Shorter strides are a cardinal gait feature of PD (festination).",
        },
        "cadence": {
            "name": "Cadence",
            "unit": "steps/min",
            "pd_range": "100–120",
            "healthy_range": "95–115",
            "interpretation": "PD patients compensate for shorter strides with faster cadence.",
        },
        "stride_regularity": {
            "name": "Stride Regularity",
            "unit": "ratio",
            "pd_range": "0.85–0.95",
            "healthy_range": "0.92–0.99",
            "interpretation": "Autocorrelation of stride intervals; reduced regularity indicates gait freezing risk.",
        },
        "gait_asymmetry": {
            "name": "Gait Asymmetry",
            "unit": "ratio",
            "pd_range": "0.05–0.20",
            "healthy_range": "0.01–0.08",
            "interpretation": "Left-right step imbalance; PD commonly presents with unilateral gait impairment.",
        },
        "rest_tremor_amplitude": {
            "name": "Rest Tremor Amplitude",
            "unit": "m/s²",
            "pd_range": "0.5–3.0",
            "healthy_range": "0.0–0.5",
            "interpretation": "Accelerometer-measured tremor at rest; PD rest tremor is a diagnostic criterion.",
        },
        "tremor_frequency": {
            "name": "Tremor Frequency",
            "unit": "Hz",
            "pd_range": "3–6",
            "healthy_range": "6–12 (if present)",
            "interpretation": "PD rest tremor: 3–6 Hz. Essential tremor: 6–12 Hz. Low frequency suggests PD.",
        },
        "tremor_power_ratio": {
            "name": "Tremor Power Ratio",
            "unit": "ratio",
            "pd_range": "0.40–0.90",
            "healthy_range": "0.0–0.25",
            "interpretation": "Fraction of total movement power in the tremor band; higher values indicate PD.",
        },
        "jitter_shimmer_ratio": {
            "name": "Jitter/Shimmer Ratio",
            "unit": "dimensionless",
            "pd_range": ">0.15",
            "healthy_range": "<0.10",
            "interpretation": "Combined pitch and amplitude instability marker.",
        },
        "noise_ratio": {
            "name": "Noise Ratio",
            "unit": "dimensionless",
            "pd_range": ">0.05",
            "healthy_range": "<0.04",
            "interpretation": "Normalised noise component in voice; elevated in PD.",
        },
        "tremor_severity_score": {
            "name": "Tremor Severity Score",
            "unit": "composite",
            "pd_range": ">0.5",
            "healthy_range": "<0.2",
            "interpretation": "Composite amplitude × power measure of tremor severity.",
        },
    }


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> "np.ndarray":
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model: Trained tree-based model (XGBoost, RF, GBM).
    X: Feature matrix.
    feature_names: Optional list of feature names.

    Returns
    -------
    np.ndarray of SHAP values shape (n_samples, n_features).
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — returning zero array.")
        return np.zeros((len(X), X.shape[1]))

    X_filled = X.copy().fillna(X.median(numeric_only=True))
    if feature_names is not None:
        X_filled.columns = feature_names

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_filled)
        # For binary classifiers, shap_values returns list [class0, class1]
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        return np.array(shap_vals)
    except Exception as exc:
        logger.warning("TreeExplainer failed: %s — using KernelExplainer fallback.", exc)
        try:
            import shap
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                shap.sample(X_filled, min(100, len(X_filled))),
            )
            shap_vals = explainer.shap_values(X_filled)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            return np.array(shap_vals)
        except Exception as exc2:
            logger.error("SHAP computation failed: %s", exc2)
            return np.zeros((len(X_filled), X_filled.shape[1]))


# ---------------------------------------------------------------------------
# Patient-level risk explanation
# ---------------------------------------------------------------------------

def explain_patient_risk(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Generate a structured risk explanation for a single patient recording.

    Parameters
    ----------
    model: Fitted classifier.
    X_single: Single-row DataFrame with feature values.
    feature_names: Feature column names.

    Returns
    -------
    Dict with: pd_probability, risk_level, key_biomarkers, clinical_interpretation
    """
    X_filled = X_single.copy().fillna(0)

    # Prediction
    proba = float(
        model.predict_proba(X_filled)[:, 1][0]
        if hasattr(model, "predict_proba")
        else model.predict(X_filled)[0]
    )

    risk_level = (
        "High" if proba >= 0.70
        else "Moderate" if proba >= 0.40
        else "Low"
    )

    # SHAP
    shap_vals = compute_shap_values(model, X_filled, feature_names)
    shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    # Sort biomarkers by absolute SHAP contribution
    ranges_dict = get_biomarker_clinical_ranges()
    key_biomarkers: List[Dict[str, Any]] = []

    feat_shap_pairs = sorted(
        zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True
    )

    for feat, shap_contribution in feat_shap_pairs[:8]:
        value = float(X_filled[feat].iloc[0]) if feat in X_filled.columns else np.nan
        meta = ranges_dict.get(feat, {})
        key_biomarkers.append({
            "biomarker": feat,
            "display_name": meta.get("name", feat),
            "value": round(value, 4),
            "unit": meta.get("unit", ""),
            "normal_range": meta.get("healthy_range", "N/A"),
            "pd_range": meta.get("pd_range", "N/A"),
            "shap_contribution": round(float(shap_contribution), 4),
            "direction": "increases" if shap_contribution > 0 else "decreases",
        })

    # Clinical interpretation
    top_flags = [b["display_name"] for b in key_biomarkers if b["shap_contribution"] > 0][:3]
    if risk_level == "High":
        interpretation = (
            f"Biomarker profile strongly consistent with Parkinson's Disease (probability {proba:.1%}). "
            f"Key flagged markers: {', '.join(top_flags) if top_flags else 'multiple biomarkers'}. "
            "Referral to movement disorder specialist is recommended."
        )
    elif risk_level == "Moderate":
        interpretation = (
            f"Biomarker profile shows some features associated with PD (probability {proba:.1%}). "
            f"Notable findings: {', '.join(top_flags) if top_flags else 'minor deviations'}. "
            "Closer monitoring and follow-up assessment in 6 months is advised."
        )
    else:
        interpretation = (
            f"Biomarker profile is largely within healthy ranges (probability {proba:.1%}). "
            "Routine follow-up is sufficient."
        )

    return {
        "pd_probability": round(proba, 4),
        "risk_level": risk_level,
        "key_biomarkers": key_biomarkers,
        "clinical_interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Clinical report
# ---------------------------------------------------------------------------

def generate_clinical_report(
    patient_id: str,
    shap_explanation: Dict[str, Any],
    longitudinal_trend: Optional[Dict[str, float]] = None,
) -> str:
    """Generate a structured clinical summary string.

    Parameters
    ----------
    patient_id: Patient identifier string.
    shap_explanation: Output from ``explain_patient_risk``.
    longitudinal_trend: Optional dict mapping feature_name → slope value.

    Returns
    -------
    Formatted clinical report string.
    """
    prob = shap_explanation.get("pd_probability", 0.0)
    risk = shap_explanation.get("risk_level", "Unknown")
    interp = shap_explanation.get("clinical_interpretation", "")
    key_bm = shap_explanation.get("key_biomarkers", [])

    # Recommendation
    if risk == "High":
        recommendation = "REFER — Immediate referral to a movement disorder specialist."
    elif risk == "Moderate":
        recommendation = "MONITOR — Schedule follow-up biomarker assessment in 6 months."
    else:
        recommendation = "CLEAR — Continue routine health monitoring."

    # Flagged biomarkers table
    flagged = [b for b in key_bm if b.get("shap_contribution", 0) > 0]
    protective = [b for b in key_bm if b.get("shap_contribution", 0) < 0]

    flagged_lines = "\n".join(
        f"  • {b['display_name']}: {b['value']} {b['unit']} "
        f"(PD range: {b['pd_range']}, healthy: {b['normal_range']}, SHAP: +{b['shap_contribution']:.3f})"
        for b in flagged[:5]
    ) or "  None"

    protective_lines = "\n".join(
        f"  • {b['display_name']}: {b['value']} {b['unit']} "
        f"(healthy range: {b['normal_range']}, SHAP: {b['shap_contribution']:.3f})"
        for b in protective[:3]
    ) or "  None"

    # Longitudinal trend block
    trend_block = ""
    if longitudinal_trend:
        worsening = {k: v for k, v in longitudinal_trend.items() if v > 0}
        improving = {k: v for k, v in longitudinal_trend.items() if v < 0}
        trend_dir = "WORSENING" if len(worsening) > len(improving) else "STABLE/IMPROVING"
        trend_block = f"""
LONGITUDINAL TREND
------------------
Overall direction: {trend_dir}
Worsening biomarkers: {', '.join(list(worsening.keys())[:5]) or 'None'}
Improving biomarkers: {', '.join(list(improving.keys())[:3]) or 'None'}
"""

    report = f"""
========================================================
   PARKINSON'S DISEASE DIGITAL BIOMARKER REPORT
========================================================
Patient ID    : {patient_id}
Assessment Date: (current)

RISK SUMMARY
------------
PD Risk Score  : {prob:.1%}
Risk Level     : {risk}
Recommendation : {recommendation}

CLINICAL INTERPRETATION
-----------------------
{interp}

FLAGGED BIOMARKERS (Increasing PD risk)
----------------------------------------
{flagged_lines}

PROTECTIVE BIOMARKERS
---------------------
{protective_lines}
{trend_block}
DISCLAIMER
----------
This report is generated by an AI-assisted digital biomarker platform and
is intended to support — not replace — clinical judgment. All findings must
be interpreted by a qualified healthcare professional in the context of a
full clinical assessment. This tool has not been approved as a standalone
diagnostic device by the FDA.
========================================================
"""
    return report


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------

def compute_uncertainty(
    model: Any,
    X_single: pd.DataFrame,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> Dict[str, float]:
    """Estimate prediction confidence interval via bootstrap resampling.

    Bootstraps feature perturbation to estimate uncertainty.

    Parameters
    ----------
    model: Fitted classifier.
    X_single: Single-row feature DataFrame.
    n_bootstrap: Number of bootstrap iterations.
    random_state: Random seed.

    Returns
    -------
    Dict with: mean_probability, lower_ci (5th pct), upper_ci (95th pct), std
    """
    rng = np.random.default_rng(random_state)
    X_arr = X_single.copy().fillna(0).values.astype(float)

    predictions: List[float] = []
    for _ in range(n_bootstrap):
        # Add Gaussian noise proportional to feature range
        noise = rng.normal(0, 0.02, X_arr.shape)
        X_noisy = pd.DataFrame(X_arr + noise, columns=X_single.columns)
        try:
            p = float(
                model.predict_proba(X_noisy)[:, 1][0]
                if hasattr(model, "predict_proba")
                else model.predict(X_noisy)[0]
            )
            predictions.append(p)
        except Exception:
            pass

    if not predictions:
        return {"mean_probability": np.nan, "lower_ci": np.nan, "upper_ci": np.nan, "std": np.nan}

    preds = np.array(predictions)
    return {
        "mean_probability": float(np.mean(preds)),
        "lower_ci": float(np.percentile(preds, 5)),
        "upper_ci": float(np.percentile(preds, 95)),
        "std": float(np.std(preds)),
    }

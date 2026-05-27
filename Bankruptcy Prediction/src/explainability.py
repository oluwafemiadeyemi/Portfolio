"""
explainability.py
=================
Explainability functions for Financial Distress / Bankruptcy Prediction.

Functions:
  - SHAP values (TreeExplainer)
  - Company-level risk explanation with Altman context
  - Professional risk narrative generation
  - Counterfactual analysis (what financial improvements reduce risk)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    model: Fitted tree-based model (XGBoost, LightGBM, GBM).
    X: Feature matrix.
    feature_names: Optional feature names override.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features).
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — returning zero array.")
        return np.zeros((len(X), X.shape[1]))

    X_filled = X.copy().fillna(X.median(numeric_only=True))
    if feature_names is not None and len(feature_names) == X_filled.shape[1]:
        X_filled.columns = feature_names

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_filled)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        return np.array(shap_vals)
    except Exception as exc:
        logger.warning("TreeExplainer failed: %s. Trying KernelExplainer.", exc)
        try:
            background = shap.sample(X_filled, min(50, len(X_filled)))
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_vals = explainer.shap_values(X_filled)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            return np.array(shap_vals)
        except Exception as exc2:
            logger.error("SHAP computation failed: %s", exc2)
            return np.zeros((len(X_filled), X_filled.shape[1]))


# ---------------------------------------------------------------------------
# Company-level risk explanation
# ---------------------------------------------------------------------------

def explain_company_risk(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
    company_name: str = "Company",
    altman_zscore: Optional[float] = None,
    sector: Optional[str] = None,
    peer_percentile: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate structured risk explanation for a single company.

    Parameters
    ----------
    model: Fitted binary classifier.
    X_single: Single-row feature DataFrame.
    feature_names: Feature column names.
    company_name: Company display name.
    altman_zscore: Pre-computed Altman Z-score (if available).
    sector: Company sector name.
    peer_percentile: Company's percentile rank within its sector (0–100).

    Returns
    -------
    Dict with distress_probability, altman_zone, risk_level, top_risk_factors,
    top_protective_factors, peer_comparison, narrative.
    """
    X_filled = X_single.copy().fillna(0)

    # Prediction
    proba = float(
        model.predict_proba(X_filled)[:, 1][0]
        if hasattr(model, "predict_proba")
        else model.predict(X_filled)[0]
    )

    # Risk level
    if proba >= 0.70:
        risk_level = "Critical"
    elif proba >= 0.45:
        risk_level = "High"
    elif proba >= 0.25:
        risk_level = "Elevated"
    else:
        risk_level = "Low"

    # Altman zone
    if altman_zscore is not None:
        if altman_zscore > 2.99:
            altman_zone = "Safe"
        elif altman_zscore >= 1.81:
            altman_zone = "Grey Zone"
        else:
            altman_zone = "Distress Zone"
    else:
        altman_zone = "Unknown"

    # SHAP
    shap_vals = compute_shap_values(model, X_filled, feature_names)
    shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    feat_shap_pairs = sorted(
        zip(feature_names, shap_row),
        key=lambda x: x[1],
        reverse=True,
    )

    top_risk_factors = [
        {
            "factor": feat,
            "value": round(float(X_filled[feat].iloc[0]), 4) if feat in X_filled.columns else 0.0,
            "shap_contribution": round(float(sv), 4),
        }
        for feat, sv in feat_shap_pairs
        if sv > 0
    ][:5]

    top_protective_factors = [
        {
            "factor": feat,
            "value": round(float(X_filled[feat].iloc[0]), 4) if feat in X_filled.columns else 0.0,
            "shap_contribution": round(float(sv), 4),
        }
        for feat, sv in feat_shap_pairs
        if sv < 0
    ][-5:][::-1]  # Reverse to get most protective first

    # Peer comparison
    peer_comparison = None
    if peer_percentile is not None:
        if peer_percentile < 25:
            peer_comparison = f"Bottom quartile (worst {peer_percentile:.0f}%) of {sector or 'sector'} peers"
        elif peer_percentile < 50:
            peer_comparison = f"Below median ({peer_percentile:.0f}th percentile) in {sector or 'sector'}"
        elif peer_percentile < 75:
            peer_comparison = f"Above median ({peer_percentile:.0f}th percentile) in {sector or 'sector'}"
        else:
            peer_comparison = f"Top quartile ({peer_percentile:.0f}th percentile) in {sector or 'sector'}"

    # Narrative
    narrative = generate_risk_narrative(company_name, {
        "distress_probability": proba,
        "risk_level": risk_level,
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,
    }, altman_zscore, sector)

    return {
        "distress_probability": round(proba, 4),
        "altman_zscore": round(altman_zscore, 3) if altman_zscore is not None else None,
        "altman_zone": altman_zone,
        "risk_level": risk_level,
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,
        "peer_comparison": peer_comparison,
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Risk narrative
# ---------------------------------------------------------------------------

def generate_risk_narrative(
    company_name: str,
    shap_explanation: Dict[str, Any],
    altman_zscore: Optional[float],
    sector: Optional[str],
) -> str:
    """Generate a professional risk report paragraph.

    Parameters
    ----------
    company_name: Display name for the company.
    shap_explanation: Output from ``explain_company_risk``.
    altman_zscore: Altman Z-score value (or None).
    sector: Sector name for context.

    Returns
    -------
    Multi-sentence risk narrative string.
    """
    prob = shap_explanation.get("distress_probability", 0.0)
    risk = shap_explanation.get("risk_level", "Unknown")
    top_risk = shap_explanation.get("top_risk_factors", [])
    top_protective = shap_explanation.get("top_protective_factors", [])

    risk_factors_str = ", ".join(f["factor"].replace("_", " ") for f in top_risk[:3]) or "multiple financial metrics"
    protective_str = ", ".join(f["factor"].replace("_", " ") for f in top_protective[:2]) or "stable cash flow"

    altman_context = ""
    if altman_zscore is not None:
        if altman_zscore < 1.81:
            altman_context = (
                f" The Altman Z-Score of {altman_zscore:.2f} places {company_name} "
                f"firmly in the distress zone (threshold: 1.81), consistent with elevated bankruptcy risk."
            )
        elif altman_zscore < 2.99:
            altman_context = (
                f" The Altman Z-Score of {altman_zscore:.2f} places {company_name} "
                f"in the grey zone (1.81–2.99), indicating material financial uncertainty."
            )
        else:
            altman_context = (
                f" The Altman Z-Score of {altman_zscore:.2f} suggests financial safety (threshold: 2.99), "
                f"though the ML model detects subtler risk signals not captured by the traditional formula."
            )

    sector_context = f" in the {sector} sector" if sector else ""

    if risk in ("Critical", "High"):
        opening = (
            f"{company_name}{sector_context} exhibits a {prob:.1%} probability of financial distress, "
            f"representing a {risk.lower()} credit risk."
        )
        body = (
            f" The primary risk drivers are elevated {risk_factors_str}, "
            f"which are strongly associated with financial deterioration in comparable firms."
        )
        closing = (
            f" Mitigating factors include {protective_str}."
            f"{altman_context}"
            f" Immediate credit review and covenant monitoring are recommended."
        )
    elif risk == "Elevated":
        opening = (
            f"{company_name}{sector_context} shows a {prob:.1%} financial distress probability, "
            f"warranting elevated monitoring."
        )
        body = (
            f" Key risk indicators include {risk_factors_str}."
            f"{altman_context}"
        )
        closing = (
            f" Protective factors ({protective_str}) partially offset these concerns. "
            f"Quarterly monitoring of liquidity and leverage metrics is advised."
        )
    else:
        opening = (
            f"{company_name}{sector_context} demonstrates a low {prob:.1%} financial distress probability."
        )
        body = f" Financial profile is supported by {protective_str}.{altman_context}"
        closing = " Standard annual credit review is appropriate."

    return opening + body + closing


# ---------------------------------------------------------------------------
# Counterfactual analysis
# ---------------------------------------------------------------------------

def compute_counterfactual(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
    target_risk: float = 0.25,
    max_iterations: int = 50,
) -> Dict[str, Any]:
    """Identify financial improvements that would reduce a company's risk level.

    Uses a greedy feature perturbation approach: iteratively improves the
    features with the highest positive SHAP contributions until the target
    risk level is reached or iterations are exhausted.

    Parameters
    ----------
    model: Fitted binary classifier.
    X_single: Single-row feature DataFrame.
    feature_names: Feature column names.
    target_risk: Desired maximum distress probability (default 0.25 = "Elevated").
    max_iterations: Maximum perturbation iterations.

    Returns
    -------
    Dict with: current_probability, target_probability, achievable,
               required_changes (list of {feature, current_value, target_value, change_pct})
    """
    X_filled = X_single.copy().fillna(0)
    current_proba = float(
        model.predict_proba(X_filled)[:, 1][0]
        if hasattr(model, "predict_proba")
        else model.predict(X_filled)[0]
    )

    if current_proba <= target_risk:
        return {
            "current_probability": round(current_proba, 4),
            "target_probability": target_risk,
            "achievable": True,
            "required_changes": [],
            "message": "Company already below target risk threshold.",
        }

    shap_vals = compute_shap_values(model, X_filled, feature_names)
    shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    # Rank features by positive SHAP contribution (highest risk drivers)
    risk_features = sorted(
        [(feat, sv) for feat, sv in zip(feature_names, shap_row) if sv > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    X_counter = X_filled.copy()
    changes: List[Dict[str, Any]] = []
    achievable = False

    # Improvement direction heuristics
    improve_by_decreasing = {
        "debt_to_equity", "debt_to_assets", "debt_to_equity_delta",
        "altman_distress_zone", "negative_news_spike",
        "interest_expense_ratio",
    }
    improve_by_increasing = {
        "current_ratio", "quick_ratio", "cash_ratio",
        "altman_zscore", "ROA", "ROE", "EBITDA_margin",
        "news_sentiment_30d", "altman_re_ta", "altman_X2_weighted",
        "revenue_growth_yoy",
    }

    for feat, _ in risk_features[:max_iterations]:
        if feat not in X_counter.columns:
            continue

        current_val = float(X_counter[feat].iloc[0])

        if feat in improve_by_decreasing:
            new_val = current_val * 0.80  # Reduce by 20%
        elif feat in improve_by_increasing:
            new_val = current_val * 1.20  # Increase by 20%
        else:
            # Default: move 20% toward zero (reduce magnitude)
            new_val = current_val * 0.80

        old_val = X_counter.at[X_counter.index[0], feat]
        X_counter.at[X_counter.index[0], feat] = new_val

        new_proba = float(
            model.predict_proba(X_counter)[:, 1][0]
            if hasattr(model, "predict_proba")
            else model.predict(X_counter)[0]
        )

        changes.append({
            "feature": feat,
            "current_value": round(current_val, 4),
            "target_value": round(new_val, 4),
            "change_pct": round((new_val - current_val) / (abs(current_val) + 1e-9) * 100, 1),
            "probability_after": round(new_proba, 4),
        })

        if new_proba <= target_risk:
            achievable = True
            break

    final_proba = float(
        model.predict_proba(X_counter)[:, 1][0]
        if hasattr(model, "predict_proba")
        else model.predict(X_counter)[0]
    )

    return {
        "current_probability": round(current_proba, 4),
        "target_probability": target_risk,
        "achieved_probability": round(final_proba, 4),
        "achievable": achievable,
        "required_changes": changes,
        "message": (
            f"Risk reduced from {current_proba:.1%} to {final_proba:.1%} "
            f"by modifying {len(changes)} financial metric(s)."
        ),
    }

"""
Model Explainability for Churn Prediction.
SHAP-based feature importance and user-level narratives.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — using permutation importance fallback.")


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 5000,
    background_samples: int = 100,
) -> tuple[Any, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer for tree models,
    KernelExplainer as fallback for others.

    Returns: (explainer, shap_values_array)
    shap_values_array shape: (n_samples, n_features) for binary classification
    """
    X_clean = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)

    # Subsample for efficiency
    if len(X_clean) > max_samples:
        X_sample = X_clean.sample(max_samples, random_state=42)
    else:
        X_sample = X_clean

    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available — using coefficient/feature_importance proxy.")
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            shap_vals = np.outer(np.ones(len(X_sample)), imp)
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
            shap_vals = np.outer(np.ones(len(X_sample)), imp * X_sample.values)
        else:
            shap_vals = np.zeros((len(X_sample), len(feature_names)))
        return None, shap_vals

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        logger.info(f"SHAP TreeExplainer computed for {len(X_sample):,} samples.")
        return explainer, shap_values

    except Exception as exc:
        logger.warning(f"TreeExplainer failed ({exc}) — trying KernelExplainer...")
        try:
            background = shap.sample(X_sample, min(background_samples, len(X_sample)))

            def predict_fn(x: np.ndarray) -> np.ndarray:
                return model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1]

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample.values[:200], nsamples=100)
            logger.info("SHAP KernelExplainer computed.")
            return explainer, shap_values
        except Exception as exc2:
            logger.error(f"SHAP computation failed: {exc2}")
            return None, np.zeros((len(X_sample), len(feature_names)))


def explain_churn_risk(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Explain churn risk for a single user.
    Returns top churn factors and top retention factors with SHAP values.
    """
    X_clean = X_single[feature_names].fillna(0).replace([np.inf, -np.inf], 0).head(1)

    churn_prob = float(model.predict_proba(X_clean)[0, 1])

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_clean)
            if isinstance(sv, list):
                sv = sv[1]
            shap_row = sv[0]
        except Exception:
            shap_row = _fallback_shap(model, X_clean, feature_names)
    else:
        shap_row = _fallback_shap(model, X_clean, feature_names)

    feature_contributions = pd.Series(shap_row, index=feature_names)
    sorted_contribs = feature_contributions.sort_values(ascending=False)

    top_churn_factors = [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "feature_value": round(float(X_clean[feat].iloc[0]), 4),
            "direction": "increases_churn_risk",
        }
        for feat, val in sorted_contribs[sorted_contribs > 0].head(top_n).items()
    ]

    top_retention_factors = [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "feature_value": round(float(X_clean[feat].iloc[0]), 4),
            "direction": "decreases_churn_risk",
        }
        for feat, val in sorted_contribs[sorted_contribs < 0].tail(top_n).items()
    ]

    return {
        "churn_probability": round(churn_prob, 4),
        "top_churn_factors": top_churn_factors,
        "top_retention_factors": top_retention_factors,
        "all_shap_values": {
            feat: round(float(val), 4)
            for feat, val in feature_contributions.items()
        },
    }


def _fallback_shap(model: Any, X_clean: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Compute approximate SHAP-like contributions using feature importance or coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_values = X_clean.values[0]
        std = feature_values.std() if feature_values.std() > 0 else 1.0
        shap_approx = importances * (feature_values - feature_values.mean()) / std
        return shap_approx
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return coef * X_clean.values[0]
    elif hasattr(model, "lr_"):
        coef = model.lr_.coef_[0]
        return coef * X_clean.values[0]
    else:
        return np.zeros(len(feature_names))


def generate_user_narrative(
    user_id: str,
    churn_prob: float,
    clv: float,
    uplift_segment: str,
    shap_explanation: dict[str, Any],
    tenure_days: Optional[int] = None,
    lifecycle_stage: Optional[str] = None,
    recommended_action: Optional[str] = None,
) -> str:
    """
    Generate a human-readable narrative explaining a user's churn risk profile.

    Example output:
    "User U0001234 has 78% churn risk. CLV: $450. Primary churn drivers:
    23 days without login, completion rate dropped 40%.
    Recommended: send premium offer + 3-month discount."
    """
    risk_level = (
        "CRITICAL" if churn_prob >= 0.75
        else "HIGH" if churn_prob >= 0.55
        else "MEDIUM" if churn_prob >= 0.35
        else "LOW"
    )

    top_churn = shap_explanation.get("top_churn_factors", [])[:3]
    top_retain = shap_explanation.get("top_retention_factors", [])[:2]

    churn_driver_strs = []
    for factor in top_churn:
        feat = factor["feature"].replace("_", " ").replace("rfm ", "")
        val = factor["feature_value"]
        churn_driver_strs.append(f"{feat}: {val:.2f}")

    retention_str_parts = []
    for factor in top_retain:
        feat = factor["feature"].replace("_", " ")
        val = factor["feature_value"]
        retention_str_parts.append(f"{feat}: {val:.2f}")

    narrative_parts = [
        f"User {user_id} has {churn_prob:.0%} churn risk ({risk_level}).",
        f"Predicted CLV: ${clv:,.0f}.",
    ]

    if tenure_days is not None:
        narrative_parts.append(
            f"Tenure: {tenure_days} days ({lifecycle_stage or 'established'} stage)."
        )

    if churn_driver_strs:
        narrative_parts.append(f"Primary churn drivers: {', '.join(churn_driver_strs)}.")

    if retention_str_parts:
        narrative_parts.append(f"Protective factors: {', '.join(retention_str_parts)}.")

    narrative_parts.append(f"Uplift segment: {uplift_segment}.")

    if recommended_action:
        narrative_parts.append(f"Recommended action: {recommended_action}.")
    else:
        action_map = {
            "Persuadables": "Send targeted retention offer immediately.",
            "Sure Things": "Reward loyalty — no aggressive intervention needed.",
            "Lost Causes": "Low save probability — minimal intervention budget.",
            "Sleeping Dogs": "Do NOT contact — intervention may increase churn.",
        }
        narrative_parts.append(action_map.get(uplift_segment, "Monitor and reassess."))

    return " ".join(narrative_parts)

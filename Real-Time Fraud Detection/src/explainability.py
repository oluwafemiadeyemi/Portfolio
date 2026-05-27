"""
Explainability Module
Real-Time Credit Risk & Fraud Detection Intelligence Platform

SHAP-based model explanations for regulatory compliance (FCRA/ECOA).
Supports TreeExplainer for XGBoost/LightGBM, feature attribution ranking,
and human-readable adverse action reason generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SHAP Feature Importance Mapping
# Maps feature names to human-readable descriptions for adverse action notices
# ─────────────────────────────────────────────

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "TransactionAmt": "Transaction amount",
    "log_amount": "Transaction amount (log scale)",
    "amount_zscore_per_card": "Amount deviation from your card's typical pattern",
    "amt_deviation_from_card_avg": "Amount significantly above historical average",
    "amount_rounded": "Unusual round-number transaction amount",
    "tx_count_1h": "Number of transactions in the past hour",
    "tx_count_6h": "Number of transactions in the past 6 hours",
    "tx_count_24h": "Number of transactions in the past 24 hours",
    "amt_sum_1h": "Total spending volume in the past hour",
    "amt_sum_6h": "Total spending volume in the past 6 hours",
    "amt_sum_24h": "Total spending volume in the past 24 hours",
    "hour_of_day": "Time of transaction",
    "day_of_week": "Day of week pattern",
    "is_night": "Unusual transaction time (late night/early morning)",
    "is_weekend": "Weekend transaction pattern",
    "email_domain_match": "Email domain mismatch between purchaser and recipient",
    "p_email_risk_score": "High-risk purchaser email domain",
    "r_email_risk_score": "High-risk recipient email domain",
    "max_email_risk_score": "Use of high-risk or temporary email domain",
    "card1": "Card identifier pattern",
    "card2": "Card feature 2 deviation",
    "C1": "Unusual account activity count (C1)",
    "C2": "Unusual account activity count (C2)",
    "C3": "Unusual account activity count (C3)",
    "C5": "Unusual account activity count (C5)",
    "C6": "Unusual account activity count (C6)",
    "C9": "Unusual account activity count (C9)",
    "C13": "Unusual account activity count (C13)",
    "C14": "Unusual account activity count (C14)",
    "D1": "Account age / time since last transaction",
    "D2": "Time delta feature pattern",
    "D4": "Time delta feature pattern",
    "V258": "Vesta risk signal (V258)",
    "V257": "Vesta risk signal (V257)",
    "addr1": "Billing address mismatch",
    "addr2": "Billing country code anomaly",
}

# ─────────────────────────────────────────────
# SHAP Value Computation
# ─────────────────────────────────────────────

def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    max_samples: int = 1000,
) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """
    Compute SHAP values using TreeExplainer (XGBoost / LightGBM compatible).

    Subsamples to max_samples for speed while maintaining representativeness.

    Args:
        model: Trained tree-based model (XGBClassifier, LGBMClassifier,
               or ensemble dict with 'xgb_models'/'lgb_models').
        X: Feature matrix (DataFrame or ndarray).
        max_samples: Maximum number of rows to explain.

    Returns:
        Tuple of (shap_values array of shape [n, features], explainer object).
    """
    # For ensemble dict, use the best individual model (first XGB fold model)
    if isinstance(model, dict) and "xgb_models" in model:
        underlying_model = model["xgb_models"][0]
        logger.info("Using first XGBoost fold model for SHAP explanation")
    elif isinstance(model, dict) and "lgb_models" in model:
        underlying_model = model["lgb_models"][0]
        logger.info("Using first LightGBM fold model for SHAP explanation")
    else:
        underlying_model = model

    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        feature_names = list(X.columns)
    else:
        X_arr = X
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    # Subsample
    n = len(X_arr)
    if n > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=max_samples, replace=False)
        X_sample = X_arr[indices]
        logger.info(f"Subsampled {n:,} → {max_samples:,} rows for SHAP computation")
    else:
        X_sample = X_arr
        logger.info(f"Computing SHAP values for {n:,} rows")

    # Fill NaN for SHAP (trees handle -999 sentinel fine, but TreeExplainer needs clean arrays)
    X_sample = np.where(np.isnan(X_sample), -999, X_sample)

    explainer = shap.TreeExplainer(
        underlying_model,
        feature_names=feature_names,
    )

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, LightGBM may return a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class

    logger.info(f"SHAP values computed: shape={shap_values.shape}")
    return shap_values, explainer


# ─────────────────────────────────────────────
# Top Feature Extraction
# ─────────────────────────────────────────────

def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    n: int = 15,
) -> List[Dict[str, Any]]:
    """
    Return the top N features ranked by mean absolute SHAP value.

    Args:
        shap_values: SHAP array of shape [n_samples, n_features].
        feature_names: Feature name list aligned with shap_values columns.
        n: Number of top features to return.

    Returns:
        List of dicts: [{feature, mean_abs_shap, rank}, ...]
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked_indices = np.argsort(mean_abs_shap)[::-1][:n]

    top_features = [
        {
            "rank": int(i + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": float(mean_abs_shap[idx]),
        }
        for i, idx in enumerate(ranked_indices)
    ]

    logger.debug(
        f"Top feature: {top_features[0]['feature']} "
        f"(mean |SHAP|={top_features[0]['mean_abs_shap']:.4f})"
    )
    return top_features


# ─────────────────────────────────────────────
# Single Prediction Explanation
# ─────────────────────────────────────────────

def explain_prediction(
    model: Any,
    X_single: pd.DataFrame | np.ndarray,
    feature_names: List[str],
    expected_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a full explanation for a single transaction prediction.

    Computes fraud probability, assigns risk level, and returns the top
    positive (fraud-pushing) and negative (legit-pushing) SHAP factors.

    Args:
        model: Trained model (XGB, LGB, or ensemble dict).
        X_single: Single-row feature matrix (1 × n_features).
        feature_names: Feature names aligned with X_single columns.
        expected_value: SHAP base value (if pre-computed). If None, computed fresh.

    Returns:
        Dict with keys:
          - prediction (int): 0 or 1 using 0.5 threshold
          - probability (float): fraud probability
          - risk_level (str): 'low' | 'medium' | 'high'
          - top_positive_factors: list of {feature, value, shap_value, description}
          - top_negative_factors: list of {feature, value, shap_value, description}
          - expected_value (float): SHAP base rate
    """
    # Ensure 2-D
    if isinstance(X_single, pd.DataFrame):
        X_arr = X_single.values.astype(np.float32)
    else:
        X_arr = np.atleast_2d(X_single).astype(np.float32)

    X_clean = np.where(np.isnan(X_arr), -999.0, X_arr)

    # Get the underlying model for SHAP
    if isinstance(model, dict) and "xgb_models" in model:
        underlying = model["xgb_models"][0]
    elif isinstance(model, dict) and "lgb_models" in model:
        underlying = model["lgb_models"][0]
    else:
        underlying = model

    # Probability
    proba_2d = underlying.predict_proba(X_clean)
    probability = float(proba_2d[0, 1])

    # Risk level
    if probability < 0.3:
        risk_level = "low"
    elif probability < 0.7:
        risk_level = "medium"
    else:
        risk_level = "high"

    prediction = int(probability >= 0.5)

    # SHAP for this single row
    explainer = shap.TreeExplainer(underlying)
    sv = explainer.shap_values(X_clean)

    # For binary LGB, sv may be list
    if isinstance(sv, list):
        sv = sv[1]

    sv_row = sv[0]  # shape (n_features,)
    base_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value[1])

    # Build factor dicts
    factors = []
    for i, (feat_name, shap_val) in enumerate(zip(feature_names, sv_row)):
        feat_val = float(X_arr[0, i]) if i < X_arr.shape[1] else 0.0
        description = FEATURE_DESCRIPTIONS.get(feat_name, feat_name.replace("_", " ").title())
        factors.append(
            {
                "feature": feat_name,
                "value": feat_val,
                "shap_value": float(shap_val),
                "description": description,
            }
        )

    # Sort into positive (fraud-increasing) and negative (fraud-decreasing)
    factors_sorted_pos = sorted(
        [f for f in factors if f["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )[:5]

    factors_sorted_neg = sorted(
        [f for f in factors if f["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:5]

    result: Dict[str, Any] = {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "top_positive_factors": factors_sorted_pos,
        "top_negative_factors": factors_sorted_neg,
        "expected_value": base_value,
    }

    logger.debug(
        f"Prediction explained | prob={probability:.4f} | risk={risk_level} | "
        f"top_factor={factors_sorted_pos[0]['feature'] if factors_sorted_pos else 'N/A'}"
    )
    return result


# ─────────────────────────────────────────────
# Adverse Action Reason Generation (FCRA)
# ─────────────────────────────────────────────

# FCRA-compliant adverse action code mapping
ADVERSE_ACTION_CODES: Dict[str, str] = {
    "is_night": "Unusual transaction time pattern (late night / early morning activity)",
    "hour_of_day": "Unusual transaction time pattern",
    "tx_count_1h": "Excessive transaction frequency in a short time window",
    "tx_count_6h": "Abnormally high transaction velocity in the past 6 hours",
    "tx_count_24h": "Elevated transaction count over the past 24 hours",
    "amt_sum_1h": "Unusual spending volume in the past hour",
    "amt_sum_6h": "Elevated cumulative spending in the past 6 hours",
    "TransactionAmt": "Transaction amount outside normal spending pattern",
    "log_amount": "Transaction amount outside normal spending pattern",
    "amount_zscore_per_card": "Transaction amount significantly deviates from card history",
    "amt_deviation_from_card_avg": "Amount significantly above historical average for this card",
    "amount_rounded": "Suspicious round-number transaction amount",
    "email_domain_match": "Email address domain mismatch between purchaser and recipient",
    "p_email_risk_score": "Purchaser email domain associated with elevated risk",
    "r_email_risk_score": "Recipient email domain associated with elevated risk",
    "max_email_risk_score": "Use of high-risk, temporary, or anonymizing email service",
    "C1": "Unusual account activity patterns detected",
    "C2": "Abnormal transaction count metrics",
    "C6": "Unusual card usage frequency",
    "C13": "Elevated account-level activity count",
    "C14": "Abnormal linked account activity",
    "D1": "Account history inconsistency detected",
    "D2": "Unusual time pattern between transactions",
    "addr1": "Billing address inconsistency",
    "addr2": "Geographic pattern inconsistency",
    "card1": "Card usage pattern anomaly detected",
    "V258": "Complex risk signal pattern detected",
    "V257": "Complex risk signal pattern detected",
}

_DEFAULT_ADVERSE_ACTION = "Transaction pattern inconsistent with historical behavior"


def generate_adverse_action_reasons(
    shap_explanation: Dict[str, Any],
    n: int = 3,
) -> List[str]:
    """
    Generate human-readable adverse action reason codes per FCRA requirements.

    FCRA Section 615(a) requires specific reasons when adverse action is taken
    based on a credit score or risk model output.

    Args:
        shap_explanation: Dict returned by explain_prediction().
        n: Number of adverse action reasons to return (typically 3-5).

    Returns:
        List of n human-readable FCRA-compliant adverse action reason strings.
    """
    positive_factors = shap_explanation.get("top_positive_factors", [])

    reasons: List[str] = []
    seen_reasons: set[str] = set()

    for factor in positive_factors:
        feature_name = factor.get("feature", "")
        shap_val = factor.get("shap_value", 0.0)

        if shap_val <= 0:
            continue

        # Look up FCRA reason code
        reason = ADVERSE_ACTION_CODES.get(feature_name)
        if reason is None:
            # Generate from feature description
            description = factor.get("description", feature_name.replace("_", " ").title())
            reason = f"Unusual pattern detected: {description}"

        if reason not in seen_reasons:
            reasons.append(reason)
            seen_reasons.add(reason)

        if len(reasons) >= n:
            break

    # Pad with default if insufficient reasons
    while len(reasons) < n:
        default = _DEFAULT_ADVERSE_ACTION
        if default not in seen_reasons:
            reasons.append(default)
            seen_reasons.add(default)
        else:
            break

    logger.debug(f"Generated {len(reasons)} adverse action reasons")
    return reasons[:n]

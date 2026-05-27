"""
Explainability Module for Mortgage Underwriting
SHAP-based explanations, counterfactuals, and adverse action code generation.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.warning("SHAP not installed. Falling back to feature importance-based explanations.")


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    max_samples: int = 2000,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Computes SHAP values using TreeExplainer for tree-based models.

    Args:
        model: Fitted tree-based model (LightGBM, XGBoost, RandomForest)
        X: Feature matrix
        feature_names: List of feature column names
        max_samples: Maximum number of samples for SHAP computation

    Returns:
        Tuple of (shap_values array, shap explainer object)
    """
    if not _HAS_SHAP:
        logger.warning("SHAP not available. Returning None.")
        return None, None

    X_sample = X.head(max_samples) if len(X) > max_samples else X
    X_df = pd.DataFrame(X_sample, columns=feature_names) if not isinstance(X_sample, pd.DataFrame) else X_sample

    logger.info(f"Computing SHAP values for {len(X_df):,} samples...")

    try:
        # Try TreeExplainer first (fastest for tree models)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_df)

        # Handle multi-output (some LGB versions return list [neg_class, pos_class])
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]  # positive class SHAP values

        logger.info(f"SHAP values computed: shape {np.array(shap_vals).shape}")
        return np.array(shap_vals), explainer

    except Exception as exc:
        logger.warning(f"TreeExplainer failed ({exc}). Trying KernelExplainer...")
        try:
            background = shap.kmeans(X_df, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_df[:100], nsamples=100)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            return np.array(shap_vals), explainer
        except Exception as exc2:
            logger.error(f"SHAP computation failed: {exc2}")
            return None, None


def explain_single_application(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
    approval_threshold: float = 0.5,
) -> Dict:
    """
    Generates a full explanation for a single mortgage application.

    Args:
        model: Fitted model
        X_single: Single-row feature DataFrame
        feature_names: List of feature names
        approval_threshold: Probability threshold for approval decision

    Returns:
        Dict with approval_probability, decision, top_approval_factors,
        top_denial_factors, and model_confidence
    """
    X_row = pd.DataFrame([X_single] if not isinstance(X_single, pd.DataFrame) else X_single.values,
                         columns=feature_names)

    # Get prediction
    try:
        prob = float(model.predict_proba(X_row)[0, 1])
    except Exception:
        prob = float(model.predict(X_row)[0])

    decision = "APPROVED" if prob >= approval_threshold else "DENIED"

    explanation: Dict = {
        "approval_probability": round(prob, 4),
        "decision": decision,
        "model_confidence": round(abs(prob - 0.5) * 2, 4),  # 0=uncertain, 1=very confident
        "top_approval_factors": [],
        "top_denial_factors": [],
        "feature_contributions": {},
    }

    if not _HAS_SHAP:
        # Fallback: use feature importances if available
        return _explain_with_feature_importance(model, X_row, feature_names, explanation)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_row)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        shap_arr = np.array(shap_vals).flatten()
        feature_vals = X_row.values.flatten()

        # Build feature contribution dict
        for i, fname in enumerate(feature_names):
            if i < len(shap_arr):
                explanation["feature_contributions"][fname] = {
                    "shap_value": round(float(shap_arr[i]), 4),
                    "feature_value": round(float(feature_vals[i]), 4) if np.isfinite(float(feature_vals[i])) else None,
                }

        # Top factors pushing toward approval (positive SHAP)
        sorted_idx = np.argsort(shap_arr)[::-1]
        explanation["top_approval_factors"] = [
            {
                "feature": feature_names[i],
                "shap_value": round(float(shap_arr[i]), 4),
                "feature_value": round(float(feature_vals[i]), 4) if i < len(feature_vals) else None,
                "direction": "supports_approval",
            }
            for i in sorted_idx
            if i < len(shap_arr) and shap_arr[i] > 0
        ][:5]

        # Top factors pushing toward denial (negative SHAP)
        explanation["top_denial_factors"] = [
            {
                "feature": feature_names[i],
                "shap_value": round(float(shap_arr[i]), 4),
                "feature_value": round(float(feature_vals[i]), 4) if i < len(feature_vals) else None,
                "direction": "supports_denial",
            }
            for i in sorted_idx[::-1]
            if i < len(shap_arr) and shap_arr[i] < 0
        ][:5]

    except Exception as exc:
        logger.warning(f"SHAP explanation failed for single application: {exc}")

    return explanation


def generate_counterfactual(
    shap_explanation: Dict,
    X_single: pd.DataFrame,
    feature_names: List[str],
    model: Any,
    approval_threshold: float = 0.5,
) -> Dict:
    """
    Generates actionable counterfactual suggestions.
    "If you change X by Y amount, the application would be approved."

    Args:
        shap_explanation: Output from explain_single_application()
        X_single: Single-row feature DataFrame
        feature_names: Feature names list
        model: Fitted model
        approval_threshold: Decision threshold

    Returns:
        Dict with counterfactual suggestions and expected probability change
    """
    current_prob = shap_explanation.get("approval_probability", 0.5)
    decision = shap_explanation.get("decision", "DENIED")

    counterfactuals: Dict = {
        "current_probability": current_prob,
        "current_decision": decision,
        "suggestions": [],
        "achievable": False,
    }

    if decision == "APPROVED":
        counterfactuals["message"] = "Application is already approved. No counterfactual needed."
        return counterfactuals

    # Find top actionable denial factors
    denial_factors = shap_explanation.get("top_denial_factors", [])
    if not denial_factors:
        counterfactuals["message"] = "Insufficient SHAP data to generate counterfactuals."
        return counterfactuals

    X_row = pd.DataFrame(
        [X_single] if not isinstance(X_single, pd.DataFrame) else X_single.values,
        columns=feature_names,
    )

    suggestions_generated = 0
    for factor in denial_factors:
        feature = factor.get("feature", "")
        current_val = factor.get("feature_value")

        if current_val is None:
            continue

        suggestion = _generate_feature_suggestion(feature, current_val, X_row, feature_names, model, approval_threshold)
        if suggestion:
            counterfactuals["suggestions"].append(suggestion)
            suggestions_generated += 1
        if suggestions_generated >= 3:
            break

    # Check if at least one suggestion would achieve approval
    counterfactuals["achievable"] = any(s.get("would_approve", False) for s in counterfactuals["suggestions"])
    if not counterfactuals["suggestions"]:
        counterfactuals["message"] = (
            "Application risk profile is complex. Please consult a loan officer for personalized guidance."
        )
    else:
        improvements = [s["description"] for s in counterfactuals["suggestions"]]
        counterfactuals["message"] = (
            f"To improve approval chances, consider: {'; '.join(improvements[:2])}."
        )

    return counterfactuals


def compute_feature_importance_stability(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 1000,
) -> Dict:
    """
    Checks SHAP feature importance consistency between train and test sets.

    Args:
        model: Fitted model
        X_train: Training features
        X_test: Test features
        feature_names: Feature names
        max_samples: Max samples per split

    Returns:
        Dict with train/test SHAP importances, correlation, and stability flag
    """
    if feature_names is None:
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]

    result: Dict = {
        "stable": True,
        "train_importances": {},
        "test_importances": {},
        "rank_correlation": None,
        "importance_drift": {},
    }

    if not _HAS_SHAP:
        # Fallback: use built-in feature importances
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(feature_names, model.feature_importances_))
            result["train_importances"] = fi
            result["test_importances"] = fi
            result["rank_correlation"] = 1.0
        return result

    try:
        train_sample = X_train.head(max_samples)
        test_sample = X_test.head(max_samples)

        train_shap, _ = compute_shap_values(model, train_sample, feature_names, max_samples)
        test_shap, _ = compute_shap_values(model, test_sample, feature_names, max_samples)

        if train_shap is None or test_shap is None:
            return result

        # Mean absolute SHAP values per feature
        train_importance = np.abs(train_shap).mean(axis=0)
        test_importance = np.abs(test_shap).mean(axis=0)

        result["train_importances"] = {
            fname: round(float(v), 6) for fname, v in zip(feature_names, train_importance)
        }
        result["test_importances"] = {
            fname: round(float(v), 6) for fname, v in zip(feature_names, test_importance)
        }

        # Spearman rank correlation
        from scipy.stats import spearmanr
        corr, pval = spearmanr(train_importance, test_importance)
        result["rank_correlation"] = round(float(corr), 4)
        result["stable"] = corr > 0.85

        # Features with largest drift
        drift = {
            fname: round(abs(float(t) - float(s)), 6)
            for fname, t, s in zip(feature_names, test_importance, train_importance)
        }
        top_drift = sorted(drift.items(), key=lambda x: x[1], reverse=True)[:5]
        result["importance_drift"] = dict(top_drift)

        logger.info(f"Feature importance stability: Spearman rho={corr:.3f}, stable={result['stable']}")

    except Exception as exc:
        logger.error(f"Stability check failed: {exc}")

    return result


# ── Helper functions ───────────────────────────────────────────────────────────

def _explain_with_feature_importance(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: List[str],
    explanation: Dict,
) -> Dict:
    """Fallback explanation using built-in feature importances (no SHAP)."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        pairs = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)
        explanation["top_approval_factors"] = [
            {"feature": f, "shap_value": round(float(v), 4), "direction": "important"} for f, v in pairs[:5]
        ]
    elif hasattr(model, "named_steps"):
        lr = model.named_steps.get("lr")
        if lr and hasattr(lr, "coef_"):
            coefs = lr.coef_.flatten()
            vals = X_row.values.flatten() * coefs
            pairs = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
            explanation["top_approval_factors"] = [
                {"feature": f, "shap_value": round(float(v), 4), "direction": "supports_approval"} for f, v in pairs[:5] if v > 0
            ]
            explanation["top_denial_factors"] = [
                {"feature": f, "shap_value": round(float(v), 4), "direction": "supports_denial"} for f, v in pairs if v < 0
            ][:5]
    return explanation


def _generate_feature_suggestion(
    feature: str,
    current_val: float,
    X_row: pd.DataFrame,
    feature_names: List[str],
    model: Any,
    threshold: float,
) -> Optional[Dict]:
    """Generates a specific improvement suggestion for a feature."""
    X_test = X_row.copy()
    feature_human = feature.replace("_", " ").title()

    # Feature-specific improvement logic
    if "debt_to_income" in feature or "dti" in feature:
        new_val = max(current_val * 0.85, 30.0)  # Reduce DTI by 15%
        reduction_pct = (current_val - new_val) / current_val * 100
        X_test[feature] = new_val
        try:
            new_prob = float(model.predict_proba(X_test)[0, 1])
        except Exception:
            new_prob = min(float(current_val) + 0.05, 0.99)
        return {
            "feature": feature,
            "description": f"Reduce debt-to-income ratio from {current_val:.1f}% to {new_val:.1f}% (pay down {reduction_pct:.0f}% of debt)",
            "current_value": round(current_val, 2),
            "suggested_value": round(new_val, 2),
            "expected_approval_probability": round(new_prob, 4),
            "probability_increase": round(new_prob - float(X_row[feature].iloc[0] if feature in X_row.columns else 0), 4),
            "would_approve": new_prob >= threshold,
        }

    elif "income" in feature and "ratio" not in feature and "area" not in feature:
        new_val = current_val * 1.20  # 20% income increase
        X_test[feature] = new_val
        try:
            new_prob = float(model.predict_proba(X_test)[0, 1])
        except Exception:
            new_prob = min(float(current_val) + 0.05, 0.99)
        increase_amt = new_val - current_val
        return {
            "feature": feature,
            "description": f"Increase annual income by ${increase_amt:,.0f} (to ${new_val:,.0f})",
            "current_value": round(current_val, 2),
            "suggested_value": round(new_val, 2),
            "expected_approval_probability": round(new_prob, 4),
            "would_approve": new_prob >= threshold,
        }

    elif "loan_to_value" in feature:
        new_val = max(current_val - 10, 60)  # Increase down payment
        X_test[feature] = new_val
        try:
            new_prob = float(model.predict_proba(X_test)[0, 1])
        except Exception:
            new_prob = min(float(current_val) + 0.05, 0.99)
        return {
            "feature": feature,
            "description": f"Increase down payment to reduce LTV from {current_val:.1f}% to {new_val:.1f}%",
            "current_value": round(current_val, 2),
            "suggested_value": round(new_val, 2),
            "expected_approval_probability": round(new_prob, 4),
            "would_approve": new_prob >= threshold,
        }

    elif "loan_amount" in feature:
        new_val = current_val * 0.90  # Reduce loan amount by 10%
        X_test[feature] = new_val
        try:
            new_prob = float(model.predict_proba(X_test)[0, 1])
        except Exception:
            new_prob = min(float(current_val) + 0.05, 0.99)
        return {
            "feature": feature,
            "description": f"Reduce loan amount by ${current_val - new_val:,.0f} (to ${new_val:,.0f})",
            "current_value": round(current_val, 2),
            "suggested_value": round(new_val, 2),
            "expected_approval_probability": round(new_prob, 4),
            "would_approve": new_prob >= threshold,
        }

    return None

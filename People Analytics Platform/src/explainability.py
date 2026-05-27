"""
Explainability Module for Employee Attrition Prediction
SHAP-based explanations, intervention recommendations, and HR narrative generation.
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
    logger.warning("SHAP not installed. Using feature importance fallback.")


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    max_samples: int = 2000,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Computes SHAP values using TreeExplainer for tree-based models.

    Args:
        model: Fitted tree-based model
        X: Feature matrix
        feature_names: Feature column names
        max_samples: Max samples for SHAP computation

    Returns:
        Tuple of (shap_values array, explainer object)
    """
    if not _HAS_SHAP:
        return None, None

    X_sample = X.head(max_samples) if len(X) > max_samples else X
    X_df = pd.DataFrame(X_sample, columns=feature_names) if not isinstance(X_sample, pd.DataFrame) else X_sample

    logger.info(f"Computing SHAP values for {len(X_df):,} employees...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_df)

        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]  # Attrition class

        logger.info(f"SHAP values shape: {np.array(shap_vals).shape}")
        return np.array(shap_vals), explainer

    except Exception as exc:
        logger.warning(f"TreeExplainer failed: {exc}")
        try:
            background = shap.kmeans(X_df.head(200), min(50, len(X_df)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_df.head(100), nsamples=50)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            return np.array(shap_vals), explainer
        except Exception as exc2:
            logger.error(f"SHAP fallback failed: {exc2}")
            return None, None


def explain_flight_risk(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
    employee_id: Any = None,
    attrition_threshold: float = 0.5,
) -> Dict:
    """
    Explains flight risk for a single employee with top risk and retention factors.

    Args:
        model: Fitted model
        X_single: Single-row feature DataFrame
        feature_names: Feature column names
        employee_id: Employee identifier for the report
        attrition_threshold: Probability threshold for "at risk" classification

    Returns:
        Dict with attrition_probability, risk_factors, retention_factors, explanation_text
    """
    X_row = pd.DataFrame(
        [X_single.values[0]] if isinstance(X_single, pd.DataFrame) else [X_single],
        columns=feature_names,
    )

    try:
        prob = float(model.predict_proba(X_row)[0, 1])
    except Exception:
        prob = float(model.predict(X_row)[0])

    at_risk = prob >= attrition_threshold

    explanation: Dict = {
        "employee_id": employee_id,
        "attrition_probability": round(prob, 4),
        "at_risk": at_risk,
        "risk_level": _risk_level(prob),
        "top_risk_factors": [],
        "top_retention_factors": [],
        "feature_contributions": {},
    }

    if not _HAS_SHAP:
        return _explain_with_feature_importance(model, X_row, feature_names, explanation)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_row)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        shap_arr = np.array(shap_vals).flatten()
        feat_vals = X_row.values.flatten()

        for i, fname in enumerate(feature_names):
            if i < len(shap_arr):
                explanation["feature_contributions"][fname] = {
                    "shap_value": round(float(shap_arr[i]), 4),
                    "feature_value": round(float(feat_vals[i]), 4) if np.isfinite(float(feat_vals[i])) else None,
                }

        sorted_idx = np.argsort(shap_arr)[::-1]

        # Top factors increasing attrition risk (positive SHAP toward attrition class)
        explanation["top_risk_factors"] = [
            {
                "feature": _humanize_feature(feature_names[i]),
                "raw_feature": feature_names[i],
                "shap_value": round(float(shap_arr[i]), 4),
                "feature_value": round(float(feat_vals[i]), 4) if i < len(feat_vals) else None,
                "explanation": _get_risk_factor_explanation(feature_names[i], float(feat_vals[i]) if i < len(feat_vals) else 0),
            }
            for i in sorted_idx if i < len(shap_arr) and shap_arr[i] > 0
        ][:5]

        # Top retention factors (negative SHAP = reducing attrition risk)
        explanation["top_retention_factors"] = [
            {
                "feature": _humanize_feature(feature_names[i]),
                "raw_feature": feature_names[i],
                "shap_value": round(float(shap_arr[i]), 4),
                "feature_value": round(float(feat_vals[i]), 4) if i < len(feat_vals) else None,
                "explanation": _get_retention_factor_explanation(feature_names[i], float(feat_vals[i]) if i < len(feat_vals) else 0),
            }
            for i in sorted_idx[::-1] if i < len(shap_arr) and shap_arr[i] < 0
        ][:5]

    except Exception as exc:
        logger.warning(f"SHAP explanation failed for employee {employee_id}: {exc}")

    return explanation


def generate_intervention_recommendations(
    shap_explanation: Dict,
    df_employee: pd.DataFrame,
) -> List[Dict]:
    """
    Generates top 3 actionable intervention recommendations based on SHAP analysis.

    Args:
        shap_explanation: Output from explain_flight_risk()
        df_employee: Single-row DataFrame with employee data

    Returns:
        List of recommendation dicts with action, expected_risk_reduction, cost_estimate
    """
    risk_factors = shap_explanation.get("top_risk_factors", [])
    recommendations = []
    emp = df_employee.iloc[0] if len(df_employee) > 0 else {}

    # Map risk factors to interventions
    intervention_library = {
        "overtime": {
            "action": "Redistribute overtime workload — assign 2+ team members to absorb excess work",
            "cost_estimate": 3000.0,
            "risk_reduction": 0.15,
            "description": "OverTime is the #1 predictor of attrition in IBM HR data.",
        },
        "satisfaction_composite": {
            "action": "Conduct 1:1 engagement review and create individualized development plan",
            "cost_estimate": 500.0,
            "risk_reduction": 0.12,
            "description": "Low satisfaction composite correlates strongly with attrition.",
        },
        "stagnation_flag": {
            "action": "Initiate promotion review or lateral growth opportunity discussion",
            "cost_estimate": 8000.0,
            "risk_reduction": 0.18,
            "description": "3+ years without promotion/role change strongly predicts departure.",
        },
        "below_market_flag": {
            "action": "Conduct compensation benchmarking review and adjust to 50th percentile",
            "cost_estimate": float(emp.get("MonthlyIncome", 5000)) * 0.10 * 12 if hasattr(emp, "get") else 6000.0,
            "risk_reduction": 0.14,
            "description": "Below-market pay is a primary retention vulnerability.",
        },
        "pay_equity_gap": {
            "action": "Address compensation gap through equity adjustment in next review cycle",
            "cost_estimate": 7500.0,
            "risk_reduction": 0.10,
            "description": "Negative pay equity gap creates flight risk.",
        },
        "years_since_last_promotion": {
            "action": "Fast-track for promotion consideration or title advancement in H1",
            "cost_estimate": 10000.0,
            "risk_reduction": 0.20,
            "description": "Long time since last promotion is a key attrition predictor.",
        },
        "career_velocity": {
            "action": "Create stretch assignment or high-visibility project opportunity",
            "cost_estimate": 2000.0,
            "risk_reduction": 0.10,
            "description": "Stalled career velocity is a leading attrition indicator.",
        },
        "work_life_balance": {
            "action": "Implement flexible work arrangements (remote/hybrid days, schedule adjustment)",
            "cost_estimate": 1500.0,
            "risk_reduction": 0.12,
            "description": "Poor work-life balance is among top reasons cited in exit interviews.",
        },
    }

    seen_actions = set()
    for factor in risk_factors[:5]:
        raw_feature = factor.get("raw_feature", "").lower()
        matched = False

        for key, intervention in intervention_library.items():
            if key in raw_feature and key not in seen_actions:
                recommendations.append({
                    "priority": len(recommendations) + 1,
                    "action": intervention["action"],
                    "expected_risk_reduction": intervention["risk_reduction"],
                    "cost_estimate": round(intervention["cost_estimate"], 2),
                    "root_cause": factor.get("explanation", ""),
                    "feature": factor.get("feature", ""),
                    "shap_impact": factor.get("shap_value", 0),
                    "payback_period": _compute_payback(
                        intervention["cost_estimate"],
                        float(emp.get("MonthlyIncome", 5000)) * 12 * 1.5,
                        intervention["risk_reduction"],
                    ),
                })
                seen_actions.add(key)
                matched = True
                break

        if not matched and len(recommendations) < 3:
            recommendations.append({
                "priority": len(recommendations) + 1,
                "action": f"Address '{factor.get('feature', 'key risk factor')}' — consult with HR business partner",
                "expected_risk_reduction": 0.08,
                "cost_estimate": 2000.0,
                "root_cause": factor.get("explanation", ""),
                "feature": factor.get("feature", ""),
                "shap_impact": factor.get("shap_value", 0),
                "payback_period": "6-12 months",
            })

        if len(recommendations) >= 3:
            break

    return recommendations[:3]


def generate_hr_narrative(employee_summary: Dict) -> str:
    """
    Generates a template-based HR narrative for an employee's flight risk profile.
    No LLM required — uses structured f-strings with employee data.

    Args:
        employee_summary: Dict with employee data including attrition_probability,
                          top risk factors, recommendations, etc.

    Returns:
        Formatted HR narrative string
    """
    emp_id = employee_summary.get("employee_id", "Unknown")
    prob = employee_summary.get("attrition_probability", 0.0)
    prob_pct = f"{prob:.0%}"
    risk_level = employee_summary.get("risk_level", "Medium")

    risk_factors = employee_summary.get("top_risk_factors", [])
    recommendations = employee_summary.get("recommendations", [])

    dept = employee_summary.get("department", "their department")
    role = employee_summary.get("job_role", "their role")
    tenure = employee_summary.get("years_at_company", 0)
    income = employee_summary.get("monthly_income", 0)
    income_annual = income * 12
    replacement_cost = employee_summary.get("replacement_cost", income_annual * 1.5)
    elv = employee_summary.get("elv", 0)

    # Primary risk factors text
    risk_text = ""
    if risk_factors:
        factor_parts = []
        for f in risk_factors[:3]:
            factor_parts.append(f.get("explanation", f.get("feature", "")))
        risk_text = "; ".join(factor_parts)
    else:
        risk_text = "below-average satisfaction and limited growth opportunities"

    # Recommended actions text
    action_text = ""
    if recommendations:
        rec_parts = []
        for rec in recommendations[:2]:
            action_str = rec.get("action", "")
            cost_str = f"(${rec.get('cost_estimate', 0):,.0f} est.)"
            rec_parts.append(f"{action_str} {cost_str}")
        action_text = "; ".join(rec_parts)
    else:
        action_text = "consult HR business partner for personalized retention plan"

    # Attrition risk interpretation
    if prob >= 0.75:
        urgency = "URGENT: Immediate retention intervention required."
    elif prob >= 0.50:
        urgency = "HIGH PRIORITY: Retention action recommended within 30 days."
    elif prob >= 0.30:
        urgency = "MONITOR: Review in next 1:1 and quarterly check-in."
    else:
        urgency = "LOW RISK: Maintain standard engagement practices."

    narrative = (
        f"EMPLOYEE FLIGHT RISK REPORT\n"
        f"{'='*50}\n"
        f"Employee ID: {emp_id}\n"
        f"Role: {role} | Department: {dept}\n"
        f"Tenure: {tenure:.0f} year(s)\n\n"
        f"RISK ASSESSMENT\n"
        f"Attrition Probability: {prob_pct} ({risk_level} Risk)\n"
        f"{urgency}\n\n"
        f"PRIMARY RISK DRIVERS\n"
        f"{risk_text}\n\n"
        f"FINANCIAL CONTEXT\n"
        f"Monthly Income: ${income:,.0f} | Annual: ${income_annual:,.0f}\n"
        f"Replacement Cost (SHRM Formula): ${replacement_cost:,.0f}\n"
        f"Employee Lifetime Value (ELV): ${elv:,.0f}\n\n"
        f"RECOMMENDED ACTIONS\n"
        f"{action_text}\n\n"
        f"NOTE: This analysis is generated by the People Analytics AI system "
        f"for HR business partner review. All retention decisions should involve "
        f"direct manager and HR partnership. Attrition probabilities are model "
        f"estimates and should be validated with qualitative assessment."
    )

    return narrative


# ── Helper functions ───────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    """Maps probability to human-readable risk level."""
    if prob >= 0.75:
        return "Critical"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"


def _humanize_feature(feature_name: str) -> str:
    """Converts feature name to human-readable label."""
    humanized = {
        "OverTime": "Overtime Hours",
        "satisfaction_composite": "Overall Satisfaction Score",
        "stagnation_flag": "Career Stagnation",
        "below_market_flag": "Below-Market Compensation",
        "pay_equity_gap": "Compensation Gap vs. Peers",
        "YearsSinceLastPromotion": "Years Since Last Promotion",
        "career_velocity": "Career Growth Rate",
        "WorkLifeBalance": "Work-Life Balance Score",
        "JobSatisfaction": "Job Satisfaction",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "DistanceFromHome": "Commute Distance",
        "MonthlyIncome": "Monthly Income",
        "YearsAtCompany": "Tenure at Company",
        "NumCompaniesWorked": "Number of Previous Companies",
        "JobLevel": "Job Level",
        "tenure_loyalty_score": "Tenure Loyalty",
        "flight_risk_score": "Composite Flight Risk Score",
    }
    for k, v in humanized.items():
        if k.lower() in feature_name.lower():
            return v
    return feature_name.replace("_", " ").title()


def _get_risk_factor_explanation(feature: str, value: float) -> str:
    """Returns explanation string for why a feature is a risk factor."""
    explanations = {
        "overtime": f"Working overtime significantly increases attrition risk (current: {'Yes' if value > 0.5 else 'No'})",
        "satisfaction_composite": f"Low satisfaction score ({value:.1f}/4.0) drives departure intent",
        "stagnation": "3+ years without promotion or role change is a leading attrition predictor",
        "below_market": "Compensation below 25th percentile for role creates flight risk",
        "years_since_last_promotion": f"{value:.0f} years since last promotion exceeds healthy benchmark of 2 years",
        "career_velocity": f"Career growth rate ({value:.1f}) is below peer benchmarks",
        "work_life_balance": f"Work-life balance score ({value:.1f}/4.0) indicates burnout risk",
        "distance": f"Commute of {value:.0f} units increases attrition propensity",
        "num_companies": f"History of {value:.0f} previous employers indicates mobile career",
    }
    for key, explanation in explanations.items():
        if key in feature.lower():
            return explanation
    return f"{_humanize_feature(feature)} is contributing to attrition risk (value: {value:.2f})"


def _get_retention_factor_explanation(feature: str, value: float) -> str:
    """Returns explanation string for why a feature is a retention factor."""
    explanations = {
        "tenure_loyalty": f"Strong tenure loyalty score ({value:.2f}) indicates company attachment",
        "monthly_income": f"Competitive income (${value * 1000:,.0f}/mo) supports retention",
        "job_satisfaction": f"High job satisfaction ({value:.1f}/4.0) is a strong retention anchor",
        "stock_option": f"Stock option level {value:.0f} creates financial retention incentive",
        "years_at_company": f"{value:.0f} years of institutional knowledge creates switching costs",
        "job_level": f"Job level {value:.0f} indicates senior standing and investment",
    }
    for key, explanation in explanations.items():
        if key in feature.lower():
            return explanation
    return f"{_humanize_feature(feature)} is a protective retention factor (value: {value:.2f})"


def _compute_payback(intervention_cost: float, replacement_cost: float, risk_reduction: float) -> str:
    """Computes payback period string for an intervention."""
    if replacement_cost > 0 and risk_reduction > 0:
        expected_savings = replacement_cost * risk_reduction
        if expected_savings > 0:
            months = (intervention_cost / expected_savings) * 12
            return f"{months:.0f} months"
    return "Immediate (risk mitigation value)"


def _explain_with_feature_importance(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: List[str],
    explanation: Dict,
) -> Dict:
    """Fallback explanation using built-in feature importances."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        pairs = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)
        explanation["top_risk_factors"] = [
            {"feature": _humanize_feature(f), "raw_feature": f, "shap_value": round(float(v), 4),
             "explanation": f"High feature importance: {_humanize_feature(f)}"}
            for f, v in pairs[:5]
        ]
    elif hasattr(model, "named_steps"):
        lr = model.named_steps.get("lr")
        if lr and hasattr(lr, "coef_"):
            coefs = lr.coef_.flatten()
            feat_vals = X_row.values.flatten()
            contributions = feat_vals * coefs
            pairs = sorted(zip(feature_names, contributions), key=lambda x: x[1], reverse=True)
            explanation["top_risk_factors"] = [
                {"feature": _humanize_feature(f), "raw_feature": f, "shap_value": round(float(v), 4),
                 "explanation": f"Positive contribution to attrition risk: {_humanize_feature(f)}"}
                for f, v in pairs[:5] if v > 0
            ]
    return explanation

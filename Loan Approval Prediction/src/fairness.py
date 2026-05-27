"""
Fair Lending Analysis Module
ECOA/Fair Housing Act compliance testing for mortgage underwriting models.
Implements demographic parity, equalized odds, disparate impact, and adverse action codes.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ECOA protected attributes
PROTECTED_ATTRIBUTES = {
    "applicant_sex": {
        1: "Male",
        2: "Female",
        3: "Info Not Provided",
        4: "Not Applicable",
        6: "Both",
    },
    "applicant_race_1": {
        1: "American Indian/Alaska Native",
        2: "Asian",
        3: "Black/African American",
        4: "Native Hawaiian/Pacific Islander",
        5: "White",
        6: "Info Not Provided",
        7: "Not Applicable",
    },
    "applicant_ethnicity_1": {
        1: "Hispanic/Latino",
        2: "Not Hispanic/Latino",
        3: "Info Not Provided",
        4: "Not Applicable",
    },
}

# HMDA Adverse Action Reason Codes (Regulation B, 12 CFR Part 202)
ADVERSE_ACTION_CODES = {
    1: "Debt-to-income ratio",
    2: "Employment history",
    3: "Credit history",
    4: "Collateral (property value/type)",
    5: "Insufficient cash (downpayment/closing costs)",
    6: "Unverifiable information",
    7: "Credit application incomplete",
    8: "Mortgage insurance denied",
    9: "Other",
    10: "Loan-to-value ratio",
    11: "Insufficient income",
    12: "Temporary employment or self-employment",
    13: "Credit obligations (existing debt)",
}

# Feature → adverse action code mapping
_FEATURE_TO_CODE = {
    "debt_to_income_ratio": 1,
    "dti": 1,
    "combined_ltv_dti_score": 1,
    "income": 11,
    "monthly_payment_estimate": 11,
    "payment_to_income_ratio": 11,
    "affordability_index": 11,
    "loan_to_value_ratio": 10,
    "loan_amount": 4,
    "property_type": 4,
    "lien_status": 4,
    "employment": 2,
    "loan_term": 9,
    "hoepa_status": 9,
    "reverse_mortgage": 9,
    "balloon_payment": 9,
}


def compute_demographic_parity_difference(
    y_pred: np.ndarray,
    protected_col: str,
    df: pd.DataFrame,
    reference_group: Optional[Any] = None,
) -> Dict:
    """
    Computes the demographic parity difference (approval rate gap between groups).

    Args:
        y_pred: Binary predictions
        protected_col: Column name of the protected attribute
        df: DataFrame with protected attribute column
        reference_group: Value of the reference/majority group (auto-detected if None)

    Returns:
        Dict with approval rates per group, max gap, and reference group
    """
    if protected_col not in df.columns:
        logger.warning(f"Protected column '{protected_col}' not found in DataFrame")
        return {}

    y_pred_arr = np.asarray(y_pred)
    results: Dict = {"attribute": protected_col, "groups": {}, "max_disparity": 0.0}

    group_rates = {}
    for group_val in sorted(df[protected_col].dropna().unique()):
        mask = df[protected_col] == group_val
        if mask.sum() < 10:
            continue
        approval_rate = y_pred_arr[mask.values].mean()
        label = _get_group_label(protected_col, group_val)
        group_rates[label] = {"approval_rate": round(float(approval_rate), 4), "n": int(mask.sum()), "code": group_val}

    results["groups"] = group_rates

    if len(group_rates) >= 2:
        rates = [v["approval_rate"] for v in group_rates.values()]
        results["max_disparity"] = round(max(rates) - min(rates), 4)
        results["max_approval_group"] = max(group_rates, key=lambda k: group_rates[k]["approval_rate"])
        results["min_approval_group"] = min(group_rates, key=lambda k: group_rates[k]["approval_rate"])

    return results


def compute_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_col: str,
    df: pd.DataFrame,
) -> Dict:
    """
    Computes equalized odds: TPR and FPR by demographic group.

    Args:
        y_true: True binary labels
        y_pred: Binary predictions
        protected_col: Protected attribute column name
        df: DataFrame with protected attribute

    Returns:
        Dict with TPR, FPR, TNR per group and max TPR/FPR disparity
    """
    if protected_col not in df.columns:
        return {}

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    results: Dict = {"attribute": protected_col, "groups": {}, "tpr_disparity": 0.0, "fpr_disparity": 0.0}

    group_tprs = {}
    group_fprs = {}

    for group_val in sorted(df[protected_col].dropna().unique()):
        mask = (df[protected_col] == group_val).values
        if mask.sum() < 10:
            continue

        gt = y_true_arr[mask]
        gp = y_pred_arr[mask]
        label = _get_group_label(protected_col, group_val)

        tp = ((gt == 1) & (gp == 1)).sum()
        fp = ((gt == 0) & (gp == 1)).sum()
        tn = ((gt == 0) & (gp == 0)).sum()
        fn = ((gt == 1) & (gp == 0)).sum()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tnr = tn / max(tn + fp, 1)

        results["groups"][label] = {
            "tpr": round(float(tpr), 4),
            "fpr": round(float(fpr), 4),
            "tnr": round(float(tnr), 4),
            "n": int(mask.sum()),
            "code": group_val,
        }
        group_tprs[label] = tpr
        group_fprs[label] = fpr

    if len(group_tprs) >= 2:
        tpr_vals = list(group_tprs.values())
        fpr_vals = list(group_fprs.values())
        results["tpr_disparity"] = round(max(tpr_vals) - min(tpr_vals), 4)
        results["fpr_disparity"] = round(max(fpr_vals) - min(fpr_vals), 4)

    return results


def compute_disparate_impact_ratio(
    y_pred: np.ndarray,
    protected_col: str,
    df: pd.DataFrame,
    reference_group: Optional[Any] = None,
) -> Dict:
    """
    Computes disparate impact ratio (80% / 4/5ths rule per ECOA/EEOC guidelines).

    Args:
        y_pred: Binary predictions
        protected_col: Protected attribute column name
        df: DataFrame with protected attributes
        reference_group: Reference group code (auto-selects highest approval rate if None)

    Returns:
        Dict with DIR per group, pass/fail at 80% threshold
    """
    if protected_col not in df.columns:
        return {}

    y_pred_arr = np.asarray(y_pred)
    results: Dict = {
        "attribute": protected_col,
        "groups": {},
        "reference_group": None,
        "reference_approval_rate": 0.0,
        "ecoa_80_rule_overall": "PASS",
    }

    group_rates: Dict = {}
    for group_val in sorted(df[protected_col].dropna().unique()):
        mask = df[protected_col] == group_val
        if mask.sum() < 10:
            continue
        rate = float(y_pred_arr[mask.values].mean())
        label = _get_group_label(protected_col, group_val)
        group_rates[label] = {"rate": rate, "code": group_val, "n": int(mask.sum())}

    if not group_rates:
        return results

    # Reference group = highest approval rate (typically majority group)
    if reference_group is not None:
        ref_label = _get_group_label(protected_col, reference_group)
        ref_rate = group_rates.get(ref_label, {}).get("rate", max(v["rate"] for v in group_rates.values()))
    else:
        ref_label = max(group_rates, key=lambda k: group_rates[k]["rate"])
        ref_rate = group_rates[ref_label]["rate"]

    results["reference_group"] = ref_label
    results["reference_approval_rate"] = round(ref_rate, 4)

    any_fail = False
    for label, info in group_rates.items():
        if ref_rate > 0:
            dir_val = info["rate"] / ref_rate
        else:
            dir_val = 1.0
        passes = dir_val >= 0.8
        if not passes:
            any_fail = True
        results["groups"][label] = {
            "approval_rate": round(info["rate"], 4),
            "disparate_impact_ratio": round(dir_val, 4),
            "passes_80_rule": passes,
            "n": info["n"],
            "code": info["code"],
        }

    results["ecoa_80_rule_overall"] = "FAIL" if any_fail else "PASS"
    return results


def compute_statistical_significance(
    group_a_outcomes: np.ndarray,
    group_b_outcomes: np.ndarray,
) -> Dict:
    """
    Tests statistical significance of approval rate difference using chi-square test.

    Args:
        group_a_outcomes: Binary approval outcomes for group A
        group_b_outcomes: Binary approval outcomes for group B

    Returns:
        Dict with chi2 statistic, p-value, effect size (Cohen's h), and significance flag
    """
    a = np.asarray(group_a_outcomes)
    b = np.asarray(group_b_outcomes)

    n_a, n_b = len(a), len(b)
    a_approved = a.sum()
    b_approved = b.sum()

    contingency_table = np.array(
        [[a_approved, n_a - a_approved], [b_approved, n_b - b_approved]]
    )

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table, correction=True)

    # Cohen's h for proportions
    p_a = a_approved / max(n_a, 1)
    p_b = b_approved / max(n_b, 1)
    cohens_h = abs(2 * np.arcsin(np.sqrt(p_a)) - 2 * np.arcsin(np.sqrt(p_b)))

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        "approval_rate_a": round(float(p_a), 4),
        "approval_rate_b": round(float(p_b), 4),
        "rate_difference": round(float(p_a - p_b), 4),
        "cohens_h": round(float(cohens_h), 4),
        "statistically_significant": bool(p_value < 0.05),
        "n_a": n_a,
        "n_b": n_b,
    }


def run_ecoa_compliance_test(
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray],
    df: pd.DataFrame,
) -> Dict:
    """
    Runs full ECOA compliance tests across all protected attributes.

    Args:
        y_pred: Binary model predictions
        y_true: True labels (needed for equalized odds; can be None)
        df: DataFrame with protected attribute columns

    Returns:
        Dict with compliance results per attribute and overall status
    """
    results: Dict = {
        "overall_status": "PASS",
        "attributes_tested": [],
        "compliance_summary": {},
        "failed_attributes": [],
    }

    protected_cols = [c for c in PROTECTED_ATTRIBUTES.keys() if c in df.columns]
    if "applicant_age_group" in df.columns:
        protected_cols.append("applicant_age_group")

    for col in protected_cols:
        attr_result: Dict = {}

        # Demographic parity
        dp = compute_demographic_parity_difference(y_pred, col, df)
        attr_result["demographic_parity"] = dp

        # Disparate impact ratio
        di = compute_disparate_impact_ratio(y_pred, col, df)
        attr_result["disparate_impact"] = di
        passes_di = di.get("ecoa_80_rule_overall", "PASS") == "PASS"

        # Equalized odds (if y_true available)
        if y_true is not None:
            eo = compute_equalized_odds(y_true, y_pred, col, df)
            attr_result["equalized_odds"] = eo
            tpr_disp = eo.get("tpr_disparity", 0)
            passes_eo = tpr_disp < 0.1
        else:
            passes_eo = True

        # Statistical significance test: majority vs most-affected minority
        sig_result = _run_significance_for_attribute(y_pred, col, df)
        attr_result["statistical_significance"] = sig_result

        attribute_passes = passes_di and passes_eo
        attr_result["passes_ecoa"] = attribute_passes
        attr_result["attribute"] = col

        results["compliance_summary"][col] = attr_result
        results["attributes_tested"].append(col)
        if not attribute_passes:
            results["failed_attributes"].append(col)
            results["overall_status"] = "FAIL"

    logger.info(
        f"ECOA compliance: {results['overall_status']} "
        f"({len(results['failed_attributes'])} attributes failed)"
    )
    return results


def generate_adverse_action_codes(
    shap_explanation: Dict,
    top_n: int = 4,
) -> List[Dict]:
    """
    Generates ECOA-required adverse action reason codes from SHAP values.

    Args:
        shap_explanation: Output from explainability.explain_single_application()
        top_n: Number of adverse action reasons to return

    Returns:
        List of dicts with code, description, feature, and shap_value
    """
    denial_factors = shap_explanation.get("top_denial_factors", [])
    adverse_codes = []
    seen_codes = set()

    for factor in denial_factors:
        feature_name = factor.get("feature", "").lower()
        shap_val = factor.get("shap_value", 0)

        # Map feature to adverse action code
        code = None
        for key, val in _FEATURE_TO_CODE.items():
            if key in feature_name:
                code = val
                break

        if code is None:
            code = 9  # Other

        if code not in seen_codes:
            seen_codes.add(code)
            adverse_codes.append(
                {
                    "code": code,
                    "description": ADVERSE_ACTION_CODES.get(code, "Other"),
                    "feature": factor.get("feature", "unknown"),
                    "shap_contribution": round(float(shap_val), 4),
                }
            )
        if len(adverse_codes) >= top_n:
            break

    # Always return at least one code
    if not adverse_codes:
        adverse_codes.append(
            {"code": 9, "description": "Other", "feature": "model_score", "shap_contribution": 0.0}
        )

    return adverse_codes


def generate_fairness_report(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_true: np.ndarray,
) -> Dict:
    """
    Generates comprehensive fairness report for the entire portfolio.

    Args:
        df: DataFrame with protected attribute columns
        y_pred: Binary model predictions
        y_proba: Predicted probabilities
        y_true: True labels

    Returns:
        Comprehensive fairness report dict
    """
    report: Dict = {
        "portfolio_stats": {
            "total_applications": len(y_true),
            "overall_approval_rate": round(float(y_pred.mean()), 4),
            "model_approval_rate": round(float(y_pred.mean()), 4),
            "actual_approval_rate": round(float(np.asarray(y_true).mean()), 4),
        },
        "ecoa_compliance": run_ecoa_compliance_test(y_pred, y_true, df),
        "demographic_parity": {},
        "disparate_impact": {},
        "equalized_odds": {},
        "statistical_tests": {},
    }

    protected_cols = [c for c in list(PROTECTED_ATTRIBUTES.keys()) + ["applicant_age_group"] if c in df.columns]

    for col in protected_cols:
        report["demographic_parity"][col] = compute_demographic_parity_difference(y_pred, col, df)
        report["disparate_impact"][col] = compute_disparate_impact_ratio(y_pred, col, df)
        report["equalized_odds"][col] = compute_equalized_odds(y_true, y_pred, col, df)

        # Statistical test between most common groups
        sig_result = _run_significance_for_attribute(y_pred, col, df)
        report["statistical_tests"][col] = sig_result

    # Score distribution by race
    if "applicant_race_1" in df.columns:
        race_labels = {1: "AIAN", 2: "Asian", 3: "Black", 4: "NHPI", 5: "White"}
        proba_by_race = {}
        for code, label in race_labels.items():
            mask = df["applicant_race_1"] == code
            if mask.sum() >= 10:
                proba_by_race[label] = {
                    "mean_approval_proba": round(float(y_proba[mask.values].mean()), 4),
                    "median_approval_proba": round(float(np.median(y_proba[mask.values])), 4),
                    "n": int(mask.sum()),
                }
        report["score_distribution_by_race"] = proba_by_race

    logger.info("Fairness report generated")
    return report


# ── Helper functions ───────────────────────────────────────────────────────────

def _get_group_label(attribute: str, code: Any) -> str:
    """Returns human-readable label for a demographic group code."""
    mapping = PROTECTED_ATTRIBUTES.get(attribute, {})
    if mapping and code in mapping:
        return mapping[code]
    return str(code)


def _run_significance_for_attribute(
    y_pred: np.ndarray,
    protected_col: str,
    df: pd.DataFrame,
) -> Dict:
    """Runs chi-square test between majority group and most-affected minority."""
    y_pred_arr = np.asarray(y_pred)
    group_rates = {}
    group_outcomes = {}

    for group_val in sorted(df[protected_col].dropna().unique()):
        mask = (df[protected_col] == group_val).values
        if mask.sum() >= 30:
            rate = y_pred_arr[mask].mean()
            group_rates[group_val] = rate
            group_outcomes[group_val] = y_pred_arr[mask]

    if len(group_rates) < 2:
        return {"insufficient_data": True}

    ref_group = max(group_rates, key=lambda k: group_rates[k])
    min_group = min(group_rates, key=lambda k: group_rates[k])

    if ref_group == min_group:
        return {"no_disparity": True}

    return compute_statistical_significance(group_outcomes[ref_group], group_outcomes[min_group])

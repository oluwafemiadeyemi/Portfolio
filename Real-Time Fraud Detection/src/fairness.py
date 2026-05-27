"""
Fairness & Bias Monitoring Module
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Implements demographic parity, equal opportunity, and disparate impact metrics
for ECOA (Equal Credit Opportunity Act) compliance.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Demographic Parity
# ─────────────────────────────────────────────

def compute_demographic_parity(
    y_pred: np.ndarray | pd.Series,
    protected_col: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute the approval (non-fraud-flag) rate for each group in a protected attribute.

    Demographic Parity: P(Y_hat=1 | A=a) should be equal across groups.
    In fraud detection, 'approval' = predicted legitimate (y_pred = 0).

    Args:
        y_pred: Binary predictions array (0=legitimate, 1=fraud).
        protected_col: Column name of protected attribute in df.
        df: DataFrame containing protected_col.

    Returns:
        Dict with keys:
          - group_rates: {group_value: approval_rate}
          - max_gap: max difference in approval rates between groups
          - group_counts: {group_value: count}
    """
    if protected_col not in df.columns:
        logger.warning(f"Protected column '{protected_col}' not in DataFrame — skipping")
        return {"group_rates": {}, "max_gap": None, "group_counts": {}}

    y_pred_arr = np.asarray(y_pred)
    groups = df[protected_col].fillna("unknown").astype(str)

    group_rates: Dict[str, float] = {}
    group_counts: Dict[str, int] = {}

    for group in groups.unique():
        mask = groups == group
        n_group = int(mask.sum())
        if n_group == 0:
            continue
        # Approval rate = fraction predicted legitimate (0)
        approval_rate = float(np.mean(y_pred_arr[mask.values] == 0))
        group_rates[group] = round(approval_rate, 4)
        group_counts[group] = n_group

    rates = list(group_rates.values())
    max_gap = float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0

    logger.info(
        f"Demographic Parity [{protected_col}] | "
        f"groups={list(group_rates.keys())} | max_gap={max_gap:.4f}"
    )
    return {
        "metric": "demographic_parity",
        "protected_col": protected_col,
        "group_rates": group_rates,
        "group_counts": group_counts,
        "max_gap": round(max_gap, 4),
    }


# ─────────────────────────────────────────────
# Equal Opportunity
# ─────────────────────────────────────────────

def compute_equal_opportunity(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    protected_col: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute True Positive Rate (TPR / Recall) per protected group.

    Equal Opportunity: P(Y_hat=1 | Y=1, A=a) should be equal across groups.
    In fraud detection context, TPR = fraction of true frauds correctly flagged.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        protected_col: Protected attribute column name.
        df: DataFrame containing protected_col.

    Returns:
        Dict with group TPR values and max TPR gap.
    """
    if protected_col not in df.columns:
        logger.warning(f"Protected column '{protected_col}' not in DataFrame — skipping")
        return {"group_tpr": {}, "max_gap": None}

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    groups = df[protected_col].fillna("unknown").astype(str)

    group_tpr: Dict[str, float] = {}
    group_counts_positive: Dict[str, int] = {}

    for group in groups.unique():
        mask = (groups == group).values
        y_true_group = y_true_arr[mask]
        y_pred_group = y_pred_arr[mask]

        # Only compute TPR on actual positives
        positive_mask = y_true_group == 1
        n_positive = int(positive_mask.sum())

        if n_positive == 0:
            logger.debug(f"Group '{group}' has no positive samples — TPR undefined")
            group_tpr[group] = float("nan")
            group_counts_positive[group] = 0
            continue

        tpr = float(np.mean(y_pred_group[positive_mask] == 1))
        group_tpr[group] = round(tpr, 4)
        group_counts_positive[group] = n_positive

    valid_tprs = [v for v in group_tpr.values() if not np.isnan(v)]
    max_gap = float(max(valid_tprs) - min(valid_tprs)) if len(valid_tprs) >= 2 else 0.0

    logger.info(
        f"Equal Opportunity [{protected_col}] | "
        f"group_tprs={group_tpr} | max_gap={max_gap:.4f}"
    )
    return {
        "metric": "equal_opportunity",
        "protected_col": protected_col,
        "group_tpr": group_tpr,
        "group_counts_positive": group_counts_positive,
        "max_gap": round(max_gap, 4),
    }


# ─────────────────────────────────────────────
# Disparate Impact (80% Rule)
# ─────────────────────────────────────────────

def compute_disparate_impact(
    y_pred: np.ndarray | pd.Series,
    protected_col: str,
    df: pd.DataFrame,
    favorable_outcome: int = 0,
) -> Dict[str, Any]:
    """
    Compute disparate impact ratio and apply the EEOC 80% rule.

    Disparate Impact = (approval rate of protected group) / (approval rate of reference group)
    Pass threshold: DI >= 0.80 (80% rule from EEOC / ECOA guidelines).

    The reference group is the group with the highest approval rate.
    The protected group is the group with the lowest approval rate.

    Args:
        y_pred: Predicted labels.
        protected_col: Protected attribute column name.
        df: DataFrame containing protected_col.
        favorable_outcome: Label value considered favorable (default 0 = not flagged).

    Returns:
        Dict with disparate_impact_ratio, pass_fail, group_rates.
    """
    if protected_col not in df.columns:
        logger.warning(f"Protected column '{protected_col}' not in DataFrame — skipping")
        return {"disparate_impact_ratio": None, "passes_80_rule": None}

    y_pred_arr = np.asarray(y_pred)
    groups = df[protected_col].fillna("unknown").astype(str)

    group_rates: Dict[str, float] = {}
    for group in groups.unique():
        mask = (groups == group).values
        if mask.sum() == 0:
            continue
        rate = float(np.mean(y_pred_arr[mask] == favorable_outcome))
        group_rates[group] = round(rate, 4)

    if len(group_rates) < 2:
        logger.warning(f"Need ≥2 groups for disparate impact; found {len(group_rates)}")
        return {
            "metric": "disparate_impact",
            "protected_col": protected_col,
            "group_rates": group_rates,
            "disparate_impact_ratio": None,
            "passes_80_rule": None,
            "reference_group": None,
            "disadvantaged_group": None,
        }

    max_rate_group = max(group_rates, key=group_rates.get)
    min_rate_group = min(group_rates, key=group_rates.get)
    max_rate = group_rates[max_rate_group]
    min_rate = group_rates[min_rate_group]

    di_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 0.0
    passes_80_rule = bool(di_ratio >= 0.80)

    logger.info(
        f"Disparate Impact [{protected_col}] | "
        f"ratio={di_ratio:.4f} | passes_80_rule={passes_80_rule} | "
        f"reference={max_rate_group}({max_rate:.3f}) | "
        f"disadvantaged={min_rate_group}({min_rate:.3f})"
    )
    return {
        "metric": "disparate_impact",
        "protected_col": protected_col,
        "group_rates": group_rates,
        "disparate_impact_ratio": di_ratio,
        "passes_80_rule": passes_80_rule,
        "threshold": 0.80,
        "reference_group": max_rate_group,
        "reference_rate": max_rate,
        "disadvantaged_group": min_rate_group,
        "disadvantaged_rate": min_rate,
    }


# ─────────────────────────────────────────────
# Full Fairness Report
# ─────────────────────────────────────────────

def generate_fairness_report(
    df: pd.DataFrame,
    y_pred: np.ndarray | pd.Series,
    y_proba: Optional[np.ndarray | pd.Series] = None,
    protected_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive fairness report across protected attributes.

    Runs demographic parity, equal opportunity, and disparate impact for
    each protected column found in the DataFrame.

    Args:
        df: DataFrame with protected attribute columns and 'isFraud' target.
        y_pred: Binary predictions.
        y_proba: Fraud probabilities (optional, stored for reference).
        protected_cols: List of protected attribute column names. If None,
            defaults to common candidate columns found in df.

    Returns:
        Dict containing all fairness metrics and a summary pass/fail table.
    """
    y_pred_arr = np.asarray(y_pred)
    y_true_arr = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))

    # Discover available protected columns
    if protected_cols is None:
        candidate_cols = [
            "card4", "card4_enc", "card6", "card6_enc",
            "P_emaildomain", "P_emaildomain_enc",
            "addr2",  # Country code proxy
        ]
        protected_cols = [c for c in candidate_cols if c in df.columns]

    if not protected_cols:
        logger.warning("No protected attribute columns found in DataFrame")
        return {
            "status": "no_protected_cols_found",
            "demographic_parity": {},
            "equal_opportunity": {},
            "disparate_impact": {},
            "summary": [],
        }

    logger.info(f"Generating fairness report for columns: {protected_cols}")

    dp_results: Dict[str, Any] = {}
    eo_results: Dict[str, Any] = {}
    di_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for col in protected_cols:
        dp = compute_demographic_parity(y_pred_arr, col, df)
        eo = compute_equal_opportunity(y_true_arr, y_pred_arr, col, df)
        di = compute_disparate_impact(y_pred_arr, col, df)

        dp_results[col] = dp
        eo_results[col] = eo
        di_results[col] = di

        summary_rows.append(
            {
                "protected_attribute": col,
                "demographic_parity_max_gap": dp.get("max_gap"),
                "equal_opportunity_max_gap": eo.get("max_gap"),
                "disparate_impact_ratio": di.get("disparate_impact_ratio"),
                "passes_80_rule": di.get("passes_80_rule"),
                "n_groups": len(dp.get("group_rates", {})),
            }
        )

    # Overall pass/fail
    all_di_pass = all(
        row["passes_80_rule"] for row in summary_rows if row["passes_80_rule"] is not None
    )

    report: Dict[str, Any] = {
        "protected_cols_analyzed": protected_cols,
        "demographic_parity": dp_results,
        "equal_opportunity": eo_results,
        "disparate_impact": di_results,
        "summary": summary_rows,
        "overall_passes_80_rule": all_di_pass,
        "n_predictions": len(y_pred_arr),
        "fraud_rate_in_predictions": float(np.mean(y_pred_arr == 1)),
    }

    logger.info(
        f"Fairness report complete | "
        f"overall_passes_80_rule={all_di_pass} | "
        f"cols_analyzed={len(protected_cols)}"
    )
    return report

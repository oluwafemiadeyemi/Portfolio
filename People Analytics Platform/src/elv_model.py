"""
Employee Lifetime Value (ELV) Model
Computes ELV, replacement costs, employee segmentation, and retention ROI.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def compute_replacement_cost(df: pd.DataFrame) -> pd.Series:
    """
    Computes employee replacement cost using the SHRM formula.
    Formula: 1.5 × MonthlyIncome × 12 (annual salary)
    Source: SHRM (Society for Human Resource Management) — average cost to replace an employee.

    For specialized roles (managers, directors), cost is 2x-3x annual salary.

    Args:
        df: DataFrame with MonthlyIncome and JobRole columns

    Returns:
        Series with replacement cost per employee
    """
    monthly = df.get("MonthlyIncome", pd.Series(5000, index=df.index)).fillna(5000)
    annual = monthly * 12

    # Role-specific multipliers
    high_cost_roles = {
        "Manager": 2.0,
        "Research Director": 2.5,
        "Manufacturing Director": 2.5,
        "Sales Executive": 1.8,
        "Healthcare Representative": 1.6,
    }

    if "JobRole" in df.columns:
        multiplier = df["JobRole"].map(high_cost_roles).fillna(1.5)
    else:
        job_level = df.get("JobLevel", pd.Series(2, index=df.index)).fillna(2)
        multiplier = np.where(job_level >= 4, 2.0, np.where(job_level == 3, 1.75, 1.5))

    return (annual * multiplier).round(2)


def compute_elv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Employee Lifetime Value (ELV) for all employees.

    ELV = annual_value × expected_tenure_years - replacement_cost_if_leaves

    Components:
      - annual_value: annual contribution = MonthlyIncome × 12 × productivity_factor
      - expected_tenure: estimated years remaining (age-adjusted + flight risk discount)
      - replacement_cost: SHRM-based replacement cost

    Args:
        df: DataFrame with MonthlyIncome, Age, JobLevel, and engineered features

    Returns:
        DataFrame with ELV columns added
    """
    df = df.copy()

    monthly = df.get("MonthlyIncome", pd.Series(5000, index=df.index)).fillna(5000)
    age = df.get("Age", pd.Series(37, index=df.index)).fillna(37)
    perf = df.get("PerformanceRating", pd.Series(3, index=df.index)).fillna(3)
    involvement = df.get("JobInvolvement", pd.Series(3, index=df.index)).fillna(3)
    flight_risk = df.get("flight_risk_score", pd.Series(30.0, index=df.index)).fillna(30.0)

    # Productivity: performance + involvement
    productivity = (0.6 * (perf / 4) + 0.4 * (involvement / 4)).clip(0.5, 1.2)
    annual_value = (monthly * 12 * productivity).round(2)

    # Expected tenure: up to retirement (65), discounted by flight risk
    max_remaining = (65 - age).clip(1, 30)
    tenure_discount = 1 - (flight_risk / 100) * 0.65
    expected_tenure = (max_remaining * tenure_discount).clip(0.5, 25).round(2)

    replacement_cost = compute_replacement_cost(df)

    # ELV = total future value minus replacement cost if employee leaves
    elv = (annual_value * expected_tenure - replacement_cost).round(2)

    df["annual_value"] = annual_value
    df["expected_tenure_years"] = expected_tenure
    df["replacement_cost"] = replacement_cost
    df["elv"] = elv
    df["elv_positive"] = (elv > 0).astype(int)

    logger.info(
        f"ELV computed: median={elv.median():,.0f}, "
        f"mean={elv.mean():,.0f}, "
        f"pct_positive={df['elv_positive'].mean():.1%}"
    )
    return df


def segment_employees(
    df: pd.DataFrame,
    attrition_proba: np.ndarray,
    flight_risk_threshold: float = 50.0,
    elv_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Segments employees into 2×2 matrix: High/Low ELV × High/Low Flight Risk.

    Quadrants:
      - Protect: High ELV + High Flight Risk → Critical retention intervention
      - Develop: Low ELV + High Flight Risk → Coaching/engagement
      - Monitor: High ELV + Low Flight Risk → Maintain engagement, watch
      - Maintain: Low ELV + Low Flight Risk → Stable, no urgent action

    Args:
        df: DataFrame with ELV columns
        attrition_proba: Predicted attrition probabilities
        flight_risk_threshold: Score above which employee is "high flight risk"
        elv_threshold: ELV above which employee is "high ELV" (default: median)

    Returns:
        DataFrame with 'segment' column added
    """
    df = df.copy()
    df["attrition_probability"] = np.round(attrition_proba[:len(df)], 4)

    if "elv" not in df.columns:
        df = compute_elv(df)

    if elv_threshold is None:
        elv_threshold = float(df["elv"].median())

    flight_risk = df.get("flight_risk_score", pd.Series(30.0, index=df.index)).fillna(30.0)
    high_flight_risk = flight_risk >= flight_risk_threshold
    high_elv = df["elv"] >= elv_threshold

    conditions = [
        high_elv & high_flight_risk,
        ~high_elv & high_flight_risk,
        high_elv & ~high_flight_risk,
        ~high_elv & ~high_flight_risk,
    ]
    choices = ["Protect", "Develop", "Monitor", "Maintain"]
    df["segment"] = np.select(conditions, choices, default="Monitor")

    segment_counts = df["segment"].value_counts()
    logger.info(f"Employee segments: {segment_counts.to_dict()}")
    return df


def compute_retention_roi(
    df: pd.DataFrame,
    attrition_proba: np.ndarray,
    intervention_cost_per_employee: float = 5000.0,
    intervention_effectiveness: float = 0.35,
) -> Dict:
    """
    Computes expected ROI from retention interventions.

    ROI = (Expected_Savings_from_Prevention - Intervention_Cost) / Intervention_Cost

    Expected savings = Σ [P(attrition) × replacement_cost × effectiveness]

    Args:
        df: DataFrame with ELV and replacement cost columns
        attrition_proba: Predicted attrition probabilities
        intervention_cost_per_employee: Cost of retention program per employee ($)
        intervention_effectiveness: Probability that intervention prevents attrition

    Returns:
        Dict with ROI metrics and recommendation
    """
    if "replacement_cost" not in df.columns:
        df = compute_elv(df)

    n_employees = len(df)
    probas = np.asarray(attrition_proba[:n_employees])
    replacement_costs = df["replacement_cost"].values[:n_employees]

    # Expected number of attritions
    expected_attritions = probas.sum()

    # High-risk employees (top 20% by probability)
    n_high_risk = max(int(n_employees * 0.20), 1)
    top_risk_idx = np.argsort(probas)[::-1][:n_high_risk]

    # Costs of interventions on high-risk employees
    total_intervention_cost = n_high_risk * intervention_cost_per_employee

    # Expected attritions prevented
    high_risk_probas = probas[top_risk_idx]
    attritions_prevented = (high_risk_probas * intervention_effectiveness).sum()

    # Expected savings from prevented replacements
    high_risk_replacement_costs = replacement_costs[top_risk_idx]
    expected_savings = (high_risk_probas * intervention_effectiveness * high_risk_replacement_costs).sum()

    # Net benefit
    net_benefit = expected_savings - total_intervention_cost
    roi = net_benefit / max(total_intervention_cost, 1) * 100

    # Annual attrition cost if no intervention
    annual_attrition_cost = (probas * replacement_costs).sum()

    result = {
        "n_employees": n_employees,
        "expected_attritions": round(float(expected_attritions), 1),
        "expected_attrition_rate": round(float(expected_attritions / n_employees), 4),
        "n_targeted_for_intervention": n_high_risk,
        "intervention_cost_per_employee": intervention_cost_per_employee,
        "total_intervention_cost": round(float(total_intervention_cost), 2),
        "intervention_effectiveness": intervention_effectiveness,
        "attritions_prevented": round(float(attritions_prevented), 1),
        "expected_savings": round(float(expected_savings), 2),
        "net_benefit": round(float(net_benefit), 2),
        "roi_pct": round(float(roi), 1),
        "annual_cost_without_intervention": round(float(annual_attrition_cost), 2),
        "payback_months": round(float(total_intervention_cost / max(expected_savings / 12, 1)), 1),
        "recommendation": (
            "Proceed with retention program — positive ROI expected."
            if roi > 0
            else "Refine targeting or reduce intervention cost — current ROI is negative."
        ),
    }

    logger.info(
        f"Retention ROI: {roi:.1f}% | "
        f"Savings: ${expected_savings:,.0f} | "
        f"Cost: ${total_intervention_cost:,.0f}"
    )
    return result

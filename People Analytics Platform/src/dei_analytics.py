"""
DEI Analytics Module
Attrition disparities, pay equity gap analysis, promotion velocity,
and comprehensive DEI scorecard for HR compliance.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def compute_attrition_by_group(
    df: pd.DataFrame,
    attrition_col: str = "Attrition",
    group_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Computes attrition rates by demographic group.

    Args:
        df: Employee DataFrame
        attrition_col: Column with binary attrition indicator
        group_cols: List of demographic grouping columns

    Returns:
        Dict of {group_col: DataFrame with attrition stats}
    """
    if group_cols is None:
        group_cols = ["Gender", "Age_group", "MaritalStatus", "Department", "JobRole"]

    group_cols = [c for c in group_cols if c in df.columns]
    results = {}

    for col in group_cols:
        agg = (
            df.groupby(col)[attrition_col]
            .agg(["mean", "count", "sum"])
            .rename(columns={"mean": "attrition_rate", "count": "n_employees", "sum": "n_attrited"})
            .reset_index()
        )
        agg["attrition_rate"] = agg["attrition_rate"].round(4)
        agg["retention_rate"] = (1 - agg["attrition_rate"]).round(4)
        agg = agg.sort_values("attrition_rate", ascending=False)
        results[col] = agg
        logger.debug(f"Attrition by {col}: computed for {len(agg)} groups")

    return results


def compute_pay_equity_gap(
    df: pd.DataFrame,
    salary_col: str = "MonthlyIncome",
    group_col: str = "Gender",
    control_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Computes unexplained pay gap using OLS regression controlling for role/level/seniority.

    The unexplained gap = coefficient on group indicator after controlling for
    job role, job level, education, and tenure.

    Args:
        df: Employee DataFrame
        salary_col: Column with monthly income
        group_col: Protected attribute (e.g., 'Gender')
        control_cols: Confound controls (job role, level, tenure, etc.)

    Returns:
        Dict with unexplained gap, p-value, and interpretation
    """
    if control_cols is None:
        control_cols = ["JobLevel", "TotalWorkingYears", "Education", "PerformanceRating", "YearsAtCompany"]

    available_controls = [c for c in control_cols if c in df.columns]

    if salary_col not in df.columns or group_col not in df.columns:
        return {"error": f"Required columns missing: {salary_col}, {group_col}"}

    df_clean = df[[salary_col, group_col] + available_controls].dropna()
    if len(df_clean) < 30:
        return {"error": "Insufficient data for pay equity analysis"}

    # Encode protected group as binary (simplest: Male=0, Female=1 for gender)
    unique_vals = df_clean[group_col].unique()
    if len(unique_vals) == 2:
        group_binary = (df_clean[group_col] == unique_vals[1]).astype(float)
        reference_group = str(unique_vals[0])
        comparison_group = str(unique_vals[1])
    else:
        # For multi-category, compare first vs all others
        group_binary = (df_clean[group_col] != unique_vals[0]).astype(float)
        reference_group = str(unique_vals[0])
        comparison_group = "all_others"

    # OLS: log(salary) ~ group + controls
    from sklearn.linear_model import LinearRegression

    log_salary = np.log(df_clean[salary_col].clip(1))
    X_control = pd.get_dummies(df_clean[available_controls], drop_first=True, dtype=float)
    X_control["group"] = group_binary.values

    lr = LinearRegression()
    lr.fit(X_control, log_salary)

    # Extract group coefficient
    group_coef = lr.coef_[X_control.columns.get_loc("group")]
    pct_gap = (np.exp(group_coef) - 1) * 100  # Convert log-coefficient to percentage

    # Statistical significance via t-test on salary by group
    group_a = df_clean.loc[df_clean[group_col] == unique_vals[0], salary_col].values
    group_b = df_clean.loc[df_clean[group_col] != unique_vals[0], salary_col].values
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    # Raw gap (before controlling)
    raw_gap_pct = (df_clean.groupby(group_col)[salary_col].mean().pct_change().iloc[-1] * 100
                   if len(unique_vals) == 2 else 0)

    result = {
        "attribute": group_col,
        "reference_group": reference_group,
        "comparison_group": comparison_group,
        "raw_pay_gap_pct": round(float(raw_gap_pct), 2),
        "unexplained_gap_pct": round(float(pct_gap), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "statistically_significant": bool(p_value < 0.05),
        "controls_used": available_controls,
        "n_employees": len(df_clean),
        "median_salary_by_group": {
            str(k): round(float(v), 2)
            for k, v in df_clean.groupby(group_col)[salary_col].median().items()
        },
        "interpretation": (
            f"{comparison_group} earns {abs(pct_gap):.1f}% {'less' if pct_gap < 0 else 'more'} "
            f"than {reference_group} after controlling for role and seniority."
        ),
    }
    logger.info(f"Pay equity gap ({group_col}): {pct_gap:.1f}% unexplained, p={p_value:.4f}")
    return result


def compute_promotion_velocity(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Computes average years to promotion by demographic group.
    Uses YearsSinceLastPromotion and YearsAtCompany as proxies.

    Args:
        df: Employee DataFrame
        group_cols: Demographic grouping columns

    Returns:
        Dict of {group_col: DataFrame with promotion velocity stats}
    """
    if group_cols is None:
        group_cols = ["Gender", "Age_group", "MaritalStatus"]

    group_cols = [c for c in group_cols if c in df.columns]
    results = {}

    promo_col = "YearsSinceLastPromotion" if "YearsSinceLastPromotion" in df.columns else None

    if promo_col is None:
        return {"error": "YearsSinceLastPromotion column not found"}

    for col in group_cols:
        agg = (
            df.groupby(col)
            .agg(
                avg_years_since_promotion=(promo_col, "mean"),
                median_years_since_promotion=(promo_col, "median"),
                n_employees=(promo_col, "count"),
            )
            .round(2)
            .reset_index()
        )

        # Add career velocity if available
        if "career_velocity" in df.columns:
            vel_agg = df.groupby(col)["career_velocity"].mean().round(3)
            agg = agg.merge(vel_agg.rename("avg_career_velocity"), on=col, how="left")

        agg = agg.sort_values("avg_years_since_promotion", ascending=False)
        results[col] = agg
        logger.debug(f"Promotion velocity by {col}: computed for {len(agg)} groups")

    return results


def statistical_significance_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    test_type: str = "auto",
) -> Dict:
    """
    Runs appropriate statistical test between two groups.

    Args:
        group_a: Values for group A
        group_b: Values for group B
        test_type: 'ttest', 'chi2', or 'auto' (auto-selects based on data)

    Returns:
        Dict with test name, statistic, p-value, and significance
    """
    a = np.asarray(group_a).flatten()
    b = np.asarray(group_b).flatten()

    # Auto-detect: binary data → chi-square, continuous → t-test
    if test_type == "auto":
        unique_a = set(np.unique(a[~np.isnan(a)]))
        test_type = "chi2" if unique_a <= {0, 1} else "ttest"

    if test_type == "chi2":
        n_a, n_b = len(a), len(b)
        a_pos = a.sum()
        b_pos = b.sum()
        contingency = np.array([[a_pos, n_a - a_pos], [b_pos, n_b - b_pos]])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency, correction=True)
        return {
            "test": "chi2",
            "statistic": round(float(chi2), 4),
            "p_value": round(float(p_val), 6),
            "statistically_significant": bool(p_val < 0.05),
            "rate_a": round(float(a.mean()), 4),
            "rate_b": round(float(b.mean()), 4),
            "rate_difference": round(float(a.mean() - b.mean()), 4),
            "n_a": n_a,
            "n_b": n_b,
        }
    else:
        t_stat, p_val = stats.ttest_ind(a[~np.isnan(a)], b[~np.isnan(b)], equal_var=False)
        cohens_d = (np.mean(a) - np.mean(b)) / max(
            np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2), 1e-9
        )
        return {
            "test": "welch_ttest",
            "statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "statistically_significant": bool(p_val < 0.05),
            "mean_a": round(float(np.mean(a)), 4),
            "mean_b": round(float(np.mean(b)), 4),
            "mean_difference": round(float(np.mean(a) - np.mean(b)), 4),
            "cohens_d": round(float(cohens_d), 4),
            "n_a": len(a),
            "n_b": len(b),
        }


def generate_dei_scorecard(df: pd.DataFrame) -> Dict:
    """
    Generates comprehensive DEI scorecard with metrics, pass/fail, and p-values.

    Args:
        df: Employee DataFrame with Attrition column

    Returns:
        Dict with all DEI metrics, scores, and compliance flags
    """
    scorecard: Dict = {
        "summary": {},
        "attrition_disparity": {},
        "pay_equity": {},
        "promotion_velocity": {},
        "overall_dei_score": None,
        "flags": [],
        "recommendations": [],
    }

    # ── Attrition disparity ───────────────────────────────────────────────────
    if "Attrition" in df.columns:
        attrition_by_group = compute_attrition_by_group(df)
        for attr, group_df in attrition_by_group.items():
            if len(group_df) < 2:
                continue
            max_rate = group_df["attrition_rate"].max()
            min_rate = group_df["attrition_rate"].min()
            disparity = max_rate - min_rate
            high_group = group_df.loc[group_df["attrition_rate"].idxmax(), attr]
            passes = disparity < 0.10  # Less than 10 pp gap

            # Statistical test
            unique_groups = group_df[attr].unique()
            if len(unique_groups) == 2 and "n_attrited" in group_df.columns:
                g1 = group_df.iloc[0]
                g2 = group_df.iloc[1]
                a = np.concatenate([np.ones(int(g1["n_attrited"])), np.zeros(int(g1["n_employees"] - g1["n_attrited"]))])
                b = np.concatenate([np.ones(int(g2["n_attrited"])), np.zeros(int(g2["n_employees"] - g2["n_attrited"]))])
                sig_test = statistical_significance_test(a, b, "chi2")
            else:
                sig_test = {"p_value": 1.0, "statistically_significant": False}

            scorecard["attrition_disparity"][attr] = {
                "passes": passes,
                "disparity": round(float(disparity), 4),
                "high_attrition_group": str(high_group),
                "max_attrition_rate": round(float(max_rate), 4),
                "min_attrition_rate": round(float(min_rate), 4),
                "statistical_test": sig_test,
                "group_breakdown": group_df.to_dict(orient="records"),
            }
            if not passes:
                scorecard["flags"].append(f"Attrition disparity in {attr}: {disparity:.1%} gap")

    # ── Pay equity ────────────────────────────────────────────────────────────
    for group_col in ["Gender", "MaritalStatus"]:
        if group_col in df.columns and "MonthlyIncome" in df.columns:
            gap = compute_pay_equity_gap(df, "MonthlyIncome", group_col)
            if "unexplained_gap_pct" in gap:
                passes = abs(gap["unexplained_gap_pct"]) < 5.0  # Less than 5% unexplained gap
                gap["passes"] = passes
                scorecard["pay_equity"][group_col] = gap
                if not passes:
                    scorecard["flags"].append(
                        f"Pay equity concern ({group_col}): {gap['unexplained_gap_pct']:.1f}% unexplained gap"
                    )

    # ── Promotion velocity ────────────────────────────────────────────────────
    promo_by_group = compute_promotion_velocity(df)
    for attr, promo_df in promo_by_group.items():
        if len(promo_df) < 2:
            continue
        max_wait = promo_df["avg_years_since_promotion"].max()
        min_wait = promo_df["avg_years_since_promotion"].min()
        disparity = max_wait - min_wait
        passes = disparity < 1.0  # Less than 1 year difference

        scorecard["promotion_velocity"][attr] = {
            "passes": passes,
            "disparity_years": round(float(disparity), 2),
            "max_wait_group": str(promo_df.loc[promo_df["avg_years_since_promotion"].idxmax(), attr]),
            "group_breakdown": promo_df.to_dict(orient="records"),
        }
        if not passes:
            scorecard["flags"].append(
                f"Promotion velocity disparity in {attr}: {disparity:.1f} year gap"
            )

    # ── Overall DEI score (0-100) ─────────────────────────────────────────────
    n_checks = (
        len(scorecard["attrition_disparity"])
        + len(scorecard["pay_equity"])
        + len(scorecard["promotion_velocity"])
    )
    n_passing = (
        sum(1 for v in scorecard["attrition_disparity"].values() if v.get("passes", True))
        + sum(1 for v in scorecard["pay_equity"].values() if v.get("passes", True))
        + sum(1 for v in scorecard["promotion_velocity"].values() if v.get("passes", True))
    )
    scorecard["overall_dei_score"] = round(n_passing / max(n_checks, 1) * 100, 1)
    scorecard["n_checks"] = n_checks
    scorecard["n_passing"] = n_passing
    scorecard["n_failing"] = n_checks - n_passing

    # Summary metrics
    if "Attrition" in df.columns:
        scorecard["summary"]["overall_attrition_rate"] = round(float(df["Attrition"].mean()), 4)
        scorecard["summary"]["n_employees"] = len(df)
    if "MonthlyIncome" in df.columns:
        scorecard["summary"]["median_monthly_income"] = round(float(df["MonthlyIncome"].median()), 2)

    # Recommendations
    if scorecard["flags"]:
        scorecard["recommendations"] = _generate_dei_recommendations(scorecard["flags"])

    logger.info(f"DEI Scorecard: {scorecard['overall_dei_score']}/100 ({n_passing}/{n_checks} checks passing)")
    return scorecard


def _generate_dei_recommendations(flags: List[str]) -> List[str]:
    """Generates actionable DEI recommendations from detected flags."""
    recommendations = []
    for flag in flags:
        if "Attrition disparity" in flag and "Gender" in flag:
            recommendations.append(
                "Conduct exit interview analysis by gender. Review workload distribution and flexible work policies."
            )
        elif "Attrition disparity" in flag and "Age" in flag:
            recommendations.append(
                "Review retention programs for different age cohorts. Consider mentorship programs for under-30 employees."
            )
        elif "Pay equity" in flag:
            recommendations.append(
                "Commission third-party pay equity audit. Establish transparent salary bands by role and level."
            )
        elif "Promotion velocity" in flag:
            recommendations.append(
                "Review promotion criteria and processes for consistency. Establish sponsorship programs for underrepresented groups."
            )
    return list(set(recommendations))  # Deduplicate

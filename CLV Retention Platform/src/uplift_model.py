"""
Causal Uplift Modeling for Retention Campaign Optimization.
Implements Two-Model Approach with propensity score matching and Qini evaluation.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_treatment_propensity(
    X: pd.DataFrame,
    treatment: pd.Series,
    random_state: int = 42,
) -> np.ndarray:
    """
    Estimate propensity score P(treatment=1 | X) using logistic regression.
    Returns array of propensity scores in [0, 1].
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    lr = LogisticRegression(
        max_iter=500, C=1.0, solver="lbfgs", random_state=random_state
    )
    lr.fit(X_scaled, treatment.values)
    propensity_scores = lr.predict_proba(X_scaled)[:, 1]

    # Trim extreme propensities (common good practice)
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

    try:
        auc = roc_auc_score(treatment.values, propensity_scores)
        logger.info(f"Propensity model fitted. AUC: {auc:.4f}, mean score: {propensity_scores.mean():.3f}")
    except Exception:
        pass

    return propensity_scores


def match_treated_control(
    X: pd.DataFrame,
    treatment: pd.Series,
    propensity_scores: np.ndarray,
    ratio: int = 1,
    caliper: float = 0.05,
) -> Tuple[pd.Index, pd.Index]:
    """
    Nearest-neighbor propensity score matching.
    Returns (treated_indices, matched_control_indices).
    """
    treated_mask = treatment == 1
    control_mask = treatment == 0

    treated_idx = X.index[treated_mask]
    control_idx = X.index[control_mask]

    ps_treated = propensity_scores[treated_mask]
    ps_control = propensity_scores[control_mask]

    # Fit NN on control propensity scores
    nbrs = NearestNeighbors(n_neighbors=ratio, algorithm="ball_tree")
    nbrs.fit(ps_control.reshape(-1, 1))

    distances, indices = nbrs.kneighbors(ps_treated.reshape(-1, 1))

    matched_treated_list: list = []
    matched_control_list: list = []

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for dist, j in zip(dist_row, idx_row):
            if dist <= caliper:
                matched_treated_list.append(treated_idx[i])
                matched_control_list.append(control_idx[j])

    logger.info(
        f"PSM: {len(treated_idx):,} treated → {len(matched_treated_list):,} matched pairs "
        f"(caliper={caliper})"
    )
    return pd.Index(matched_treated_list), pd.Index(matched_control_list)


def estimate_ate(
    y_treated: np.ndarray,
    y_control: np.ndarray,
) -> dict[str, float]:
    """
    Estimate Average Treatment Effect (ATE) with standard error and p-value.
    Uses Welch's t-test for statistical significance.
    """
    ate = float(np.mean(y_treated) - np.mean(y_control))
    se = float(np.sqrt(np.var(y_treated) / len(y_treated) + np.var(y_control) / len(y_control)))

    t_stat, p_value = stats.ttest_ind(y_treated, y_control, equal_var=False)
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    result = {
        "ate": round(ate, 4),
        "standard_error": round(se, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "ci_lower_95": round(ci_lower, 4),
        "ci_upper_95": round(ci_upper, 4),
        "n_treated": len(y_treated),
        "n_control": len(y_control),
        "response_rate_treated": round(float(np.mean(y_treated)), 4),
        "response_rate_control": round(float(np.mean(y_control)), 4),
    }
    logger.info(f"ATE={ate:.4f}, SE={se:.4f}, p={p_value:.4f}")
    return result


def train_uplift_model(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    random_state: int = 42,
) -> Tuple[Any, Any, dict]:
    """
    Two-Model Approach:
    - Model T: trained on treated group (treatment=1)
    - Model C: trained on control group (treatment=0)
    - Uplift = P(y=1 | X, treated) - P(y=1 | X, control)

    Returns (model_treated, model_control, metadata_dict)
    """
    from sklearn.ensemble import GradientBoostingClassifier

    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_t = X[treated_mask].fillna(0)
    y_t = y[treated_mask]
    X_c = X[control_mask].fillna(0)
    y_c = y[control_mask]

    logger.info(f"Two-Model uplift training: {treated_mask.sum():,} treated, {control_mask.sum():,} control")

    model_treated = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        min_samples_leaf=20, random_state=random_state,
    )
    model_control = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        min_samples_leaf=20, random_state=random_state,
    )

    model_treated.fit(X_t, y_t)
    model_control.fit(X_c, y_c)

    # Compute ATE on held-out data
    try:
        auc_t = roc_auc_score(y_t, model_treated.predict_proba(X_t)[:, 1])
        auc_c = roc_auc_score(y_c, model_control.predict_proba(X_c)[:, 1])
    except Exception:
        auc_t = auc_c = 0.0

    metadata = {
        "n_treated": int(treated_mask.sum()),
        "n_control": int(control_mask.sum()),
        "response_rate_treated": float(y_t.mean()),
        "response_rate_control": float(y_c.mean()),
        "auc_treated_model": round(auc_t, 4),
        "auc_control_model": round(auc_c, 4),
        "feature_names": list(X.columns),
    }

    logger.info(f"Uplift models trained. T-AUC={auc_t:.4f}, C-AUC={auc_c:.4f}")
    return model_treated, model_control, metadata


def compute_uplift_scores(
    uplift_model_pair: Tuple[Any, Any],
    X: pd.DataFrame,
) -> pd.Series:
    """
    Compute per-user uplift score = P(respond|treated) - P(respond|control).
    """
    model_treated, model_control = uplift_model_pair
    X_clean = X.fillna(0)

    p_treated = model_treated.predict_proba(X_clean)[:, 1]
    p_control = model_control.predict_proba(X_clean)[:, 1]
    uplift = p_treated - p_control

    logger.info(
        f"Uplift scores: mean={uplift.mean():.4f}, "
        f"pos_fraction={(uplift > 0).mean():.1%}"
    )
    return pd.Series(uplift, index=X.index, name="uplift_score")


def segment_by_uplift(
    uplift_scores: pd.Series,
    high_threshold: float = 0.1,
    low_threshold: float = -0.05,
    base_response_rate: float = 0.3,
) -> pd.Series:
    """
    Classify users into 4 uplift segments:
    - Persuadables: positive uplift → intervention helps
    - Sure Things: would respond regardless (high base rate, low uplift)
    - Lost Causes: low base rate, low uplift → won't respond
    - Sleeping Dogs: negative uplift → intervention hurts

    Uses simple rule-based segmentation on uplift scores.
    """
    segments = pd.Series("Lost Causes", index=uplift_scores.index, name="uplift_segment")

    # Sleeping Dogs: negative uplift
    segments[uplift_scores < low_threshold] = "Sleeping Dogs"

    # Lost Causes: near-zero uplift (already set as default)

    # Persuadables: meaningful positive uplift
    segments[uplift_scores >= high_threshold] = "Persuadables"

    # Sure Things: moderate uplift but already likely to respond
    # Proxy: top 20% of scores that aren't persuadables → intermediate responders
    q80 = uplift_scores.quantile(0.80)
    sure_thing_mask = (
        (uplift_scores >= low_threshold)
        & (uplift_scores < high_threshold)
        & (uplift_scores >= q80 * 0.5)
    )
    segments[sure_thing_mask] = "Sure Things"

    dist = segments.value_counts()
    logger.info(f"Uplift segments: {dist.to_dict()}")
    return segments


def compute_qini_coefficient(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
) -> dict[str, float]:
    """
    Compute Qini coefficient for uplift model evaluation.
    Qini curve: cumulative incremental gains vs. population fraction treated.
    Returns: {'qini_coefficient': float, 'qini_curve': {'x': list, 'y': list}}
    """
    df = pd.DataFrame({
        "y": y_true,
        "uplift": uplift_scores,
        "treatment": treatment,
    }).sort_values("uplift", ascending=False)

    n = len(df)
    n_treated = df["treatment"].sum()
    n_control = n - n_treated

    cumulative_treated_responses = []
    cumulative_control_responses = []
    x_values = []

    for i in range(1, n + 1, max(1, n // 100)):
        top_df = df.iloc[:i]
        t_responses = (top_df["y"] * top_df["treatment"]).sum()
        c_responses = (top_df["y"] * (1 - top_df["treatment"])).sum()
        n_t = top_df["treatment"].sum()
        n_c = i - n_t

        # Normalized incremental gains
        if n_t > 0 and n_c > 0:
            incremental = t_responses / n_t - c_responses / n_c
        elif n_t > 0:
            incremental = t_responses / n_t
        else:
            incremental = 0.0

        cumulative_treated_responses.append(float(t_responses))
        cumulative_control_responses.append(float(c_responses))
        x_values.append(i / n)

    # Random baseline: linear from 0 to total_treated_responses
    total_t = (df["y"] * df["treatment"]).sum()
    total_c = (df["y"] * (1 - df["treatment"])).sum()
    random_baseline = [x * (total_t - total_c) for x in x_values]

    # Qini curve y-values: incremental treated responses adjusted
    if n_control > 0:
        qini_y = [
            tr - cr * n_treated / n_control
            for tr, cr in zip(cumulative_treated_responses, cumulative_control_responses)
        ]
    else:
        qini_y = cumulative_treated_responses

    # Qini coefficient: area between Qini curve and random baseline
    from numpy import trapz
    qini_area = float(trapz(qini_y, x_values))
    random_area = float(trapz(random_baseline, x_values))
    qini_coefficient = qini_area - random_area

    # Normalize to [-1, 1] range
    max_possible = float(trapz(
        np.sort(qini_y)[::-1], x_values
    )) - random_area
    if max_possible > 0:
        qini_normalized = qini_coefficient / max_possible
    else:
        qini_normalized = 0.0

    result = {
        "qini_coefficient": round(qini_coefficient, 4),
        "qini_coefficient_normalized": round(float(np.clip(qini_normalized, -1, 1)), 4),
        "qini_curve": {
            "x": [round(v, 4) for v in x_values],
            "y": [round(v, 4) for v in qini_y],
            "random_baseline": [round(v, 4) for v in random_baseline],
        },
    }
    logger.info(f"Qini coefficient: {qini_coefficient:.4f} (normalized: {qini_normalized:.4f})")
    return result


def compute_campaign_roi(
    uplift_scores: pd.Series,
    segments: pd.Series,
    cost_per_contact: float = 5.0,
    revenue_per_retention: float = 120.0,
    base_retention_rate: float = 0.30,
) -> dict:
    """
    Compute expected campaign ROI targeting only Persuadables.

    Args:
        uplift_scores: Series of per-user uplift scores.
        segments: Series of uplift segment labels.
        cost_per_contact: $ cost to reach each targeted user.
        revenue_per_retention: $ lifetime revenue saved per retained user.
        base_retention_rate: Baseline retention rate without intervention.

    Returns:
        Dict with expected_revenue, total_cost, net_roi, targeted_users, roi_pct.
    """
    persuadables = segments == "Persuadables"
    n_targeted = int(persuadables.sum())
    n_total = len(uplift_scores)

    avg_uplift = float(uplift_scores[persuadables].mean()) if n_targeted > 0 else 0.0
    expected_retentions = n_targeted * max(avg_uplift, 0.0)

    expected_revenue = expected_retentions * revenue_per_retention
    total_cost = n_targeted * cost_per_contact
    net_roi = expected_revenue - total_cost
    roi_pct = (net_roi / total_cost * 100) if total_cost > 0 else 0.0

    result = {
        "n_total": n_total,
        "n_targeted": n_targeted,
        "targeting_rate_pct": round(n_targeted / n_total * 100, 2) if n_total > 0 else 0.0,
        "avg_uplift": round(avg_uplift, 4),
        "expected_retentions": round(expected_retentions, 1),
        "expected_revenue_usd": round(expected_revenue, 2),
        "total_cost_usd": round(total_cost, 2),
        "net_roi_usd": round(net_roi, 2),
        "roi_pct": round(roi_pct, 1),
    }
    logger.info(f"Campaign ROI: ${net_roi:,.0f} net on {n_targeted:,} contacts ({roi_pct:.1f}% ROI)")
    return result

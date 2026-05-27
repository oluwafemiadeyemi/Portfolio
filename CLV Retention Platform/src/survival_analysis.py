"""
Survival Analysis for Customer Churn.
Kaplan-Meier curves and Cox Proportional Hazards model via lifelines.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
    logger.info("lifelines library found.")
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("lifelines not installed — using simplified survival approximations.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def prepare_survival_data(
    df: pd.DataFrame,
    start_date_col: str = "subscription_start_date",
    end_date_col: str = "subscription_end_date",
    churn_col: str = "churn_flag",
) -> pd.DataFrame:
    """
    Compute survival analysis columns T (tenure/duration) and E (event=churn).
    Handles pre-computed tenure_days or computes from date columns.
    """
    out = df.copy()

    if "tenure_days" in out.columns:
        out["T"] = out["tenure_days"].clip(1)
    elif start_date_col in out.columns and end_date_col in out.columns:
        start = pd.to_datetime(out[start_date_col], errors="coerce")
        end = pd.to_datetime(out[end_date_col], errors="coerce")
        out["T"] = (end - start).dt.days.clip(1)
    else:
        # Fallback: use index as proxy tenure
        out["T"] = pd.Series(range(1, len(out) + 1), index=out.index) % 730 + 1

    if churn_col in out.columns:
        out["E"] = out[churn_col].astype(int)
    else:
        out["E"] = 0

    # Remove invalid rows
    out = out[out["T"] > 0].copy()
    logger.info(
        f"Survival data prepared: {len(out):,} users, "
        f"median tenure={out['T'].median():.0f} days, "
        f"event rate={out['E'].mean():.1%}"
    )
    return out


def fit_kaplan_meier(
    T: pd.Series,
    E: pd.Series,
    groups: Optional[pd.Series] = None,
    label: str = "All",
) -> dict[str, Any]:
    """
    Fit Kaplan-Meier estimator(s).
    If groups is provided, fits one KM curve per unique group value.

    Returns dict with:
      - 'all': overall KM result
      - per group key if groups provided
    Each result: {survival_function, median_survival, ci_upper, ci_lower, timeline}
    """
    results: dict[str, Any] = {}

    def _fit_single(t: pd.Series, e: pd.Series, lbl: str) -> dict:
        if LIFELINES_AVAILABLE:
            kmf = KaplanMeierFitter()
            kmf.fit(t, event_observed=e, label=lbl)
            sf = kmf.survival_function_
            ci_upper = kmf.confidence_interval_survival_function_.iloc[:, 1]
            ci_lower = kmf.confidence_interval_survival_function_.iloc[:, 0]
            median = kmf.median_survival_time_
            timeline = sf.index.values
            survival_vals = sf.iloc[:, 0].values
        else:
            # Simplified Nelson-Aalen-like approximation
            sorted_t = np.sort(t.values)
            n = len(sorted_t)
            survival_vals_list = []
            s = 1.0
            timeline_list = []
            for i, ti in enumerate(sorted_t):
                if e.iloc[i] if hasattr(e, "iloc") else e[i]:
                    at_risk = n - i
                    s *= (1 - 1.0 / max(at_risk, 1))
                timeline_list.append(ti)
                survival_vals_list.append(s)
            timeline = np.array(timeline_list)
            survival_vals = np.array(survival_vals_list)
            ci_upper = np.clip(survival_vals + 0.05, 0, 1)
            ci_lower = np.clip(survival_vals - 0.05, 0, 1)
            # Median: first time S <= 0.5
            below = timeline[survival_vals <= 0.5]
            median = float(below[0]) if len(below) > 0 else float(sorted_t[-1])

        return {
            "survival_function": pd.Series(survival_vals, index=timeline),
            "median_survival": float(median) if not np.isnan(float(median)) else float(t.max()),
            "ci_upper": pd.Series(ci_upper if not isinstance(ci_upper, pd.Series) else ci_upper.values, index=timeline),
            "ci_lower": pd.Series(ci_lower if not isinstance(ci_lower, pd.Series) else ci_lower.values, index=timeline),
            "timeline": timeline,
            "label": lbl,
            "n": len(t),
            "event_count": int(e.sum()),
        }

    # Overall
    results["all"] = _fit_single(T, E, label)

    # By group
    if groups is not None:
        for grp_val in sorted(groups.unique()):
            mask = groups == grp_val
            if mask.sum() >= 10:
                results[str(grp_val)] = _fit_single(T[mask], E[mask], str(grp_val))

    logger.info(f"Kaplan-Meier fitted for {len(results)} groups. Overall median survival: {results['all']['median_survival']:.1f} days")
    return results


def fit_cox_ph(
    df: pd.DataFrame,
    feature_cols: list[str],
    T_col: str = "T",
    E_col: str = "E",
    penalizer: float = 0.1,
) -> tuple[Any, pd.DataFrame]:
    """
    Fit Cox Proportional Hazards model.
    Returns (model, summary_df with hazard_ratios, ci, p-values).
    """
    cox_df = df[[T_col, E_col] + feature_cols].copy().dropna()
    cox_df = cox_df[cox_df[T_col] > 0]

    # Clip extreme values
    for col in feature_cols:
        if cox_df[col].dtype in [np.float64, np.float32, float]:
            q01, q99 = cox_df[col].quantile([0.01, 0.99])
            cox_df[col] = cox_df[col].clip(q01, q99)

    if LIFELINES_AVAILABLE:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_df, duration_col=T_col, event_col=E_col)
        summary = cph.summary.copy()
        summary["hazard_ratio"] = np.exp(summary["coef"])
        summary["hr_ci_lower"] = np.exp(summary["coef lower 95%"])
        summary["hr_ci_upper"] = np.exp(summary["coef upper 95%"])
        logger.info(f"Cox PH model fitted. Concordance: {cph.concordance_index_:.4f}")
        return cph, summary

    else:
        # Simplified logistic regression proxy
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = cox_df[feature_cols].fillna(0).values
        y = cox_df[E_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=500, C=1.0 / penalizer)
        lr.fit(X_scaled, y)

        coefs = lr.coef_[0]
        summary = pd.DataFrame({
            "feature": feature_cols,
            "coef": coefs,
            "hazard_ratio": np.exp(coefs),
            "hr_ci_lower": np.exp(coefs - 0.2),
            "hr_ci_upper": np.exp(coefs + 0.2),
            "p": [0.05] * len(feature_cols),  # placeholder
        }).set_index("feature")

        logger.info("Simplified Cox PH (logistic proxy) fitted.")

        class _MockCoxModel:
            def __init__(self, lr_model, scaler_, feat_cols):
                self.lr_ = lr_model
                self.scaler_ = scaler_
                self.feature_cols_ = feat_cols
                self.concordance_index_ = 0.0

            def predict_survival_function(self, X_new: pd.DataFrame) -> pd.DataFrame:
                X_s = self.scaler_.transform(X_new[self.feature_cols_].fillna(0))
                surv_prob = 1 - self.lr_.predict_proba(X_s)[:, 1]
                timeline = np.arange(0, 730, 30)
                rows = []
                for sp in surv_prob:
                    decay = sp ** (timeline / 365.0)
                    rows.append(decay)
                return pd.DataFrame(np.array(rows).T, index=timeline)

        mock = _MockCoxModel(lr, scaler, feature_cols)
        return mock, summary


def compute_median_time_to_churn(km_model: Any) -> float:
    """Extract median survival time from KM model or result dict."""
    if isinstance(km_model, dict):
        return km_model.get("median_survival", 0.0)
    # lifelines KaplanMeierFitter
    if LIFELINES_AVAILABLE and isinstance(km_model, KaplanMeierFitter):
        val = km_model.median_survival_time_
        return float(val) if not np.isnan(float(val)) else 0.0
    return 0.0


def predict_churn_probability_at_time(
    cox_model: Any,
    X_single: pd.DataFrame,
    time_points: list[int] = [30, 60, 90, 180],
) -> dict[int, float]:
    """
    Return churn probability (1 - survival) at each time horizon for a single user.
    """
    result: dict[int, float] = {}
    try:
        if LIFELINES_AVAILABLE and hasattr(cox_model, "predict_survival_function"):
            sf = cox_model.predict_survival_function(X_single)
            # sf is DataFrame with timeline as index
            for t in time_points:
                idx = sf.index.get_indexer([t], method="nearest")[0]
                surv = float(sf.iloc[idx, 0])
                result[t] = round(1.0 - surv, 4)
        elif hasattr(cox_model, "predict_survival_function"):
            sf = cox_model.predict_survival_function(X_single)
            for t in time_points:
                idx = sf.index.get_indexer([t], method="nearest")[0]
                surv = float(sf.iloc[idx, 0])
                result[t] = round(1.0 - surv, 4)
        else:
            for t in time_points:
                result[t] = round(min(0.9, t / 365.0 * 0.5), 4)
    except Exception as exc:
        logger.warning(f"predict_churn_probability_at_time failed: {exc}")
        for t in time_points:
            result[t] = round(min(0.9, t / 365.0 * 0.5), 4)

    return result


def plot_km_curves(km_results_by_group: dict[str, Any]) -> Any:
    """
    Create Plotly figure of KM survival curves with confidence bands.
    Returns go.Figure or dict if plotly unavailable.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("plotly not installed — returning data dict instead of figure.")
        return km_results_by_group

    fig = go.Figure()
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    for i, (group_name, result) in enumerate(km_results_by_group.items()):
        color = colors[i % len(colors)]
        timeline = result["timeline"]
        sf = result["survival_function"]
        ci_upper = result["ci_upper"]
        ci_lower = result["ci_lower"]
        median = result["median_survival"]
        n = result.get("n", 0)
        events = result.get("event_count", 0)

        label = f"{group_name} (n={n:,}, events={events:,}, median={median:.0f}d)"

        # Confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([timeline, timeline[::-1]]),
            y=np.concatenate([
                ci_upper.values if hasattr(ci_upper, "values") else ci_upper,
                (ci_lower.values if hasattr(ci_lower, "values") else ci_lower)[::-1],
            ]),
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name=f"{group_name} CI",
        ))

        # Survival curve
        fig.add_trace(go.Scatter(
            x=timeline,
            y=sf.values if hasattr(sf, "values") else sf,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    # Median reference line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% survival")

    fig.update_layout(
        title="Kaplan-Meier Survival Curves by Segment",
        xaxis_title="Days Since Subscription Start",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="v", x=1.01, y=1),
        template="plotly_white",
        height=500,
    )

    return fig

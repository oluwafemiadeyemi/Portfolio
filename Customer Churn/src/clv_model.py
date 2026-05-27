"""
Customer Lifetime Value Modeling.
Implements BG/NBD + Gamma-Gamma CLV model via the lifetimes library,
with fallback simplified CLV computation when lifetimes is unavailable.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing lifetimes — use simplified models if not available
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    LIFETIMES_AVAILABLE = True
    logger.info("lifetimes library found — using BG/NBD + Gamma-Gamma models.")
except ImportError:
    LIFETIMES_AVAILABLE = False
    logger.warning("lifetimes not installed — falling back to simplified CLV approximation.")


def _prepare_rfm_table(
    df: pd.DataFrame,
    frequency_col: str,
    recency_col: str,
    T_col: str,
    monetary_col: str,
) -> pd.DataFrame:
    """Ensure non-negative values required by lifetimes models."""
    rfm = df[[frequency_col, recency_col, T_col, monetary_col]].copy()
    rfm[frequency_col] = rfm[frequency_col].clip(0)
    rfm[recency_col] = rfm[recency_col].clip(0)
    rfm[T_col] = rfm[T_col].clip(1)  # T must be > 0
    rfm[monetary_col] = rfm[monetary_col].clip(0.01)
    # Recency must be <= T
    rfm[recency_col] = rfm[[recency_col, T_col]].min(axis=1)
    return rfm.dropna()


class _SimpleBGNBD:
    """
    Simplified Pareto/NBD-style approximation when lifetimes is not installed.
    Uses log-linear approximation: E[X(t)] ≈ (frequency+1) * (t / T) * alive_prob
    """

    def __init__(self) -> None:
        self.params_: dict[str, float] = {}

    def fit(
        self,
        rfm: pd.DataFrame,
        frequency_col: str,
        recency_col: str,
        T_col: str,
    ) -> "_SimpleBGNBD":
        freq = rfm[frequency_col]
        T = rfm[T_col]
        recency = rfm[recency_col]

        # Store aggregate stats
        self.params_["mean_freq"] = float(freq.mean())
        self.params_["mean_T"] = float(T.mean())
        self.params_["mean_purchase_rate"] = float((freq / T.replace(0, 1)).mean())
        logger.info(f"SimpleBGNBD fitted: mean_freq={self.params_['mean_freq']:.2f}")
        return self

    def conditional_expected_number_of_purchases_up_to_time(
        self, t: float, frequency: pd.Series, recency: pd.Series, T: pd.Series
    ) -> pd.Series:
        purchase_rate = frequency / T.replace(0, 1)
        recency_ratio = recency / T.replace(0, 1)
        alive_prob = np.exp(-0.5 * (1 - recency_ratio))
        return purchase_rate * t * alive_prob


class _SimpleGammaGamma:
    """Simplified Gamma-Gamma expected average order value model."""

    def __init__(self) -> None:
        self.mean_monetary_: float = 0.0

    def fit(
        self, rfm: pd.DataFrame, frequency_col: str, monetary_col: str
    ) -> "_SimpleGammaGamma":
        active = rfm[rfm[frequency_col] > 0]
        self.mean_monetary_ = float(active[monetary_col].mean()) if len(active) > 0 else 10.0
        logger.info(f"SimpleGammaGamma fitted: mean_monetary={self.mean_monetary_:.2f}")
        return self

    def conditional_expected_average_profit(
        self, frequency: pd.Series, monetary_value: pd.Series
    ) -> pd.Series:
        # Shrink individual monetary value toward global mean
        shrinkage = 0.5
        return shrinkage * self.mean_monetary_ + (1 - shrinkage) * monetary_value


def fit_bgnbd_model(
    df: pd.DataFrame,
    frequency_col: str = "rfm_frequency",
    recency_col: str = "rfm_recency",
    T_col: str = "tenure_days",
    monetary_col: str = "rfm_monetary",
    penalizer_coef: float = 0.01,
) -> Any:
    """
    Fit BG/NBD model for predicting future purchase counts.
    Uses lifetimes.BetaGeoFitter if available, else SimpleBGNBD.
    """
    rfm = _prepare_rfm_table(df, frequency_col, recency_col, T_col, monetary_col)
    # Convert recency to weeks for better model stability
    rfm_weeks = rfm.copy()
    rfm_weeks[recency_col] = rfm_weeks[recency_col] / 7.0
    rfm_weeks[T_col] = rfm_weeks[T_col] / 7.0

    if LIFETIMES_AVAILABLE:
        model = BetaGeoFitter(penalizer_coef=penalizer_coef)
        model.fit(
            rfm_weeks[frequency_col],
            rfm_weeks[recency_col],
            rfm_weeks[T_col],
        )
        logger.info(f"BG/NBD fitted. Params: {model.params_}")
    else:
        model = _SimpleBGNBD()
        model.fit(rfm, frequency_col, recency_col, T_col)

    return model


def predict_future_purchases(
    model: Any,
    df: pd.DataFrame,
    t_weeks: float = 52,
    frequency_col: str = "rfm_frequency",
    recency_col: str = "rfm_recency",
    T_col: str = "tenure_days",
) -> pd.Series:
    """Predict expected number of purchases in next t_weeks weeks."""
    freq = df[frequency_col].fillna(0).clip(0)
    recency_weeks = df[recency_col].fillna(0).clip(0) / 7.0
    T_weeks = df[T_col].fillna(30).clip(1) / 7.0

    if LIFETIMES_AVAILABLE and isinstance(model, BetaGeoFitter):
        pred = model.conditional_expected_number_of_purchases_up_to_time(
            t_weeks, freq, recency_weeks, T_weeks
        )
    else:
        pred = model.conditional_expected_number_of_purchases_up_to_time(
            t_weeks, freq, recency_weeks, T_weeks
        )

    return pd.Series(pred.values if hasattr(pred, "values") else pred, index=df.index, name="predicted_purchases")


def fit_gamma_gamma(
    df: pd.DataFrame,
    frequency_col: str = "rfm_frequency",
    monetary_col: str = "rfm_monetary",
    penalizer_coef: float = 0.01,
) -> Any:
    """
    Fit Gamma-Gamma model to estimate expected average order value.
    Requires frequency > 0 customers.
    """
    rfm = _prepare_rfm_table(df, frequency_col, "tenure_days" if "tenure_days" in df.columns else frequency_col,
                              "tenure_days" if "tenure_days" in df.columns else frequency_col, monetary_col)
    active = rfm[rfm[frequency_col] > 0]

    if len(active) < 10:
        logger.warning("Insufficient repeat customers for Gamma-Gamma. Using simplified model.")
        model = _SimpleGammaGamma()
        model.fit(rfm, frequency_col, monetary_col)
        return model

    if LIFETIMES_AVAILABLE:
        model = GammaGammaFitter(penalizer_coef=penalizer_coef)
        model.fit(active[frequency_col], active[monetary_col])
        logger.info(f"Gamma-Gamma fitted. Params: {model.params_}")
    else:
        model = _SimpleGammaGamma()
        model.fit(rfm, frequency_col, monetary_col)

    return model


def compute_clv(
    bgnbd_model: Any,
    ggm_model: Any,
    df: pd.DataFrame,
    months: int = 12,
    discount_rate: float = 0.01,
    frequency_col: str = "rfm_frequency",
    recency_col: str = "rfm_recency",
    T_col: str = "tenure_days",
    monetary_col: str = "rfm_monetary",
) -> pd.Series:
    """
    Compute Customer Lifetime Value combining BG/NBD and Gamma-Gamma models.
    CLV = predicted_purchases * expected_avg_value * discount_factor
    """
    freq = df[frequency_col].fillna(0).clip(0)
    recency_weeks = df[recency_col].fillna(0).clip(0) / 7.0
    T_weeks = df[T_col].fillna(30).clip(1) / 7.0
    monetary = df[monetary_col].fillna(10.0).clip(0.01)

    t_weeks = months * 4.33

    if LIFETIMES_AVAILABLE and isinstance(bgnbd_model, BetaGeoFitter):
        predicted_purchases = bgnbd_model.conditional_expected_number_of_purchases_up_to_time(
            t_weeks, freq, recency_weeks, T_weeks
        )
        if isinstance(ggm_model, GammaGammaFitter):
            expected_avg_profit = ggm_model.conditional_expected_average_profit(freq, monetary)
        else:
            expected_avg_profit = ggm_model.conditional_expected_average_profit(freq, monetary)
    else:
        predicted_purchases = bgnbd_model.conditional_expected_number_of_purchases_up_to_time(
            t_weeks, freq, recency_weeks, T_weeks
        )
        expected_avg_profit = ggm_model.conditional_expected_average_profit(freq, monetary)

    predicted_purchases = pd.Series(
        predicted_purchases.values if hasattr(predicted_purchases, "values") else predicted_purchases,
        index=df.index,
    )
    expected_avg_profit = pd.Series(
        expected_avg_profit.values if hasattr(expected_avg_profit, "values") else expected_avg_profit,
        index=df.index,
    )

    # Apply discount factor for time value of money
    discount_factor = 1.0 / (1.0 + discount_rate) ** months
    clv = predicted_purchases * expected_avg_profit * discount_factor
    clv = clv.clip(0).rename("clv")
    logger.info(f"CLV computed: mean=${clv.mean():.2f}, median=${clv.median():.2f}, max=${clv.max():.2f}")
    return clv


def segment_by_clv(
    clv_series: pd.Series,
    quantiles: list[float] = [0.25, 0.75],
) -> pd.Series:
    """
    Segment customers into Low / Medium / High CLV tiers.
    Returns string series with segment labels.
    """
    q_low = clv_series.quantile(quantiles[0])
    q_high = clv_series.quantile(quantiles[1])

    conditions = [clv_series <= q_low, clv_series <= q_high, clv_series > q_high]
    choices = ["Low", "Medium", "High"]
    return pd.Series(
        np.select(conditions, choices, default="Medium"),
        index=clv_series.index,
        name="clv_segment",
    )


def compute_simple_clv(
    df: pd.DataFrame,
    avg_monthly_revenue: Optional[float] = None,
    churn_proba: Optional[pd.Series] = None,
    discount_rate: float = 0.01,
) -> pd.Series:
    """
    Fallback CLV formula: CLV = avg_revenue / (discount_rate + monthly_churn_rate)
    monthly_churn_rate is approximated from churn_proba (annual).
    """
    if avg_monthly_revenue is None:
        if "plan_list_price" in df.columns:
            avg_monthly_revenue = float(df["plan_list_price"].mean() / 30 * 30)
        elif "monthly_charges" in df.columns:
            avg_monthly_revenue = float(df["monthly_charges"].mean())
        else:
            avg_monthly_revenue = 15.0

    if churn_proba is None:
        if "churn_flag" in df.columns:
            churn_proba = df["churn_flag"].astype(float)
        else:
            churn_proba = pd.Series(np.full(len(df), 0.25), index=df.index)

    monthly_churn_rate = churn_proba / 12.0
    clv = avg_monthly_revenue / (discount_rate + monthly_churn_rate.clip(0.001))
    clv = clv.clip(0).rename("clv_simple")
    logger.info(f"Simple CLV computed: mean=${clv.mean():.2f}")
    return clv

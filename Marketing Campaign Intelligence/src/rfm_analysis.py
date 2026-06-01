"""
RFM (Recency–Frequency–Monetary) analysis with quintile scoring,
segment labelling, and CLV estimation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R, F, M scores (1–5 quintiles) from the synthetic events dataframe.
    Returns enriched dataframe with scores and segment labels.
    """
    rfm = df[["customer_id", "recency_days", "frequency", "monetary"]].copy()

    # Quintile scoring: Recency lower = better → invert
    rfm["R_score"] = pd.qcut(rfm["recency_days"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5,
                              labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5,
                              labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    rfm["segment"] = rfm.apply(_label_segment, axis=1)

    # Simple CLV estimate: M * F * 12 / recency_weight
    rfm["clv_estimate"] = (
        rfm["monetary"] * rfm["frequency"] * 12 / np.log1p(rfm["recency_days"])
    ).round(2)

    return rfm


def _label_segment(row: pd.Series) -> str:
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "Promising"
    elif r >= 3 and f <= 2 and m <= 2:
        return "Potential Loyalist"
    elif r <= 2 and f >= 4:
        return "At Risk"
    elif r <= 2 and f <= 2 and m >= 4:
        return "Can't Lose Them"
    elif r <= 2 and f <= 2:
        return "Lost"
    elif r >= 4 and f == 1:
        return "New Customer"
    else:
        return "Needs Attention"


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per RFM segment for executive dashboard."""
    agg = rfm.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_clv=("clv_estimate", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(2)
    agg["pct"] = (agg["count"] / len(rfm) * 100).round(2)
    agg = agg.sort_values("avg_clv", ascending=False)
    return agg


def run_rfm_analysis(save: bool = True) -> pd.DataFrame:
    """Load synthetic events, compute RFM, save results."""
    path = DATA_PROC / "synthetic_events.parquet"
    if not path.exists():
        raise FileNotFoundError("Run data_pipeline.prepare_all() first.")

    df = pd.read_parquet(path)
    print(f"Computing RFM on {len(df):,} customers ...")
    rfm = compute_rfm(df)

    if save:
        rfm.to_parquet(DATA_PROC / "rfm_scored.parquet", index=False)
        summary = segment_summary(rfm)
        summary.to_csv(DATA_PROC / "rfm_segment_summary.csv")
        print(f"RFM saved. Segments:\n{summary[['count', 'pct', 'avg_clv', 'total_revenue']]}")

    return rfm


if __name__ == "__main__":
    rfm = run_rfm_analysis()
    print(segment_summary(rfm))

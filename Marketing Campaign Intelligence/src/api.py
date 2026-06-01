"""
FastAPI backend for the Marketing Campaign Intelligence Platform.
Endpoints: /segment, /rfm, /attribution, /basket, /health
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Marketing Campaign Intelligence Platform",
    description="Enterprise customer segmentation, RFM analysis, multi-touch attribution & basket analysis API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory caches ───────────────────────────────────────────────────────
_rfm_df: Optional[pd.DataFrame] = None
_segment_df: Optional[pd.DataFrame] = None
_attribution_df: Optional[pd.DataFrame] = None
_rules_df: Optional[pd.DataFrame] = None


def _load_rfm():
    global _rfm_df
    if _rfm_df is None:
        p = DATA_PROC / "rfm_scored.parquet"
        if p.exists():
            _rfm_df = pd.read_parquet(p)
    return _rfm_df


def _load_segments():
    global _segment_df
    if _segment_df is None:
        p = DATA_PROC / "segment_embeddings.parquet"
        if p.exists():
            _segment_df = pd.read_parquet(p)
    return _segment_df


def _load_attribution():
    global _attribution_df
    if _attribution_df is None:
        p = DATA_PROC / "attribution_pct.csv"
        if p.exists():
            _attribution_df = pd.read_csv(p, index_col=0)
    return _attribution_df


def _load_rules():
    global _rules_df
    if _rules_df is None:
        p = DATA_PROC / "association_rules.parquet"
        if p.exists():
            _rules_df = pd.read_parquet(p)
            for col in ["support", "confidence", "lift"]:
                if col in _rules_df.columns:
                    _rules_df[col] = pd.to_numeric(_rules_df[col], errors="coerce")
    return _rules_df


# ─── Request / Response Models ───────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=1, description="Number of purchases")
    monetary: float = Field(..., ge=0, description="Total spend ($)")
    age: Optional[int] = None
    income: Optional[float] = None
    channel: Optional[str] = None


class BatchCustomers(BaseModel):
    customers: List[CustomerFeatures]


class SegmentResponse(BaseModel):
    segment: str
    rfm_score: float
    r_score: int
    f_score: int
    m_score: int
    clv_estimate: float
    recommendations: List[str]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    rfm_ready = (DATA_PROC / "rfm_scored.parquet").exists()
    seg_ready = (DATA_PROC / "segment_embeddings.parquet").exists()
    return {
        "status": "healthy",
        "rfm_data_loaded": rfm_ready,
        "segmentation_data_loaded": seg_ready,
        "version": "1.0.0",
    }


@app.post("/segment", response_model=SegmentResponse)
def segment_customer(customer: CustomerFeatures):
    """Classify a single customer into a segment with CLV and next-best-action."""
    r, f, m = customer.recency_days, customer.frequency, customer.monetary

    # Quintile-based scoring (simplified for single-customer inference)
    r_score = _score_recency(r)
    f_score = _score_frequency(f)
    m_score = _score_monetary(m)
    rfm_total = r_score + f_score + m_score
    rfm_score = round(rfm_total / 15 * 100, 1)

    segment = _assign_segment(r_score, f_score, m_score)
    clv = round(m * f * 12 / (np.log1p(r) + 1e-6), 2)
    recs = _next_best_actions(segment)

    return SegmentResponse(
        segment=segment,
        rfm_score=rfm_score,
        r_score=r_score,
        f_score=f_score,
        m_score=m_score,
        clv_estimate=clv,
        recommendations=recs,
    )


@app.get("/rfm/summary")
def rfm_summary():
    """Return aggregated RFM segment summary statistics."""
    rfm = _load_rfm()
    if rfm is None:
        raise HTTPException(503, "RFM data not yet generated. Run rfm_analysis.run_rfm_analysis() first.")

    summary = (
        rfm.groupby("segment")
        .agg(
            count=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_clv=("clv_estimate", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .round(2)
    )
    summary["pct"] = (summary["count"] / len(rfm) * 100).round(2)
    return summary.reset_index().to_dict(orient="records")


@app.get("/rfm/segment/{segment_name}")
def rfm_segment_detail(segment_name: str, limit: int = 100):
    """Return individual customers in a given RFM segment."""
    rfm = _load_rfm()
    if rfm is None:
        raise HTTPException(503, "RFM data not ready.")
    mask = rfm["segment"].str.lower() == segment_name.lower()
    if not mask.any():
        raise HTTPException(404, f"Segment '{segment_name}' not found.")
    subset = rfm[mask].head(limit)
    return subset[["customer_id", "recency_days", "frequency", "monetary",
                    "R_score", "F_score", "M_score", "RFM_total", "clv_estimate"]].to_dict(orient="records")


@app.get("/attribution")
def attribution():
    """Return multi-touch attribution percentages across all models."""
    attr = _load_attribution()
    if attr is None:
        raise HTTPException(503, "Attribution data not ready.")
    return attr.to_dict()


@app.get("/segments/overview")
def segments_overview():
    """Return cluster size distribution from UMAP+HDBSCAN segmentation."""
    seg = _load_segments()
    if seg is None:
        raise HTTPException(503, "Segmentation data not ready.")
    counts = seg["segment_name"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    counts["pct"] = (counts["count"] / len(seg) * 100).round(2)
    return counts.to_dict(orient="records")


@app.get("/basket/top-rules")
def top_association_rules(limit: int = 20, min_lift: float = 1.1):
    """Return top association rules by lift, excluding channel items."""
    rules = _load_rules()
    if rules is None:
        raise HTTPException(503, "Association rules not ready.")
    mask = (
        ~rules["antecedents_str"].str.contains("ch:", na=False) &
        ~rules["consequents_str"].str.contains("ch:", na=False) &
        (rules["lift"] >= min_lift)
    )
    top = rules[mask].nlargest(limit, "lift")[
        ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    ]
    return top.to_dict(orient="records")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _score_recency(r: float) -> int:
    if r <= 7:   return 5
    if r <= 30:  return 4
    if r <= 60:  return 3
    if r <= 120: return 2
    return 1


def _score_frequency(f: int) -> int:
    if f >= 15: return 5
    if f >= 10: return 4
    if f >= 5:  return 3
    if f >= 2:  return 2
    return 1


def _score_monetary(m: float) -> int:
    if m >= 400: return 5
    if m >= 200: return 4
    if m >= 100: return 3
    if m >= 50:  return 2
    return 1


def _assign_segment(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customer"
    elif r <= 2 and f >= 4:
        return "At Risk"
    elif r <= 2 and f <= 2:
        return "Lost"
    else:
        return "Needs Attention"


def _next_best_actions(segment: str) -> List[str]:
    actions = {
        "Champions":       ["Send VIP loyalty rewards", "Request product review", "Offer referral bonus"],
        "Loyal Customers": ["Upsell premium tier", "Send personalised thank-you", "Exclusive early access"],
        "New Customer":    ["Onboarding email series", "First-purchase discount", "Product education content"],
        "At Risk":         ["Win-back campaign with 20% discount", "Personalised re-engagement email", "Survey to understand churn reason"],
        "Lost":            ["Aggressive win-back offer 30%", "Remove from email cadence", "Retargeting ads"],
        "Needs Attention": ["Check engagement score", "Offer mid-tier incentive", "Product recommendation email"],
    }
    return actions.get(segment, ["Review customer profile manually"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)

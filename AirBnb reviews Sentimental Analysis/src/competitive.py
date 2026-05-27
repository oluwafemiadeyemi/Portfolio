"""
Competitive benchmarking and market intelligence for the Brand Intelligence Platform.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brand Benchmarking
# ---------------------------------------------------------------------------

def benchmark_brands(
    df: pd.DataFrame,
    brand_col: str = "business_name",
    score_col: str = "compound_sentiment",
) -> pd.DataFrame:
    """Return DataFrame with brand → mean score, count, std, rank."""
    logger.info("Benchmarking brands on column '%s'", score_col)

    if brand_col not in df.columns:
        raise ValueError(f"Column '{brand_col}' not found in DataFrame")
    if score_col not in df.columns:
        logger.warning("Score column '%s' not found — using rating as proxy", score_col)
        df = df.copy()
        df[score_col] = (df["rating"] - 3.0) / 2.0  # Normalise rating to [-1, 1]

    summary = (
        df.groupby(brand_col)[score_col]
        .agg(["mean", "count", "std", "median"])
        .reset_index()
        .rename(columns={
            brand_col: "brand",
            "mean": "avg_sentiment",
            "count": "review_count",
            "std": "sentiment_std",
            "median": "median_sentiment",
        })
    )

    # Add rating stats if available
    if "rating" in df.columns:
        rating_stats = df.groupby(brand_col)["rating"].agg(["mean", "std"]).reset_index()
        rating_stats.columns = ["brand", "avg_rating", "rating_std"]
        summary = summary.merge(rating_stats, on="brand", how="left")

    # Rank (1 = best)
    summary = summary.sort_values("avg_sentiment", ascending=False)
    summary["rank"] = range(1, len(summary) + 1)

    # Percentile score
    summary["percentile"] = summary["avg_sentiment"].rank(pct=True).round(3)

    logger.info("Benchmarked %d brands", len(summary))
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------

def compute_gap_analysis(
    brand_scores: pd.DataFrame,
    target_brand: str,
    competitor_brands: List[str],
) -> pd.DataFrame:
    """Return per-aspect gaps between target brand and competitors."""
    logger.info("Computing gap analysis for '%s' vs %d competitors", target_brand, len(competitor_brands))

    aspect_cols = [c for c in brand_scores.columns if c.startswith("aspect_") and "compound" in c]
    if not aspect_cols:
        # Fall back to single overall score
        if "avg_sentiment" not in brand_scores.columns:
            raise ValueError("brand_scores must contain 'avg_sentiment' or 'aspect_*_compound' columns")
        aspect_cols = ["avg_sentiment"]

    target_row = brand_scores[brand_scores["brand"] == target_brand]
    if target_row.empty:
        raise ValueError(f"Target brand '{target_brand}' not found in brand_scores")

    records = []
    for competitor in competitor_brands:
        comp_row = brand_scores[brand_scores["brand"] == competitor]
        if comp_row.empty:
            continue
        row = {"competitor": competitor}
        for col in aspect_cols:
            if col not in target_row.columns or col not in comp_row.columns:
                continue
            target_val = float(target_row[col].iloc[0])
            comp_val = float(comp_row[col].iloc[0])
            gap = target_val - comp_val
            clean_name = col.replace("aspect_", "").replace("_compound", "").replace("avg_", "")
            row[f"gap_{clean_name}"] = round(gap, 4)
            row[f"target_{clean_name}"] = round(target_val, 4)
            row[f"competitor_{clean_name}"] = round(comp_val, 4)
        records.append(row)

    result = pd.DataFrame(records)
    logger.info("Gap analysis computed for %d competitors", len(result))
    return result


# ---------------------------------------------------------------------------
# Market Share of Voice
# ---------------------------------------------------------------------------

def market_share_of_voice(
    df: pd.DataFrame,
    brand_col: str = "business_name",
) -> pd.DataFrame:
    """Return review count as % of total per brand."""
    logger.info("Computing market share of voice")

    if brand_col not in df.columns:
        raise ValueError(f"Column '{brand_col}' not found in DataFrame")

    counts = df[brand_col].value_counts().reset_index()
    counts.columns = ["brand", "review_count"]
    total = counts["review_count"].sum()
    counts["share_pct"] = (counts["review_count"] / total * 100).round(2)
    counts["rank"] = range(1, len(counts) + 1)

    logger.info("Share of voice: top brand '%s' = %.1f%%",
                counts.iloc[0]["brand"], counts.iloc[0]["share_pct"])
    return counts


# ---------------------------------------------------------------------------
# NPS Proxy
# ---------------------------------------------------------------------------

def compute_nps_proxy(df: pd.DataFrame, rating_col: str = "rating") -> Dict[str, float]:
    """Map ratings to NPS-like score (9-10=promoter, 7-8=passive, 1-6=detractor) on 1-5 scale."""
    logger.info("Computing NPS proxy on %d reviews", len(df))

    if rating_col not in df.columns:
        raise ValueError(f"Column '{rating_col}' not found in DataFrame")

    ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()

    # Detect scale: 1-5 or 1-10 or 1-100
    max_rating = ratings.max()
    if max_rating <= 5:
        # Map 1-5 → 1-10
        ratings_10 = ratings * 2
    elif max_rating <= 10:
        ratings_10 = ratings
    else:
        # 1-100 → 1-10
        ratings_10 = ratings / 10.0

    n_total = len(ratings_10)
    if n_total == 0:
        return {"nps": 0.0, "promoters_pct": 0.0, "passives_pct": 0.0, "detractors_pct": 0.0, "n": 0}

    n_promoters = int((ratings_10 >= 9).sum())
    n_passives = int(((ratings_10 >= 7) & (ratings_10 < 9)).sum())
    n_detractors = int((ratings_10 < 7).sum())

    promoters_pct = n_promoters / n_total * 100
    detractors_pct = n_detractors / n_total * 100
    passives_pct = n_passives / n_total * 100
    nps = promoters_pct - detractors_pct

    result = {
        "nps": round(nps, 1),
        "promoters_pct": round(promoters_pct, 1),
        "passives_pct": round(passives_pct, 1),
        "detractors_pct": round(detractors_pct, 1),
        "n_promoters": n_promoters,
        "n_passives": n_passives,
        "n_detractors": n_detractors,
        "n": n_total,
    }
    logger.info("NPS proxy: %.1f (promoters=%.1f%%, detractors=%.1f%%)",
                nps, promoters_pct, detractors_pct)
    return result


# ---------------------------------------------------------------------------
# Full Competitive Report
# ---------------------------------------------------------------------------

def build_competitive_report(
    df: pd.DataFrame,
    target_brand: Optional[str] = None,
) -> Dict:
    """Build a comprehensive competitive report dict."""
    logger.info("Building competitive report")

    # Compute sentiment if not already present
    if "compound_sentiment" not in df.columns:
        from src.features import extract_sentiment_lexicon
        scores = df["text"].fillna("").apply(extract_sentiment_lexicon)
        df = df.copy()
        df["compound_sentiment"] = scores.apply(lambda s: s["compound"])

    benchmarks = benchmark_brands(df)
    sov = market_share_of_voice(df)
    nps = compute_nps_proxy(df)

    # Auto-select target brand if not specified
    if target_brand is None and not benchmarks.empty:
        target_brand = benchmarks.iloc[0]["brand"]

    competitor_brands = benchmarks["brand"].tolist()
    if target_brand in competitor_brands:
        competitor_brands = [b for b in competitor_brands if b != target_brand]

    gap_analysis = pd.DataFrame()
    if target_brand and competitor_brands:
        try:
            gap_analysis = compute_gap_analysis(benchmarks, target_brand, competitor_brands[:5])
        except ValueError as exc:
            logger.warning("Gap analysis skipped: %s", exc)

    return {
        "benchmarks": benchmarks,
        "share_of_voice": sov,
        "nps": nps,
        "gap_analysis": gap_analysis,
        "target_brand": target_brand,
    }

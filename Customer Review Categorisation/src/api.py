"""
FastAPI: Generative AI Customer Review Intelligence Platform
Endpoints: /classify, /search, /ask, /summary, /analytics
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"

app = FastAPI(
    title="Generative AI Customer Review Intelligence Platform",
    description="Claude-powered review classification, RAG Q&A, and executive VOC insights",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_reviews_df = None


def _load_reviews():
    global _reviews_df
    if _reviews_df is None:
        p = DATA_PROC / "reviews.parquet"
        if p.exists():
            _reviews_df = pd.read_parquet(p)
    return _reviews_df


# ─── Request / Response Models ───────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=5, example="The battery died after 2 days.")
    use_ai: bool = Field(default=True, description="Use Claude API (requires ANTHROPIC_API_KEY)")


class ClassifyResponse(BaseModel):
    category: str
    sentiment: str
    sentiment_score: float
    aspects: List[str]
    key_issues: List[str]
    action_required: bool
    priority: str
    summary: str
    model_used: str


class SearchRequest(BaseModel):
    query: str = Field(..., example="battery problems with electronics")
    n: int = Field(default=10, ge=1, le=50)
    filter_category: Optional[str] = None
    filter_sentiment: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., example="What are the most common complaints about electronics?")


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    reviews_loaded = _load_reviews() is not None
    return {
        "status": "healthy",
        "anthropic_api_configured": api_key_set,
        "reviews_loaded": reviews_loaded,
        "model": "claude-sonnet-4-6",
        "features": ["classify", "rag_search", "ask", "analytics", "batch_classify"],
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify_review(req: ClassifyRequest):
    """Classify a single review using Claude or rule-based fallback."""
    if req.use_ai and os.getenv("ANTHROPIC_API_KEY"):
        from classifier import ReviewClassifier
        clf = ReviewClassifier()
        result = clf.classify_single(req.text)
        result["model_used"] = "claude-sonnet-4-6"
    else:
        from classifier import _fallback_classification
        result = _fallback_classification(req.text)
        result["model_used"] = "rule_based_fallback"

    return ClassifyResponse(
        category=result.get("category", "Product Quality"),
        sentiment=result.get("sentiment", "Neutral"),
        sentiment_score=float(result.get("sentiment_score", 0.0)),
        aspects=result.get("aspects", []),
        key_issues=result.get("key_issues", []),
        action_required=bool(result.get("action_required", False)),
        priority=result.get("priority", "Medium"),
        summary=result.get("summary", req.text[:100]),
        model_used=result.get("model_used", "unknown"),
    )


@app.post("/search")
def semantic_search(req: SearchRequest):
    """Semantic search over the review vector store."""
    from rag_pipeline import semantic_search as _search
    results = _search(req.query, n=req.n,
                      filter_category=req.filter_category,
                      filter_sentiment=req.filter_sentiment)
    return {"query": req.query, "results": results}


@app.post("/ask")
def ask_question(req: AskRequest):
    """RAG Q&A: retrieve relevant reviews and generate Claude answer."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured.")
    from rag_pipeline import answer_question
    result = answer_question(req.question)
    return result


@app.get("/analytics/overview")
def analytics_overview():
    """Return high-level review analytics."""
    df = _load_reviews()
    if df is None:
        raise HTTPException(503, "Run data_pipeline.prepare_all() first.")
    return {
        "total_reviews":        len(df),
        "avg_rating":           round(float(df["rating"].mean()), 2) if "rating" in df.columns else None,
        "sentiment_distribution": df["sentiment"].value_counts().to_dict() if "sentiment" in df.columns else {},
        "category_distribution":  df["category"].value_counts().head(12).to_dict() if "category" in df.columns else {},
        "review_category_distribution": df["review_category_label"].value_counts().to_dict() if "review_category_label" in df.columns else {},
        "avg_word_count":       round(float(df["word_count"].mean()), 1) if "word_count" in df.columns else None,
        "action_required_pct":  round(float((df["sentiment"] == "Negative").mean() * 100), 1) if "sentiment" in df.columns else None,
    }


@app.get("/analytics/category/{category}")
def category_analytics(category: str):
    """Deep analytics for a specific product category."""
    df = _load_reviews()
    if df is None:
        raise HTTPException(503, "No data loaded.")
    subset = df[df["category"].str.lower() == category.lower()] if "category" in df.columns else df
    if len(subset) == 0:
        raise HTTPException(404, f"Category '{category}' not found.")
    return {
        "category":      category,
        "count":         len(subset),
        "avg_rating":    round(float(subset["rating"].mean()), 2) if "rating" in subset.columns else None,
        "sentiment_dist": subset["sentiment"].value_counts().to_dict() if "sentiment" in subset.columns else {},
        "top_issues":    subset["review_category_label"].value_counts().head(5).to_dict() if "review_category_label" in subset.columns else {},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8007, reload=True)

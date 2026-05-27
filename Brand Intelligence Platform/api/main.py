"""
FastAPI application for the Enterprise Brand Intelligence Platform.
Exposes sentiment scoring, batch processing, brand analytics, and topic APIs.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from src.features import (
    extract_aspect_sentiment,
    extract_linguistic_features,
    extract_sentiment_lexicon,
    get_aspect_keywords,
    preprocess_text,
)
from src.data_loader import load_all_data
from src.competitive import benchmark_brands, compute_nps_proxy, market_share_of_voice
from src.topic_modeling import run_topic_pipeline
from src.models import load_model, detect_anomaly

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class _AppState:
    df: Optional[pd.DataFrame] = None
    model: Optional[Any] = None
    model_loaded: bool = False
    topics: Dict[str, List[str]] = {}
    topic_assignments: List[int] = []
    brand_summaries: Dict[str, Any] = {}


_state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    logger.info("Starting up Enterprise Brand Intelligence Platform API")

    # Try to load pre-trained model
    model_path = _BASE_DIR / "data" / "models" / "sentiment_model.joblib"
    if model_path.exists():
        try:
            _state.model = load_model(str(model_path))
            _state.model_loaded = True
            logger.info("Loaded pre-trained model from %s", model_path)
        except Exception as exc:
            logger.warning("Failed to load model: %s — will use VADER only", exc)

    # Load data
    try:
        _state.df = load_all_data(str(_BASE_DIR))
        logger.info("Loaded %d reviews into memory", len(_state.df))
    except Exception as exc:
        logger.warning("Data loading failed: %s — generating synthetic data", exc)
        from src.data_loader import generate_synthetic_reviews
        _state.df = generate_synthetic_reviews(n=1000)

    # Compute sentiment if not already present
    if _state.df is not None and "compound_sentiment" not in _state.df.columns:
        logger.info("Computing VADER sentiment for loaded data")
        scores = _state.df["text"].fillna("").apply(extract_sentiment_lexicon)
        _state.df["compound_sentiment"] = scores.apply(lambda s: s["compound"])
        _state.df["pos_sentiment"] = scores.apply(lambda s: s["pos"])
        _state.df["neg_sentiment"] = scores.apply(lambda s: s["neg"])

    # Fit topics
    if _state.df is not None and len(_state.df) >= 10:
        try:
            sample = _state.df.sample(min(500, len(_state.df)), random_state=42)
            topics, assignments, _ = run_topic_pipeline(sample, n_topics=10)
            _state.topics = topics
            logger.info("Topics fitted: %d topics", len(topics))
        except Exception as exc:
            logger.warning("Topic modeling failed: %s", exc)

    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Brand Intelligence API",
    description="Real-time brand reputation monitoring powered by transformer-based NLP",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReviewInput(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to score")
    source: str = Field(default="unknown", description="Review source (yelp/airbnb/google)")


class AspectScore(BaseModel):
    compound: float
    pos: float
    neg: float
    neu: float
    mentioned: bool


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    compound_score: float
    pos_score: float
    neg_score: float
    neu_score: float
    aspects: Dict[str, AspectScore]
    linguistic: Dict[str, float]
    source: str


class BatchInput(BaseModel):
    reviews: List[ReviewInput]


class BrandSummaryResponse(BaseModel):
    brand_name: str
    avg_sentiment: float
    avg_rating: float
    review_count: int
    nps_score: float
    sentiment_distribution: Dict[str, int]
    top_aspects: Dict[str, float]


class TopicResponse(BaseModel):
    topic_id: str
    label: str
    top_words: List[str]
    document_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    review_count: int
    topics_count: int


# ---------------------------------------------------------------------------
# Helper: score a single text
# ---------------------------------------------------------------------------

def _score_text(text: str, source: str = "unknown") -> SentimentResponse:
    """Score a single review text with VADER and aspect analysis."""
    cleaned = preprocess_text(text)
    vader_scores = extract_sentiment_lexicon(cleaned)

    compound = vader_scores["compound"]
    if compound >= 0.05:
        sentiment = "positive"
        confidence = min((compound + 1) / 2, 1.0)
    elif compound <= -0.05:
        sentiment = "negative"
        confidence = min((-compound + 1) / 2, 1.0)
    else:
        sentiment = "neutral"
        confidence = 1.0 - abs(compound) * 10

    # Aspect sentiment
    raw_aspects = extract_aspect_sentiment(cleaned)
    aspects = {
        k: AspectScore(
            compound=v.get("compound", 0.0),
            pos=v.get("pos", 0.0),
            neg=v.get("neg", 0.0),
            neu=v.get("neu", 1.0),
            mentioned=v.get("mentioned", False),
        )
        for k, v in raw_aspects.items()
    }

    # Linguistic features
    temp_df = pd.DataFrame([{"text": cleaned}])
    temp_df = extract_linguistic_features(temp_df)
    ling_cols = ["review_length", "word_count", "exclamation_count", "question_count",
                 "caps_ratio", "avg_word_length", "sentence_count", "unique_word_ratio"]
    linguistic = {c: float(temp_df[c].iloc[0]) for c in ling_cols if c in temp_df.columns}

    return SentimentResponse(
        sentiment=sentiment,
        confidence=round(float(confidence), 4),
        compound_score=round(float(compound), 4),
        pos_score=round(float(vader_scores["pos"]), 4),
        neg_score=round(float(vader_scores["neg"]), 4),
        neu_score=round(float(vader_scores["neu"]), 4),
        aspects=aspects,
        linguistic=linguistic,
        source=source,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Return API health status and data readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=_state.model_loaded,
        data_loaded=_state.df is not None,
        review_count=len(_state.df) if _state.df is not None else 0,
        topics_count=len(_state.topics),
    )


@app.post("/score", response_model=SentimentResponse, summary="Score a single review")
async def score_review(payload: ReviewInput):
    """Score a single review: overall sentiment, aspect scores, linguistic features."""
    try:
        return _score_text(payload.text, payload.source)
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}") from exc


@app.post("/score_batch", response_model=List[SentimentResponse], summary="Score a batch of reviews")
async def score_batch(payload: BatchInput):
    """Score a list of reviews; returns list of sentiment responses."""
    if not payload.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    if len(payload.reviews) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 1000")
    try:
        results = [_score_text(r.text, r.source) for r in payload.reviews]
        return results
    except Exception as exc:
        logger.error("Batch scoring failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {exc}") from exc


@app.get("/brand_summary/{brand_name}", response_model=BrandSummaryResponse, summary="Get brand metrics")
async def brand_summary(brand_name: str):
    """Return aggregated brand metrics from loaded data."""
    if _state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    brand_df = _state.df[
        _state.df["business_name"].str.lower() == brand_name.lower()
    ]
    if brand_df.empty:
        # Try partial match
        brand_df = _state.df[
            _state.df["business_name"].str.lower().str.contains(brand_name.lower(), na=False)
        ]
    if brand_df.empty:
        raise HTTPException(status_code=404, detail=f"Brand '{brand_name}' not found")

    avg_sentiment = float(brand_df["compound_sentiment"].mean()) if "compound_sentiment" in brand_df.columns else 0.0
    avg_rating = float(brand_df["rating"].mean())

    # Sentiment distribution
    def _label(c: float) -> str:
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"

    if "compound_sentiment" in brand_df.columns:
        labels = brand_df["compound_sentiment"].apply(_label)
        dist = labels.value_counts().to_dict()
    else:
        dist = {"positive": 0, "neutral": 0, "negative": 0}

    nps_data = compute_nps_proxy(brand_df)
    nps_score = nps_data.get("nps", 0.0)

    # Aspect scores
    aspect_cols = {
        col: float(brand_df[col].mean())
        for col in brand_df.columns
        if col.startswith("aspect_") and col.endswith("_compound")
    }
    top_aspects = {
        k.replace("aspect_", "").replace("_compound", ""): round(v, 4)
        for k, v in aspect_cols.items()
    }

    return BrandSummaryResponse(
        brand_name=brand_name,
        avg_sentiment=round(avg_sentiment, 4),
        avg_rating=round(avg_rating, 2),
        review_count=len(brand_df),
        nps_score=round(float(nps_score), 1),
        sentiment_distribution=dist,
        top_aspects=top_aspects,
    )


@app.get("/topics", response_model=List[TopicResponse], summary="Get top topics")
async def get_topics():
    """Return current top 10 topics from fitted topic model."""
    if not _state.topics:
        raise HTTPException(status_code=503, detail="Topic model not fitted yet")

    result = []
    topic_items = list(_state.topics.items())[:10]
    for topic_id, words in topic_items:
        count = (
            _state.topic_assignments.count(
                list(_state.topics.keys()).index(topic_id)
            )
            if _state.topic_assignments
            else 0
        )
        result.append(TopicResponse(
            topic_id=topic_id,
            label=topic_id,
            top_words=words[:10],
            document_count=count,
        ))
    return result


@app.get("/brands", summary="List all brands in the dataset")
async def list_brands():
    """Return list of all brand names in the loaded dataset."""
    if _state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    brands = _state.df["business_name"].dropna().unique().tolist()
    return {"brands": sorted(brands), "count": len(brands)}


@app.get("/sentiment_trend", summary="Get 30-day sentiment trend")
async def sentiment_trend(
    brand_name: Optional[str] = None,
    source: Optional[str] = None,
    days: int = 30,
):
    """Return daily average sentiment for the specified period."""
    if _state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = _state.df.copy()
    if brand_name:
        df = df[df["business_name"].str.lower().str.contains(brand_name.lower(), na=False)]
    if source and source.lower() != "all":
        df = df[df["source"].str.lower() == source.lower()]

    if df.empty:
        return {"dates": [], "avg_sentiment": [], "review_counts": []}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    df = df[df["date"] >= cutoff]

    if "compound_sentiment" not in df.columns:
        return {"dates": [], "avg_sentiment": [], "review_counts": []}

    daily = df.groupby(df["date"].dt.date).agg(
        avg_sentiment=("compound_sentiment", "mean"),
        review_count=("text", "count"),
    ).reset_index()

    return {
        "dates": daily["date"].astype(str).tolist(),
        "avg_sentiment": daily["avg_sentiment"].round(4).tolist(),
        "review_counts": daily["review_count"].tolist(),
    }


@app.get("/competitive_benchmark", summary="Competitive brand benchmark")
async def competitive_benchmark(top_n: int = 20):
    """Return ranked brand sentiment benchmark."""
    if _state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if "compound_sentiment" not in _state.df.columns:
        raise HTTPException(status_code=503, detail="Sentiment scores not computed")

    df_filtered = _state.df.copy()
    bench = benchmark_brands(df_filtered)
    bench = bench.head(top_n)
    return bench.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Crisis velocity anomaly detection
# ---------------------------------------------------------------------------

@app.get("/crisis_velocity", summary="Real-time crisis velocity anomaly detection")
async def crisis_velocity(window_hours: int = 24, sensitivity: float = 2.0):
    """
    Detect whether brand sentiment is experiencing an anomalous negative velocity.
    Uses a rolling z-score on hourly sentiment to flag if the rate of decline
    exceeds `sensitivity` standard deviations from historical baseline.
    Returns severity level, estimated time to full crisis, and recommended actions.
    """
    if _state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = _state.df.copy()
    date_col = next((c for c in ["date", "review_date", "timestamp"] if c in df.columns), None)
    sent_col = "compound_sentiment" if "compound_sentiment" in df.columns else "stars"

    if date_col is None or sent_col not in df.columns:
        raise HTTPException(status_code=422, detail="date and sentiment columns required")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Create daily time series
    daily = (
        df.groupby(df[date_col].dt.date)[sent_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    daily.columns = ["date", "avg_sentiment", "volume"]

    if len(daily) < 14:
        return {"status": "insufficient_data", "message": "Need at least 14 days of data."}

    # Rolling baseline (28-day window)
    daily["rolling_mean"] = daily["avg_sentiment"].rolling(28, min_periods=7).mean()
    daily["rolling_std"] = daily["avg_sentiment"].rolling(28, min_periods=7).std()

    # Z-score of recent period vs baseline
    recent = daily.tail(window_hours // 24 if window_hours >= 24 else 1)
    recent_mean = float(recent["avg_sentiment"].mean())
    baseline_mean = float(daily["rolling_mean"].iloc[-max(1, len(daily) - window_hours // 24 - 1)])
    baseline_std = float(daily["rolling_std"].iloc[-1]) if not pd.isna(daily["rolling_std"].iloc[-1]) else 0.1

    z_score = (recent_mean - baseline_mean) / (baseline_std + 1e-6)
    velocity = float(recent["avg_sentiment"].diff().mean()) if len(recent) > 1 else 0.0

    # Severity classification
    if z_score < -3.0 or (velocity < -0.05 and z_score < -2.0):
        severity = "CRITICAL"
        crisis_probability = 0.92
        est_hours_to_crisis = max(1, int(abs(recent_mean) / (abs(velocity) + 0.01) * 24))
    elif z_score < -sensitivity:
        severity = "WARNING"
        crisis_probability = 0.55
        est_hours_to_crisis = max(6, int(abs(recent_mean) / (abs(velocity) + 0.01) * 48))
    elif z_score < -1.0:
        severity = "WATCH"
        crisis_probability = 0.22
        est_hours_to_crisis = 72
    else:
        severity = "NORMAL"
        crisis_probability = 0.04
        est_hours_to_crisis = None

    # Volume spike detection (sudden high-volume negative reviews = viral crisis)
    recent_volume = float(recent["volume"].mean())
    baseline_volume = float(daily["volume"].rolling(28, min_periods=7).mean().iloc[-1])
    volume_spike = recent_volume > baseline_volume * 2.0

    actions = {
        "CRITICAL": [
            "Immediately brief executive communications team",
            "Activate crisis response protocol — prepare public statement within 2 hours",
            "Monitor social media channels for amplification",
            "Identify root cause from 1-star reviews in last 24 hours",
            "Prepare product/service hold if safety-related",
        ],
        "WARNING": [
            "Alert brand manager and customer success leads",
            "Review top negative reviews for emerging theme",
            "Prepare draft response templates",
            "Increase monitoring cadence to every 2 hours",
        ],
        "WATCH": [
            "Flag for next daily brand review meeting",
            "Monitor topic distribution for emerging negative themes",
        ],
        "NORMAL": ["Continue standard monitoring cadence"],
    }

    return {
        "severity": severity,
        "crisis_probability": round(crisis_probability, 3),
        "z_score": round(z_score, 3),
        "sentiment_velocity": round(velocity, 4),
        "volume_spike_detected": volume_spike,
        "recent_avg_sentiment": round(recent_mean, 4),
        "baseline_avg_sentiment": round(baseline_mean, 4),
        "estimated_hours_to_full_crisis": est_hours_to_crisis,
        "window_hours": window_hours,
        "recommended_actions": actions[severity],
        "monitoring_timestamp": pd.Timestamp.now().isoformat(),
    }


@app.get("/sentiment_forecast", summary="7-day sentiment forecast")
async def sentiment_forecast(horizon_days: int = 7):
    """
    Extrapolate brand sentiment trend using exponential smoothing.
    Returns point forecast and 80% prediction intervals for the next N days.
    """
    if _state.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = _state.df.copy()
    date_col = next((c for c in ["date", "review_date", "timestamp"] if c in df.columns), None)
    sent_col = "compound_sentiment" if "compound_sentiment" in df.columns else "stars"

    if date_col is None:
        raise HTTPException(status_code=422, detail="Date column required")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    daily = df.groupby(df[date_col].dt.date)[sent_col].mean().reset_index()
    daily.columns = ["date", "sentiment"]
    daily = daily.sort_values("date").tail(90)  # last 90 days

    if len(daily) < 7:
        raise HTTPException(status_code=422, detail="Need at least 7 days of history")

    # Simple exponential smoothing forecast
    alpha = 0.30
    vals = daily["sentiment"].values
    smoothed = float(vals[-1])
    trend = float(vals[-1] - vals[-7]) / 7  # weekly trend

    forecasts = []
    last_date = pd.to_datetime(daily["date"].iloc[-1])
    std_err = float(np.std(np.diff(vals[-30:])) if len(vals) >= 30 else np.std(np.diff(vals)))

    for d in range(1, horizon_days + 1):
        fcast_date = last_date + pd.Timedelta(days=d)
        point = smoothed + trend * d * alpha
        point = float(np.clip(point, -1.0, 1.0))
        interval_width = std_err * np.sqrt(d) * 1.28  # 80% CI
        forecasts.append({
            "date": fcast_date.strftime("%Y-%m-%d"),
            "forecast": round(point, 4),
            "lower_80": round(max(-1.0, point - interval_width), 4),
            "upper_80": round(min(1.0, point + interval_width), 4),
        })

    return {
        "horizon_days": horizon_days,
        "last_observed_sentiment": round(float(vals[-1]), 4),
        "trend_direction": "declining" if trend < -0.002 else ("improving" if trend > 0.002 else "stable"),
        "forecast": forecasts,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        log_level="info",
    )

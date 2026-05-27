"""
FastAPI Application
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Endpoints:
  POST /score_transaction   — Score a single transaction
  POST /score_batch         — Score a batch of transactions in parallel
  GET  /model_performance   — Retrieve stored evaluation metrics
  GET  /drift_report        — Latest drift analysis
  GET  /health              — Service health + uptime
  POST /explain/{tid}       — Detailed SHAP explanation for a stored transaction

Target latency: <50 ms per transaction.

Note: For production add-ons:
  - Rate limiting: pip install slowapi && wrap app with Limiter
  - Auth: pip install python-jose[cryptography] fastapi-users
  - TLS: terminate at nginx/ALB reverse proxy
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import load_artifacts, ensemble_predict_proba
from src.features import build_feature_set
from src.explainability import (
    explain_prediction,
    generate_adverse_action_reasons,
    compute_shap_values,
    get_top_features,
)
from src.monitoring import generate_drift_report, compute_feature_drift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_api")

# ─────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────

class AppState:
    """Global application state loaded at startup."""
    model: Any = None
    threshold: float = 0.5
    feature_cols: List[str] = []
    eval_metrics: Dict[str, Any] = {}
    drift_report: Dict[str, Any] = {}
    start_time: float = time.time()
    model_loaded: bool = False
    scored_today: int = 0
    recent_scores: List[float] = []
    transaction_cache: Dict[str, Dict[str, Any]] = {}  # tid → explanation

app_state = AppState()
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk & Fraud Detection API",
    description=(
        "Real-time fraud scoring with SHAP explainability for FCRA compliance. "
        "Targets JPMorgan Chase, Bank of America, Capital One, American Express."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Middleware: Latency Logging
# ─────────────────────────────────────────────

@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    if latency_ms > 50 and request.url.path in ("/score_transaction", "/score_batch"):
        logger.warning(f"LATENCY ALERT: {request.url.path} took {latency_ms:.1f}ms (target <50ms)")
    else:
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({latency_ms:.1f}ms)")
    return response


# ─────────────────────────────────────────────
# Pydantic v2 Models
# ─────────────────────────────────────────────

class TransactionInput(BaseModel):
    """Single transaction input model — all fields optional with sensible defaults."""
    TransactionID: Optional[str] = Field(default=None, description="Unique transaction identifier")
    TransactionAmt: float = Field(default=50.0, ge=0.0, description="Transaction amount in USD")
    TransactionDT: int = Field(default=86400, ge=0, description="Seconds from reference epoch")
    ProductCD: str = Field(default="W", description="Product code")
    card1: float = Field(default=10000.0, description="Card feature 1")
    card2: Optional[float] = Field(default=321.0, description="Card feature 2")
    card3: Optional[float] = Field(default=150.0, description="Card feature 3")
    card4: Optional[str] = Field(default="visa", description="Card type")
    card5: Optional[float] = Field(default=226.0, description="Card feature 5")
    card6: Optional[str] = Field(default="debit", description="Card category")
    addr1: Optional[float] = Field(default=299.0, description="Address 1 (zip code prefix)")
    addr2: Optional[float] = Field(default=87.0, description="Address 2 (country code)")
    dist1: Optional[float] = Field(default=None, description="Distance 1")
    dist2: Optional[float] = Field(default=None, description="Distance 2")
    P_emaildomain: Optional[str] = Field(default="gmail.com", description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(default="gmail.com", description="Recipient email domain")
    C1: Optional[float] = Field(default=1.0)
    C2: Optional[float] = Field(default=1.0)
    C3: Optional[float] = Field(default=0.0)
    C4: Optional[float] = Field(default=0.0)
    C5: Optional[float] = Field(default=0.0)
    C6: Optional[float] = Field(default=1.0)
    C7: Optional[float] = Field(default=0.0)
    C8: Optional[float] = Field(default=0.0)
    C9: Optional[float] = Field(default=1.0)
    C10: Optional[float] = Field(default=0.0)
    C11: Optional[float] = Field(default=1.0)
    C12: Optional[float] = Field(default=0.0)
    C13: Optional[float] = Field(default=1.0)
    C14: Optional[float] = Field(default=1.0)
    D1: Optional[float] = Field(default=14.0)
    D2: Optional[float] = Field(default=None)
    D3: Optional[float] = Field(default=None)
    D4: Optional[float] = Field(default=None)
    D5: Optional[float] = Field(default=None)
    M1: Optional[str] = Field(default="T")
    M2: Optional[str] = Field(default="T")
    M3: Optional[str] = Field(default="T")
    M4: Optional[str] = Field(default=None)
    M5: Optional[str] = Field(default="F")
    M6: Optional[str] = Field(default="T")
    M7: Optional[str] = Field(default=None)
    M8: Optional[str] = Field(default=None)
    M9: Optional[str] = Field(default=None)
    # Allow extra V-feature fields (V1–V339)
    model_config = {"extra": "allow"}

    @field_validator("TransactionAmt")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("TransactionAmt must be non-negative")
        return v


class SHAPFactor(BaseModel):
    feature: str
    value: float
    shap_value: float
    description: str


class SHAPExplanation(BaseModel):
    top_risk_factors: List[SHAPFactor]
    top_safe_factors: List[SHAPFactor]


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(description="Probability of fraud [0, 1]")
    risk_level: str = Field(description="'low' | 'medium' | 'high'")
    decision: str = Field(description="'approve' | 'review' | 'decline'")
    shap_explanation: SHAPExplanation
    adverse_action_reasons: List[str]
    transaction_id: str
    latency_ms: float


class BatchScoreRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    total_latency_ms: float
    n_transactions: int
    fraud_count: int
    fraud_rate: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    scored_today: int
    model_version: str = "1.0.0"
    timestamp: str


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load model artifacts at application startup."""
    model_dir = _PROJECT_ROOT / "data" / "models"
    logger.info(f"Loading model artifacts from {model_dir}")

    try:
        model, threshold, feature_cols = load_artifacts(model_dir, model_name="fraud_model")
        app_state.model = model
        app_state.threshold = threshold
        app_state.feature_cols = feature_cols
        app_state.model_loaded = True
        logger.info(
            f"Model loaded successfully | threshold={threshold:.4f} | "
            f"n_features={len(feature_cols)}"
        )
    except FileNotFoundError:
        logger.warning(
            "Model artifacts not found. Endpoints requiring the model will return 503. "
            "Run the training pipeline first: python src/train.py"
        )
        app_state.model_loaded = False

    app_state.start_time = time.time()
    logger.info("Fraud Detection API started")


@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=False)
    logger.info("Fraud Detection API stopped")


# ─────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────

def _transaction_to_df(tx: TransactionInput) -> pd.DataFrame:
    """Convert a TransactionInput pydantic model to a single-row DataFrame."""
    data = tx.model_dump()
    # Remove TransactionID from features
    data.pop("TransactionID", None)
    return pd.DataFrame([data])


def _score_single_transaction(
    tx: TransactionInput,
) -> tuple[ScoreResponse, float]:
    """
    Core scoring logic for a single transaction.

    Returns (ScoreResponse, latency_ms).
    """
    t_start = time.perf_counter()
    tid = tx.TransactionID or str(uuid.uuid4())

    # Convert to DataFrame & run feature engineering
    df = _transaction_to_df(tx)

    # Minimal feature engineering (no velocity for single tx — no card history)
    from src.features import (
        engineer_time_features,
        engineer_email_features,
        engineer_amount_features,
        encode_categoricals,
        handle_missing,
    )
    df = engineer_time_features(df)
    df = engineer_email_features(df)
    df = engineer_amount_features(df)

    cat_cols = [c for c in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] if c in df.columns]
    df = encode_categoricals(df, cat_cols, target_col=None)
    df = handle_missing(df)

    # Align to training feature columns
    feature_cols = app_state.feature_cols
    X = pd.DataFrame(columns=feature_cols)
    X = pd.concat([X, df.reindex(columns=feature_cols, fill_value=-999)], ignore_index=True)
    X = X.fillna(-999).astype(np.float32)

    # Score
    model = app_state.model
    threshold = app_state.threshold

    if isinstance(model, dict) and "xgb_models" in model:
        proba = float(ensemble_predict_proba(model, X)[0])
    else:
        proba = float(model.predict_proba(X)[0, 1])

    # Risk level & decision
    if proba < 0.3:
        risk_level = "low"
        decision = "approve"
    elif proba < 0.7:
        risk_level = "medium"
        decision = "review"
    else:
        risk_level = "high"
        decision = "decline"

    # SHAP explanation
    try:
        expl = explain_prediction(model, X, feature_cols)
        adverse_reasons = generate_adverse_action_reasons(expl, n=3)
        top_risk = [
            SHAPFactor(
                feature=f["feature"],
                value=float(f["value"]) if f["value"] is not None and not (isinstance(f["value"], float) and np.isnan(f["value"])) else 0.0,
                shap_value=float(f["shap_value"]),
                description=f["description"],
            )
            for f in expl["top_positive_factors"]
        ]
        top_safe = [
            SHAPFactor(
                feature=f["feature"],
                value=float(f["value"]) if f["value"] is not None and not (isinstance(f["value"], float) and np.isnan(f["value"])) else 0.0,
                shap_value=float(f["shap_value"]),
                description=f["description"],
            )
            for f in expl["top_negative_factors"]
        ]
    except Exception as shap_err:
        logger.warning(f"SHAP explanation failed: {shap_err}")
        top_risk = []
        top_safe = []
        adverse_reasons = ["Unable to generate explanation at this time."]

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Cache for /explain endpoint
    app_state.transaction_cache[tid] = {
        "transaction": tx.model_dump(),
        "probability": proba,
        "risk_level": risk_level,
        "top_risk_factors": [f.model_dump() for f in top_risk],
        "top_safe_factors": [f.model_dump() for f in top_safe],
        "adverse_action_reasons": adverse_reasons,
    }

    # Update counters
    app_state.scored_today += 1
    app_state.recent_scores.append(proba)
    if len(app_state.recent_scores) > 10_000:
        app_state.recent_scores = app_state.recent_scores[-10_000:]

    response = ScoreResponse(
        fraud_probability=round(proba, 6),
        risk_level=risk_level,
        decision=decision,
        shap_explanation=SHAPExplanation(top_risk_factors=top_risk, top_safe_factors=top_safe),
        adverse_action_reasons=adverse_reasons,
        transaction_id=tid,
        latency_ms=round(latency_ms, 2),
    )
    return response, latency_ms


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Service health status and uptime."""
    return HealthResponse(
        status="healthy" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        uptime_seconds=round(time.time() - app_state.start_time, 1),
        scored_today=app_state.scored_today,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/score_transaction", response_model=ScoreResponse, tags=["Scoring"])
async def score_transaction(tx: TransactionInput):
    """
    Score a single transaction for fraud probability.

    Returns fraud probability, risk level, decision, SHAP explanation,
    and FCRA-compliant adverse action reasons.
    Target latency: <50 ms.
    """
    if not app_state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    loop = asyncio.get_event_loop()
    response, _latency = await loop.run_in_executor(executor, _score_single_transaction, tx)
    return response


@app.post("/score_batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_batch(request: BatchScoreRequest):
    """
    Score a batch of transactions in parallel using a thread pool.

    Returns individual scores plus aggregate statistics.
    """
    if not app_state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    if not request.transactions:
        raise HTTPException(status_code=400, detail="transactions list is empty")

    if len(request.transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 1000 transactions per request.",
        )

    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(executor, _score_single_transaction, tx)
        for tx in request.transactions
    ]
    scored = await asyncio.gather(*tasks)

    results = [r for r, _ in scored]
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    fraud_count = sum(1 for r in results if r.decision == "decline")
    fraud_rate = fraud_count / len(results)

    return BatchScoreResponse(
        results=results,
        total_latency_ms=round(total_latency_ms, 2),
        n_transactions=len(results),
        fraud_count=fraud_count,
        fraud_rate=round(fraud_rate, 4),
    )


@app.get("/model_performance", tags=["Analytics"])
async def get_model_performance():
    """
    Retrieve stored model evaluation metrics from the last training run.

    Metrics include: AUC-ROC, AUC-PR, F1, Precision, Recall, Confusion Matrix,
    optimal classification threshold.
    """
    model_dir = _PROJECT_ROOT / "data" / "models"
    metrics_path = model_dir / "eval_metrics.json"

    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        return {"status": "ok", "metrics": metrics}
    else:
        # Return placeholder if no metrics file exists yet
        return {
            "status": "metrics_not_available",
            "detail": "Train the model first to generate evaluation metrics.",
            "metrics": {},
        }


@app.get("/drift_report", tags=["Monitoring"])
async def get_drift_report():
    """
    Return the latest feature and model performance drift analysis.

    PSI thresholds: <0.10 stable, 0.10–0.20 warning, >0.20 critical.
    """
    if app_state.drift_report:
        return {"status": "ok", "report": app_state.drift_report}
    return {
        "status": "no_drift_report",
        "detail": "Drift report not yet generated. Call POST /admin/refresh_drift to compute.",
        "report": {},
    }


@app.post("/explain/{transaction_id}", tags=["Explainability"])
async def explain_transaction(transaction_id: str):
    """
    Retrieve detailed SHAP explanation for a previously scored transaction.

    Explanation includes:
    - Top fraud-increasing factors (positive SHAP values)
    - Top fraud-decreasing factors (negative SHAP values)
    - FCRA adverse action reason codes
    - Fraud probability and risk level
    """
    cache = app_state.transaction_cache.get(transaction_id)
    if cache is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction '{transaction_id}' not found in explanation cache. "
                "Score the transaction first via POST /score_transaction."
            ),
        )
    return {
        "transaction_id": transaction_id,
        "fraud_probability": cache["probability"],
        "risk_level": cache["risk_level"],
        "top_risk_factors": cache["top_risk_factors"],
        "top_safe_factors": cache["top_safe_factors"],
        "adverse_action_reasons": cache["adverse_action_reasons"],
        "fcra_notice": (
            "This explanation is provided under the Fair Credit Reporting Act (FCRA) "
            "and Equal Credit Opportunity Act (ECOA). You have the right to know why "
            "adverse action was taken against you."
        ),
    }


# ─────────────────────────────────────────────
# Admin Endpoints (internal / ops)
# ─────────────────────────────────────────────

@app.post("/admin/reload_model", tags=["Admin"], include_in_schema=False)
async def reload_model():
    """Hot-reload model artifacts without restarting the service."""
    model_dir = _PROJECT_ROOT / "data" / "models"
    try:
        model, threshold, feature_cols = load_artifacts(model_dir, model_name="fraud_model")
        app_state.model = model
        app_state.threshold = threshold
        app_state.feature_cols = feature_cols
        app_state.model_loaded = True
        logger.info("Model reloaded successfully")
        return {"status": "ok", "message": "Model reloaded", "n_features": len(feature_cols)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )

"""
FastAPI service for Customer Lifetime Value & Intelligent Retention Platform.
Endpoints: score_user, score_cohort, survival_curves, campaign_roi,
           model_performance, portfolio_summary
"""

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, generate_synthetic_streaming_data
from src.features import build_feature_pipeline
from src.clv_model import (
    fit_bgnbd_model, predict_future_purchases, fit_gamma_gamma,
    compute_clv, segment_by_clv, compute_simple_clv,
)
from src.survival_analysis import (
    prepare_survival_data, fit_kaplan_meier, plot_km_curves,
    predict_churn_probability_at_time,
)
from src.uplift_model import (
    compute_uplift_scores, segment_by_uplift, compute_campaign_roi as _compute_roi,
)
from src.retention_messaging import generate_retention_message, personalize_offer
from src.explainability import explain_churn_risk, generate_user_narrative
from src.models import train_churn_classifier, evaluate_model, load_artifacts, save_artifacts

app = FastAPI(
    title="CLV & Retention Intelligence API",
    description="Customer Lifetime Value modeling, churn prediction, causal uplift, and retention messaging.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "df": None,
    "X": None,
    "y": None,
    "feature_names": None,
    "churn_model": None,
    "uplift_models": None,
    "clv_series": None,
    "clv_segments": None,
    "uplift_scores": None,
    "uplift_segments": None,
    "survival_data": None,
    "km_results": None,
    "model_metrics": None,
    "initialized": False,
}


def _initialize() -> None:
    """Load/train all models at startup."""
    if _state["initialized"]:
        return

    logger.info("Initializing CLV & Retention API...")
    data_dir = PROJECT_ROOT / "data" / "raw"
    project_root = PROJECT_ROOT

    df = load_data(str(data_dir), str(project_root))
    _state["df"] = df

    # Feature engineering
    from sklearn.model_selection import train_test_split
    X, y, feat_names = build_feature_pipeline(df)
    _state["X"] = X
    _state["y"] = y
    _state["feature_names"] = feat_names

    # Train churn model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    churn_model, _ = train_churn_classifier(X_train, y_train, eval_set=(X_test, y_test))
    _state["churn_model"] = churn_model

    # Model metrics
    metrics = evaluate_model(churn_model, X_test, y_test)
    _state["model_metrics"] = metrics

    # CLV
    try:
        clv = compute_simple_clv(
            df,
            churn_proba=pd.Series(churn_model.predict_proba(X)[:, 1], index=X.index),
        )
    except Exception:
        clv = pd.Series(np.random.uniform(50, 500, len(df)), index=df.index)
    _state["clv_series"] = clv
    _state["clv_segments"] = segment_by_clv(clv)

    # Uplift scores (simulate treatment = sent_offer flag)
    rng = np.random.default_rng(42)
    treatment = pd.Series(
        rng.integers(0, 2, len(df)), index=df.index, name="treatment"
    )
    try:
        from src.uplift_model import train_uplift_model
        mt, mc, _ = train_uplift_model(X, y, treatment)
        uplift_scores = compute_uplift_scores((mt, mc), X)
        _state["uplift_models"] = (mt, mc)
    except Exception:
        uplift_scores = pd.Series(rng.uniform(-0.1, 0.3, len(df)), index=df.index)

    _state["uplift_scores"] = uplift_scores
    _state["uplift_segments"] = segment_by_uplift(uplift_scores)

    # Survival analysis
    surv_df = prepare_survival_data(df)
    _state["survival_data"] = surv_df
    km_results = fit_kaplan_meier(
        surv_df["T"], surv_df["E"],
        groups=_state["clv_segments"].reindex(surv_df.index),
    )
    _state["km_results"] = km_results

    _state["initialized"] = True
    logger.info("API initialization complete.")


@app.on_event("startup")
async def startup() -> None:
    try:
        _initialize()
    except Exception as exc:
        logger.error(f"Startup initialization failed: {exc}")


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class UserFeatures(BaseModel):
    user_id: str = Field(default="unknown")
    tenure_days: float = Field(default=180.0)
    days_since_last_login: float = Field(default=15.0)
    avg_completion_rate: float = Field(default=0.65)
    total_secs: float = Field(default=500000.0)
    plan_list_price: float = Field(default=148.0)
    payment_method_id: float = Field(default=10.0)
    num_25: float = Field(default=50.0)
    num_50: float = Field(default=40.0)
    num_75: float = Field(default=30.0)
    num_985: float = Field(default=20.0)
    num_100: float = Field(default=60.0)
    num_unq: float = Field(default=100.0)
    bd: float = Field(default=30.0)
    registered_via: float = Field(default=4.0)


class CohortRequest(BaseModel):
    users: list[UserFeatures]


class SiteConfigRequest(BaseModel):
    zones: dict[str, list[str]] = Field(default={})


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.post("/score_user")
async def score_user(user: UserFeatures) -> dict[str, Any]:
    """Score a single user for churn probability, CLV, uplift, and retention message."""
    if not _state["initialized"]:
        _initialize()

    t0 = time.time()
    churn_model = _state["churn_model"]
    feature_names = _state["feature_names"]

    if churn_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Build user DataFrame
    user_dict = user.model_dump()
    user_id = user_dict.pop("user_id", "unknown")
    single_df = pd.DataFrame([user_dict])

    from src.features import (
        engineer_rfm_features, engineer_engagement_features,
        engineer_payment_features, engineer_lifecycle_features,
        engineer_behavioral_signals,
    )
    single_df = engineer_rfm_features(single_df)
    single_df = engineer_engagement_features(single_df)
    single_df = engineer_payment_features(single_df)
    single_df = engineer_lifecycle_features(single_df)
    single_df = engineer_behavioral_signals(single_df)

    # Align columns
    for col in feature_names:
        if col not in single_df.columns:
            single_df[col] = 0.0
    X_user = single_df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)

    churn_prob = float(churn_model.predict_proba(X_user)[0, 1])
    risk_level = (
        "CRITICAL" if churn_prob >= 0.75
        else "HIGH" if churn_prob >= 0.55
        else "MEDIUM" if churn_prob >= 0.35
        else "LOW"
    )

    clv = float(compute_simple_clv(
        single_df,
        churn_proba=pd.Series([churn_prob]),
    ).iloc[0])

    clv_segment = segment_by_clv(pd.Series([clv])).iloc[0]

    # Uplift
    if _state["uplift_models"] is not None:
        uplift_score = float(compute_uplift_scores(_state["uplift_models"], X_user).iloc[0])
    else:
        uplift_score = 0.05
    uplift_seg = segment_by_uplift(pd.Series([uplift_score])).iloc[0]

    offer = personalize_offer(clv_segment, uplift_seg, churn_prob)
    tenure = int(single_df.get("tenure_days", pd.Series([180])).iloc[0])

    retention_msg = generate_retention_message(
        user_segment=uplift_seg,
        clv_tier=clv_segment,
        churn_risk=churn_prob,
        tenure_days=tenure,
        plan_name="Premium Streaming",
        days_inactive=int(user.days_since_last_login),
    )

    shap_exp = explain_churn_risk(churn_model, X_user, feature_names)

    lifecycle = single_df.get("lifecycle_stage", pd.Series(["mature"])).iloc[0]
    narrative = generate_user_narrative(
        user_id=user_id,
        churn_prob=churn_prob,
        clv=clv,
        uplift_segment=uplift_seg,
        shap_explanation=shap_exp,
        tenure_days=tenure,
        lifecycle_stage=str(lifecycle),
        recommended_action=offer["action"],
    )

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return {
        "user_id": user_id,
        "churn_probability": round(churn_prob, 4),
        "churn_risk_level": risk_level,
        "clv_estimate": round(clv, 2),
        "clv_segment": clv_segment,
        "uplift_score": round(uplift_score, 4),
        "uplift_segment": uplift_seg,
        "recommended_action": offer["action"],
        "retention_message": retention_msg,
        "shap_explanation": shap_exp,
        "narrative": narrative,
        "processing_time_ms": elapsed_ms,
    }


@app.post("/score_cohort")
async def score_cohort(cohort: CohortRequest) -> dict[str, Any]:
    """Score a batch of users and return cohort summary + individual scores."""
    if not _state["initialized"]:
        _initialize()

    results = []
    for user in cohort.users:
        try:
            r = await score_user(user)
            results.append(r)
        except Exception as exc:
            results.append({"user_id": user.user_id, "error": str(exc)})

    churn_probs = [r.get("churn_probability", 0) for r in results if "churn_probability" in r]
    clv_vals = [r.get("clv_estimate", 0) for r in results if "clv_estimate" in r]
    risk_counts: dict[str, int] = {}
    for r in results:
        rl = r.get("churn_risk_level", "UNKNOWN")
        risk_counts[rl] = risk_counts.get(rl, 0) + 1

    uplift_counts: dict[str, int] = {}
    for r in results:
        us = r.get("uplift_segment", "Unknown")
        uplift_counts[us] = uplift_counts.get(us, 0) + 1

    return {
        "n_users": len(results),
        "cohort_summary": {
            "avg_churn_probability": round(np.mean(churn_probs), 4) if churn_probs else 0,
            "avg_clv": round(np.mean(clv_vals), 2) if clv_vals else 0,
            "total_at_risk_clv": round(
                sum(c for c, cp in zip(clv_vals, churn_probs) if cp >= 0.5), 2
            ),
            "risk_level_distribution": risk_counts,
            "uplift_segment_distribution": uplift_counts,
        },
        "individual_scores": results,
    }


@app.get("/survival_curves")
async def survival_curves() -> dict[str, Any]:
    """Return Kaplan-Meier survival curve data for each user segment."""
    if not _state["initialized"]:
        _initialize()

    km_results = _state.get("km_results", {})
    if not km_results:
        raise HTTPException(status_code=503, detail="Survival models not yet computed")

    output: dict[str, Any] = {}
    for group, result in km_results.items():
        sf = result["survival_function"]
        output[group] = {
            "timeline": [float(t) for t in result["timeline"].tolist()],
            "survival_probability": [float(v) for v in (sf.values if hasattr(sf, "values") else sf).tolist()],
            "ci_upper": [float(v) for v in result["ci_upper"].values.tolist()],
            "ci_lower": [float(v) for v in result["ci_lower"].values.tolist()],
            "median_survival_days": round(result["median_survival"], 1),
            "n_users": result["n"],
            "event_count": result["event_count"],
        }
    return {"survival_curves": output, "n_groups": len(output)}


@app.get("/campaign_roi_analysis")
async def campaign_roi_analysis() -> dict[str, Any]:
    """Compute expected ROI of retention campaign by intervention type."""
    if not _state["initialized"]:
        _initialize()

    df = _state["df"]
    uplift_scores = _state["uplift_scores"]
    uplift_segments = _state["uplift_segments"]

    if df is None or uplift_scores is None:
        raise HTTPException(status_code=503, detail="Data not initialized")

    df_copy = df.copy()
    if "churn_flag" not in df_copy.columns:
        df_copy["churn_flag"] = 0
    df_copy["uplift_segment"] = uplift_segments.reindex(df_copy.index).fillna("Lost Causes")

    roi_results: dict[str, Any] = {}
    for cost, name in [(2.5, "email_only"), (5.0, "email_sms"), (15.0, "email_sms_call")]:
        roi = _compute_roi(
            df_copy, uplift_scores.reindex(df_copy.index).fillna(0),
            intervention_cost_per_user=cost,
            avg_monthly_revenue=15.0,
        )
        roi_results[name] = roi

    return {
        "campaign_roi_by_intervention": roi_results,
        "recommendation": max(roi_results.items(), key=lambda kv: kv[1]["roi_pct"])[0],
    }


@app.get("/model_performance")
async def model_performance() -> dict[str, Any]:
    """Return churn model evaluation metrics."""
    if not _state["initialized"]:
        _initialize()

    metrics = _state.get("model_metrics", {})
    return {
        "churn_model_metrics": metrics,
        "model_type": type(_state.get("churn_model")).__name__,
        "feature_count": len(_state.get("feature_names", [])),
    }


@app.get("/portfolio_summary")
async def portfolio_summary() -> dict[str, Any]:
    """Return total at-risk revenue, CLV distribution, and segment breakdown."""
    if not _state["initialized"]:
        _initialize()

    df = _state["df"]
    clv = _state["clv_series"]
    clv_segs = _state["clv_segments"]
    uplift_segs = _state["uplift_segments"]
    churn_model = _state["churn_model"]
    X = _state["X"]

    if df is None:
        raise HTTPException(status_code=503, detail="Data not initialized")

    churn_probs = churn_model.predict_proba(X.fillna(0))[:, 1] if churn_model else np.full(len(df), 0.25)
    high_risk_mask = churn_probs >= 0.55
    at_risk_clv = float(clv.reindex(df.index).fillna(0)[high_risk_mask].sum())

    clv_vals = clv.reindex(df.index).fillna(0)

    seg_summary: dict[str, dict] = {}
    for seg in ["Low", "Medium", "High"]:
        seg_mask = clv_segs.reindex(df.index) == seg
        seg_summary[seg] = {
            "n_users": int(seg_mask.sum()),
            "avg_clv": round(float(clv_vals[seg_mask].mean()), 2),
            "total_clv": round(float(clv_vals[seg_mask].sum()), 2),
            "avg_churn_prob": round(float(pd.Series(churn_probs)[seg_mask.values].mean()), 4),
        }

    return {
        "total_users": len(df),
        "at_risk_users": int(high_risk_mask.sum()),
        "at_risk_fraction": round(float(high_risk_mask.mean()), 4),
        "at_risk_clv_usd": round(at_risk_clv, 2),
        "total_portfolio_clv_usd": round(float(clv_vals.sum()), 2),
        "avg_clv_per_user": round(float(clv_vals.mean()), 2),
        "clv_segments": seg_summary,
        "uplift_segment_counts": uplift_segs.value_counts().to_dict() if uplift_segs is not None else {},
        "overall_churn_rate": round(float(np.mean(churn_probs)), 4),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "initialized": str(_state["initialized"])}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

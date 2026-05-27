"""
api/main.py
===========
FastAPI clinical REST API for the Parkinson's Disease Digital Biomarker Platform.

Endpoints:
  POST /analyze_biomarkers       — full multi-modal analysis
  POST /submit_recording         — single recording submission
  GET  /patient_timeline/{id}    — longitudinal biomarker trends
  GET  /cohort_comparison        — patient vs. population distribution
  GET  /model_performance        — clinical validation metrics
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_synthetic_multimodal, load_data
from src.explainability import (
    compute_uncertainty,
    explain_patient_risk,
    generate_clinical_report,
    get_biomarker_clinical_ranges,
)
from src.features import build_multimodal_features, extract_gait_features, extract_tremor_features, extract_voice_features
from src.models import evaluate_classifier, load_artifacts, train_binary_classifier, train_updrs_regressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Parkinson's Disease Digital Biomarker API",
    description=(
        "Multi-modal PD detection using voice, gait and tremor biomarkers. "
        "Intended for research and clinical decision-support only — not a diagnostic device."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "data" / "models"
_classifier = None
_regressor = None
_feature_cols: List[str] = []
_population_df: Optional[pd.DataFrame] = None
_patient_recordings: Dict[str, List[Dict[str, Any]]] = {}
_cv_results: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load or train models on startup."""
    global _classifier, _regressor, _feature_cols, _population_df

    try:
        _classifier, _regressor, _feature_cols = load_artifacts(MODEL_DIR)
        logger.info("Loaded persisted model artifacts from %s", MODEL_DIR)
    except Exception as exc:
        logger.warning("Could not load artifacts (%s) — training on synthetic data.", exc)
        _train_default_models()

    # Load population data for cohort comparison
    _population_df = generate_synthetic_multimodal(n_subjects=200, recordings_per_subject=3)
    logger.info("Population reference dataset loaded (%d records).", len(_population_df))


def _train_default_models() -> None:
    global _classifier, _regressor, _feature_cols

    df = generate_synthetic_multimodal(n_subjects=300, recordings_per_subject=4)
    feat_df = build_multimodal_features(df)
    _feature_cols = feat_df.columns.tolist()

    X = feat_df.fillna(feat_df.median(numeric_only=True))
    y = df["status"]
    y_updrs = df["UPDRS_total"]

    _classifier = train_binary_classifier(X, y, model_type="xgboost")
    _regressor = train_updrs_regressor(X, y_updrs)
    logger.info("Default models trained on synthetic data.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class VoiceFeatures(BaseModel):
    MDVP_Fo: float = Field(..., ge=80, le=260, description="Fundamental frequency (Hz)")
    MDVP_Jitter_pct: float = Field(..., ge=0.0001, le=0.05, description="Jitter (%)")
    MDVP_Shimmer: float = Field(..., ge=0.001, le=0.30, description="Shimmer")
    HNR: float = Field(..., ge=0, le=40, description="Harmonics-to-Noise Ratio (dB)")
    RPDE: float = Field(0.45, ge=0.0, le=1.0)
    DFA: float = Field(0.72, ge=0.0, le=1.0)
    spread1: float = Field(-5.0, ge=-10.0, le=-1.0)
    spread2: float = Field(0.25, ge=0.0, le=0.80)
    D2: float = Field(2.5, ge=0.5, le=6.0)
    PPE: float = Field(0.30, ge=0.0, le=0.90)


class GaitFeatures(BaseModel):
    stride_length: float = Field(..., ge=0.3, le=2.5, description="Stride length (m)")
    cadence: float = Field(..., ge=40, le=200, description="Cadence (steps/min)")
    stride_regularity: float = Field(..., ge=0.0, le=1.0, description="Stride regularity (0–1)")
    gait_asymmetry: float = Field(..., ge=0.0, le=0.5, description="Gait asymmetry ratio")


class TremorFeatures(BaseModel):
    rest_tremor_amplitude: float = Field(..., ge=0.0, le=10.0, description="Rest tremor amplitude (m/s²)")
    tremor_frequency: float = Field(..., ge=0.0, le=25.0, description="Tremor frequency (Hz)")
    tremor_power_ratio: float = Field(..., ge=0.0, le=1.0, description="Tremor power ratio (0–1)")


class BiomarkerAnalysisRequest(BaseModel):
    patient_id: str
    voice_features: VoiceFeatures
    gait_features: GaitFeatures
    tremor_features: TremorFeatures
    age: Optional[float] = Field(None, ge=18, le=100)
    sex: Optional[int] = Field(None, ge=0, le=1)


class RecordingSubmission(BaseModel):
    patient_id: str
    voice_features: Optional[VoiceFeatures] = None
    gait_features: Optional[GaitFeatures] = None
    tremor_features: Optional[TremorFeatures] = None
    recording_index: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Helper: build feature row from request
# ---------------------------------------------------------------------------

def _request_to_features(req: BiomarkerAnalysisRequest) -> pd.DataFrame:
    row = {
        **req.voice_features.dict(),
        **req.gait_features.dict(),
        **req.tremor_features.dict(),
    }
    if req.age is not None:
        row["age"] = req.age
    if req.sex is not None:
        row["sex"] = req.sex
    df_single = pd.DataFrame([row])

    # Extract features using same pipeline as training
    voice_df = extract_voice_features(df_single)
    gait_df = extract_gait_features(df_single)
    tremor_df = extract_tremor_features(df_single)
    feat_df = pd.concat([voice_df, gait_df, tremor_df], axis=1)
    feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]

    # Align to training features
    if _feature_cols:
        for col in _feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[_feature_cols]

    return feat_df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze_biomarkers")
async def analyze_biomarkers(req: BiomarkerAnalysisRequest) -> Dict[str, Any]:
    """Full multi-modal biomarker analysis.

    Returns PD probability, confidence interval, key biomarkers,
    UPDRS estimate, clinical interpretation and referral recommendation.
    """
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    feat_df = _request_to_features(req)

    # Classification
    explanation = explain_patient_risk(_classifier, feat_df, feat_df.columns.tolist())

    # UPDRS regression
    try:
        updrs_pred = float(_regressor.predict(feat_df.fillna(0))[0])
        updrs_pred = round(float(np.clip(updrs_pred, 0, 176)), 1)
    except Exception:
        updrs_pred = None

    # Uncertainty
    uncertainty = compute_uncertainty(_classifier, feat_df)

    # Referral recommendation
    risk = explanation["risk_level"]
    if risk == "High":
        referral = "REFER — Immediate referral to a movement disorder specialist."
    elif risk == "Moderate":
        referral = "MONITOR — Follow-up biomarker assessment in 6 months."
    else:
        referral = "CLEAR — Routine follow-up."

    return {
        "patient_id": req.patient_id,
        "pd_probability": explanation["pd_probability"],
        "risk_level": risk,
        "confidence_interval": {
            "lower": uncertainty["lower_ci"],
            "upper": uncertainty["upper_ci"],
        },
        "key_biomarkers": explanation["key_biomarkers"],
        "updrs_estimate": updrs_pred,
        "clinical_interpretation": explanation["clinical_interpretation"],
        "referral_recommendation": referral,
    }


@app.post("/submit_recording")
async def submit_recording(req: RecordingSubmission) -> Dict[str, Any]:
    """Submit a single recording for a patient; triggers longitudinal update."""
    record: Dict[str, Any] = {"recording_index": req.recording_index}
    if req.voice_features:
        record.update(req.voice_features.dict())
    if req.gait_features:
        record.update(req.gait_features.dict())
    if req.tremor_features:
        record.update(req.tremor_features.dict())

    if req.patient_id not in _patient_recordings:
        _patient_recordings[req.patient_id] = []
    _patient_recordings[req.patient_id].append(record)

    n_recordings = len(_patient_recordings[req.patient_id])
    logger.info("Patient %s — recording %d stored.", req.patient_id, n_recordings)

    return {
        "patient_id": req.patient_id,
        "status": "stored",
        "total_recordings": n_recordings,
        "message": f"Recording {n_recordings} stored. Longitudinal analysis available after 3+ recordings.",
    }


@app.get("/patient_timeline/{patient_id}")
async def patient_timeline(patient_id: str) -> Dict[str, Any]:
    """Return longitudinal biomarker trends for a patient."""
    if patient_id not in _patient_recordings:
        raise HTTPException(status_code=404, detail=f"No recordings found for patient '{patient_id}'.")

    records = _patient_recordings[patient_id]
    df = pd.DataFrame(records).sort_values("recording_index", ignore_index=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "recording_index"]

    trends: Dict[str, Any] = {}
    for col in numeric_cols:
        vals = df[col].dropna().tolist()
        trends[col] = {
            "values": [round(v, 4) for v in vals],
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "trend": "up" if len(vals) > 1 and vals[-1] > vals[0] else "down" if len(vals) > 1 else "stable",
        }

    return {
        "patient_id": patient_id,
        "n_recordings": len(records),
        "recording_indices": df["recording_index"].tolist(),
        "biomarker_trends": trends,
    }


@app.get("/cohort_comparison")
async def cohort_comparison(
    pd_probability: float = Query(..., ge=0.0, le=1.0, description="Patient's PD probability"),
) -> Dict[str, Any]:
    """Compare a patient's risk score to the population distribution."""
    if _population_df is None:
        raise HTTPException(status_code=503, detail="Population data not loaded.")

    if _classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    pop_feat = build_multimodal_features(_population_df)
    if _feature_cols:
        for col in _feature_cols:
            if col not in pop_feat.columns:
                pop_feat[col] = 0.0
        pop_feat = pop_feat[_feature_cols]

    pop_proba = _classifier.predict_proba(pop_feat.fillna(0))[:, 1]
    pd_pop = pop_proba[_population_df["status"].values == 1]
    hc_pop = pop_proba[_population_df["status"].values == 0]

    percentile_overall = float(np.mean(pop_proba <= pd_probability) * 100)
    percentile_pd = float(np.mean(pd_pop <= pd_probability) * 100) if len(pd_pop) > 0 else 50.0

    return {
        "patient_pd_probability": pd_probability,
        "population_mean": round(float(np.mean(pop_proba)), 3),
        "population_std": round(float(np.std(pop_proba)), 3),
        "pd_group_mean": round(float(np.mean(pd_pop)), 3) if len(pd_pop) > 0 else None,
        "healthy_group_mean": round(float(np.mean(hc_pop)), 3) if len(hc_pop) > 0 else None,
        "patient_percentile_overall": round(percentile_overall, 1),
        "patient_percentile_vs_pd_cohort": round(percentile_pd, 1),
        "interpretation": (
            f"The patient's PD probability ({pd_probability:.1%}) is at the "
            f"{percentile_overall:.0f}th percentile of the reference population."
        ),
    }


@app.get("/model_performance")
async def model_performance() -> Dict[str, Any]:
    """Return clinical validation metrics from training evaluation."""
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    # Run a quick evaluation on held-out synthetic data
    eval_df = generate_synthetic_multimodal(n_subjects=100, recordings_per_subject=3, random_state=99)
    feat_df = build_multimodal_features(eval_df)

    if _feature_cols:
        for col in _feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[_feature_cols]

    metrics = evaluate_classifier(_classifier, feat_df.fillna(0), eval_df["status"])

    return {
        "model_type": type(_classifier).__name__,
        "evaluation_set": "synthetic_holdout_300_subjects",
        "metrics": {
            "auc_roc": round(metrics["auc_roc"], 4),
            "sensitivity_recall": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "ppv_precision": round(metrics["ppv"], 4),
            "npv": round(metrics["npv"], 4),
            "f1_score": round(metrics["f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        },
        "clinical_note": (
            "Sensitivity (recall for PD) is the most critical metric — false negatives "
            "carry higher clinical cost than false positives in this screening context."
        ),
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "Parkinsons Disease Digital Biomarker API"}

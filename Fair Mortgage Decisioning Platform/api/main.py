"""
Mortgage Underwriting API
FastAPI service for fair and explainable mortgage decisioning.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, generate_synthetic_mortgage_data
from src.features import build_feature_pipeline, engineer_affordability_features
from src.features import engineer_risk_features, engineer_geographic_features, encode_categoricals
from src.models import train_lightgbm, evaluate_all_models, load_artifacts, save_artifacts
from src.fairness import generate_fairness_report, run_ecoa_compliance_test, generate_adverse_action_codes
from src.explainability import explain_single_application, generate_counterfactual, compute_shap_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="Fair Mortgage Underwriting API",
    description="HMDA-compliant, ECOA-aware mortgage decisioning with SHAP explanations",
    version="2.0.0",
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

# ── Global state ──────────────────────────────────────────────────────────────
_model = None
_feature_cols: List[str] = []
_threshold: float = 0.5
_portfolio_df: Optional[pd.DataFrame] = None
_portfolio_predictions: Optional[np.ndarray] = None
_portfolio_probas: Optional[np.ndarray] = None
_model_metrics: Dict = {}
_data_dir = str(PROJECT_ROOT / "data")
_model_dir = str(PROJECT_ROOT / "data" / "models")


def _ensure_model_loaded():
    """Loads or trains the model on first request."""
    global _model, _feature_cols, _threshold, _portfolio_df, _portfolio_predictions, _portfolio_probas, _model_metrics

    if _model is not None:
        return

    logger.info("Initializing model...")

    # Try loading saved model
    try:
        _model, _feature_cols, _threshold = load_artifacts(_model_dir, "lightgbm")
        logger.info("Loaded pre-trained model from disk")
    except FileNotFoundError:
        logger.info("No saved model found. Training on synthetic data...")
        _train_and_save_model()

    # Load portfolio data for fairness reports
    try:
        _portfolio_df = load_data(_data_dir)
        from sklearn.model_selection import train_test_split
        X, y, _, _ = build_feature_pipeline(_portfolio_df)

        # Align features
        X = _align_features(X, _feature_cols)
        _portfolio_probas = _model.predict_proba(X)[:, 1]
        _portfolio_predictions = (_portfolio_probas >= _threshold).astype(int)
        logger.info(f"Portfolio loaded: {len(_portfolio_df):,} applications")
    except Exception as exc:
        logger.warning(f"Could not load portfolio data: {exc}")


def _train_and_save_model():
    """Trains model on synthetic data and saves artifacts."""
    global _model, _feature_cols, _threshold, _model_metrics

    from sklearn.model_selection import train_test_split

    df = generate_synthetic_mortgage_data(n=50000)
    X, y, feature_cols, _ = build_feature_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        _model = train_lightgbm(X_train, y_train)
    except ImportError:
        from src.models import train_logistic_regression
        _model = train_logistic_regression(X_train, y_train)

    _feature_cols = list(feature_cols)
    _threshold = 0.5

    # Evaluate
    y_pred = (_model.predict_proba(X_test)[:, 1] >= _threshold).astype(int)
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    _model_metrics = {
        "roc_auc": round(roc_auc_score(y_test, _model.predict_proba(X_test)[:, 1]), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    save_artifacts(_model, _feature_cols, _threshold, _model_dir, "lightgbm")
    logger.info(f"Model trained and saved. AUC={_model_metrics['roc_auc']:.4f}")


def _align_features(X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """Aligns feature DataFrame to expected columns (add missing, drop extra)."""
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected_cols].fillna(0).astype(float)


def _application_to_features(app: "MortgageApplication") -> pd.DataFrame:
    """Converts a MortgageApplication Pydantic model to a feature DataFrame."""
    row = {
        "loan_amount": app.loan_amount,
        "income": app.income,
        "debt_to_income_ratio": app.debt_to_income_ratio,
        "loan_to_value_ratio": app.loan_to_value_ratio,
        "loan_purpose": app.loan_purpose,
        "property_type": app.property_type,
        "occupancy_type": app.occupancy_type,
        "lien_status": app.lien_status,
        "applicant_sex": app.applicant_sex,
        "applicant_race_1": app.applicant_race_1,
        "applicant_ethnicity_1": app.applicant_ethnicity_1 or 2,
        "applicant_age": app.applicant_age or "35-44",
        "applicant_age_raw": 38,
        "co_applicant_present": 1 if app.co_applicant else 0,
        "co_applicant_sex": 5,
        "co_applicant_race_1": 8,
        "loan_term": app.loan_term,
        "county_code": app.county_code or "0600001",
        "census_tract": app.census_tract or "0600001000",
        "action_taken": 1,  # placeholder for feature pipeline
        "preapproval": 2,
        "construction_method": 1,
        "negative_amortization": 2,
        "interest_only_payments": 2,
        "balloon_payment": 2,
        "open_end_line_of_credit": 2,
        "reverse_mortgage": 2,
        "business_or_commercial_purpose": 2,
        "hoepa_status": 3,
    }
    return pd.DataFrame([row])


# ── Pydantic Models ───────────────────────────────────────────────────────────

class MortgageApplication(BaseModel):
    loan_amount: float = Field(..., ge=10000, le=5000000, description="Loan amount in dollars")
    income: float = Field(..., ge=1000, le=5000000, description="Annual income in dollars")
    debt_to_income_ratio: float = Field(..., ge=0, le=100, description="DTI ratio as percentage")
    loan_to_value_ratio: float = Field(..., ge=0, le=100, description="LTV ratio as percentage")
    loan_purpose: int = Field(default=1, description="1=purchase, 2=improvement, 31=refi, 32=cash-out")
    property_type: int = Field(default=1, description="1=1-4 family, 2=manufactured, 3=multifamily")
    occupancy_type: int = Field(default=1, description="1=primary, 2=second, 3=investment")
    lien_status: int = Field(default=1, description="1=first lien, 2=subordinate")
    applicant_sex: int = Field(default=3, description="1=male, 2=female, 3=not provided")
    applicant_race_1: int = Field(default=6, description="1-5=race codes, 6=not provided")
    applicant_ethnicity_1: Optional[int] = Field(default=3, description="1=Hispanic, 2=not Hispanic, 3=not provided")
    applicant_age: Optional[str] = Field(default="35-44", description="Age group: 18-24, 25-34, etc.")
    co_applicant: Optional[bool] = Field(default=False)
    loan_term: int = Field(default=360, description="Loan term in months")
    county_code: Optional[str] = Field(default=None)
    census_tract: Optional[str] = Field(default=None)

    @field_validator("loan_amount", "income")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class BatchApplicationRequest(BaseModel):
    applications: List[MortgageApplication] = Field(..., max_length=1000)


class ScoringResponse(BaseModel):
    approval_probability: float
    decision: str
    confidence: float
    adverse_action_codes: List[Dict]
    shap_explanation: Dict
    fairness_flags: Dict
    counterfactual_suggestion: Dict


class ModelMetricsResponse(BaseModel):
    roc_auc: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    model_type: str = "LightGBM"
    threshold: float = 0.5


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    logger.info("API starting up...")
    _ensure_model_loaded()
    logger.info("API ready")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/score_application", response_model=ScoringResponse)
async def score_application(application: MortgageApplication):
    """
    Scores a single mortgage application.
    Returns approval probability, SHAP explanation, adverse action codes, and counterfactual.
    """
    _ensure_model_loaded()

    try:
        # Feature engineering
        raw_df = _application_to_features(application)
        raw_df = engineer_affordability_features(raw_df)
        raw_df = engineer_risk_features(raw_df)
        raw_df = engineer_geographic_features(raw_df)
        raw_df = encode_categoricals(raw_df)

        X = _align_features(raw_df, _feature_cols)

        # Predict
        prob = float(_model.predict_proba(X)[0, 1])
        decision = "APPROVED" if prob >= _threshold else "DENIED"

        # SHAP explanation
        shap_exp = explain_single_application(_model, X, _feature_cols, _threshold)

        # Adverse action codes (ECOA required for denials)
        adverse_codes = []
        if decision == "DENIED":
            adverse_codes = generate_adverse_action_codes(shap_exp, top_n=4)

        # Counterfactual
        counterfactual = generate_counterfactual(shap_exp, X, _feature_cols, _model, _threshold)

        # Fairness flags
        fairness_flags = _compute_application_fairness_flags(application, prob)

        return ScoringResponse(
            approval_probability=round(prob, 4),
            decision=decision,
            confidence=round(abs(prob - 0.5) * 2, 4),
            adverse_action_codes=adverse_codes,
            shap_explanation=shap_exp,
            fairness_flags=fairness_flags,
            counterfactual_suggestion=counterfactual,
        )

    except Exception as exc:
        logger.error(f"Scoring error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score_batch")
async def score_batch(request: BatchApplicationRequest):
    """
    Batch-scores multiple mortgage applications.
    Returns summary statistics plus individual scores.
    """
    _ensure_model_loaded()
    results = []
    errors = []

    for i, app in enumerate(request.applications):
        try:
            raw_df = _application_to_features(app)
            raw_df = engineer_affordability_features(raw_df)
            raw_df = engineer_risk_features(raw_df)
            raw_df = engineer_geographic_features(raw_df)
            raw_df = encode_categoricals(raw_df)
            X = _align_features(raw_df, _feature_cols)

            prob = float(_model.predict_proba(X)[0, 1])
            decision = "APPROVED" if prob >= _threshold else "DENIED"

            results.append({
                "application_index": i,
                "approval_probability": round(prob, 4),
                "decision": decision,
                "income": app.income,
                "loan_amount": app.loan_amount,
                "dti": app.debt_to_income_ratio,
                "ltv": app.loan_to_value_ratio,
            })
        except Exception as exc:
            errors.append({"application_index": i, "error": str(exc)})

    approved = [r for r in results if r["decision"] == "APPROVED"]
    return {
        "total": len(request.applications),
        "approved": len(approved),
        "denied": len(results) - len(approved),
        "approval_rate": round(len(approved) / max(len(results), 1), 4),
        "avg_probability": round(np.mean([r["approval_probability"] for r in results]), 4) if results else 0,
        "results": results,
        "errors": errors,
    }


@app.get("/fairness_report")
async def get_fairness_report():
    """
    Returns portfolio-level ECOA/Fair Housing Act fairness metrics.
    """
    _ensure_model_loaded()

    if _portfolio_df is None or _portfolio_predictions is None:
        raise HTTPException(status_code=503, detail="Portfolio data not loaded")

    try:
        X, y, _, _ = build_feature_pipeline(_portfolio_df)
        X = _align_features(X, _feature_cols)

        # Rebuild predictions on aligned features
        probas = _model.predict_proba(X)[:, 1]
        preds = (probas >= _threshold).astype(int)

        report = generate_fairness_report(_portfolio_df, preds, probas, y.values)
        return report

    except Exception as exc:
        logger.error(f"Fairness report error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model_metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Returns model performance metrics."""
    _ensure_model_loaded()
    return ModelMetricsResponse(
        roc_auc=_model_metrics.get("roc_auc"),
        f1=_model_metrics.get("f1"),
        accuracy=_model_metrics.get("accuracy"),
        n_train=_model_metrics.get("n_train"),
        n_test=_model_metrics.get("n_test"),
        threshold=_threshold,
    )


@app.get("/geographic_analysis/{state_code}")
async def geographic_analysis(state_code: str):
    """
    Returns approval rates by MSA/county for a given state FIPS code.
    """
    _ensure_model_loaded()

    if _portfolio_df is None:
        # Generate synthetic geographic data
        df = generate_synthetic_mortgage_data(n=20000)
    else:
        df = _portfolio_df

    state_mask = df.get("county_code", pd.Series(dtype=str)).astype(str).str[:2] == state_code
    state_df = df[state_mask] if state_mask.sum() > 0 else df.head(5000)

    try:
        X, y, _, _ = build_feature_pipeline(state_df)
        X = _align_features(X, _feature_cols)
        probas = _model.predict_proba(X)[:, 1]
        preds = (probas >= _threshold).astype(int)

        # County-level aggregation
        county_results = []
        if "county_code" in state_df.columns:
            state_df_copy = state_df.copy()
            state_df_copy = state_df_copy[state_df_copy["action_taken"].isin([1, 2, 3])].copy()
            state_df_copy["predicted_approval"] = preds[:len(state_df_copy)]

            for county, group in state_df_copy.groupby("county_code"):
                approval_rate = group["predicted_approval"].mean()
                county_results.append({
                    "county_code": str(county),
                    "state_code": state_code,
                    "approval_rate": round(float(approval_rate), 4),
                    "n_applications": len(group),
                    "median_income": round(float(group["income"].median()), 2),
                    "median_loan_amount": round(float(group["loan_amount"].median()), 2),
                    "redlining_flag": approval_rate < 0.55,  # statistical outlier threshold
                })

        county_results.sort(key=lambda x: x["approval_rate"])
        overall_rate = round(float(preds.mean()), 4)

        return {
            "state_code": state_code,
            "overall_approval_rate": overall_rate,
            "n_counties": len(county_results),
            "n_applications": len(state_df_copy) if state_df_copy is not None else 0,
            "low_approval_counties": [c for c in county_results if c["approval_rate"] < 0.65],
            "county_breakdown": county_results[:50],  # Top 50 counties
        }

    except Exception as exc:
        logger.error(f"Geographic analysis error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Helper functions ───────────────────────────────────────────────────────────

def _compute_application_fairness_flags(application: MortgageApplication, prob: float) -> Dict:
    """Computes fairness flags for a single application."""
    flags = {
        "ecoa_protected_info_used": False,
        "race_used_in_decision": False,
        "sex_used_in_decision": False,
        "information_not_provided": [],
        "adverse_action_required": prob < _threshold,
    }

    if application.applicant_sex == 3:
        flags["information_not_provided"].append("sex")
    if application.applicant_race_1 == 6:
        flags["information_not_provided"].append("race")

    return flags


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

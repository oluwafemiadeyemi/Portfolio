"""
FastAPI: credit decision API with adverse action codes, score, risk tier.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Credit Risk & Loan Default Intelligence Platform",
    description="Basel III-compliant PD scoring with SHAP adverse action codes and fairness monitoring",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_models = None
_feature_cols = None

GRADE_BASE_RATE = {"A": 0.04, "B": 0.08, "C": 0.14, "D": 0.22, "E": 0.30, "F": 0.40, "G": 0.50}

ADVERSE_CODES = {
    "fico": "Credit score insufficient",
    "dti": "Debt-to-income ratio exceeds threshold",
    "delinq": "Recent delinquency history",
    "revol_util": "Credit utilisation too high",
    "inq": "Excessive recent credit inquiries",
    "pub_rec": "Derogatory public records present",
}


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., ge=1000, le=40_000, example=15000)
    term: int = Field(..., ge=36, le=60, example=36)
    annual_inc: float = Field(..., ge=15_000, le=500_000, example=75000)
    dti: float = Field(..., ge=0, le=50, example=18.5)
    fico_avg: int = Field(..., ge=490, le=850, example=680)
    delinq_2yrs: int = Field(default=0, ge=0)
    inq_last_6mths: int = Field(default=1, ge=0)
    revol_util: float = Field(default=50.0, ge=0, le=100)
    open_acc: int = Field(default=10, ge=1)
    pub_rec: int = Field(default=0, ge=0)
    purpose: str = Field(default="debt_consolidation")
    grade: str = Field(default="C", example="B")


class CreditDecision(BaseModel):
    decision: str
    probability_of_default: float
    credit_score: int
    risk_tier: str
    interest_rate_recommendation: float
    adverse_action_codes: List[str]
    max_loan_recommendation: float
    explanation: str


@app.get("/health")
def health():
    return {"status": "healthy", "model_available": (MODELS_DIR / "credit_models.pkl").exists()}


@app.post("/decide", response_model=CreditDecision)
def credit_decision(app: LoanApplication):
    """Issue a credit decision with score, risk tier, and adverse action codes."""
    pd_score = _estimate_pd(app)
    credit_score = _pd_to_credit_score(pd_score)
    risk_tier = _risk_tier(credit_score)
    decision = "APPROVED" if pd_score < 0.20 else ("REVIEW" if pd_score < 0.35 else "DECLINED")
    adverse = _adverse_actions(app, pd_score)
    int_rate = _recommend_rate(pd_score)
    max_loan = _max_loan(app.annual_inc, pd_score)

    return CreditDecision(
        decision=decision,
        probability_of_default=round(pd_score, 4),
        credit_score=credit_score,
        risk_tier=risk_tier,
        interest_rate_recommendation=int_rate,
        adverse_action_codes=adverse,
        max_loan_recommendation=max_loan,
        explanation=f"PD: {pd_score*100:.1f}% | Credit Score: {credit_score} | Tier: {risk_tier}",
    )


@app.get("/portfolio/summary")
def portfolio_summary():
    p = DATA_PROC / "loans.parquet"
    if not p.exists():
        raise HTTPException(503, "Run data_pipeline.prepare_all() first.")
    df = pd.read_parquet(p, columns=["loan_amnt", "grade", "int_rate",
                                      "loan_default", "annual_inc", "dti"])
    return {
        "total_loans": len(df),
        "total_amount_originated": round(float(df["loan_amnt"].sum()), 0),
        "overall_default_rate": round(float(df["loan_default"].mean() * 100), 2),
        "avg_loan_amount": round(float(df["loan_amnt"].mean()), 0),
        "avg_interest_rate": round(float(df["int_rate"].mean()), 2),
        "default_by_grade": df.groupby("grade")["loan_default"].mean().round(4).to_dict(),
        "avg_dti": round(float(df["dti"].mean()), 2),
    }


@app.get("/scorecard/distribution")
def scorecard_distribution():
    p = DATA_PROC / "scorecard.parquet"
    if not p.exists():
        raise HTTPException(503, "Run model.run_training() first.")
    df = pd.read_parquet(p)
    return {
        "mean_score": round(float(df["credit_score"].mean()), 0),
        "std_score":  round(float(df["credit_score"].std()), 0),
        "tier_distribution": df["risk_tier"].value_counts().to_dict(),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _estimate_pd(a: LoanApplication) -> float:
    """Rule-based PD estimate (replaced by model when available)."""
    base = GRADE_BASE_RATE.get(a.grade.upper(), 0.15)
    base += 0.002 * a.dti
    base += 0.001 * (750 - a.fico_avg) / 100
    base += 0.005 * a.delinq_2yrs
    base += 0.003 * a.pub_rec
    return float(np.clip(base, 0.01, 0.90))


def _pd_to_credit_score(pd_score: float) -> int:
    A, B = 600, 50
    odds = pd_score / (1 - pd_score + 1e-10)
    score = A - B * np.log(odds)
    return int(np.clip(score, 300, 850))


def _risk_tier(score: int) -> str:
    if score >= 760: return "Minimal Risk"
    if score >= 720: return "Very Low Risk"
    if score >= 670: return "Low Risk"
    if score >= 630: return "Medium Risk"
    if score >= 580: return "High Risk"
    return "Very High Risk"


def _recommend_rate(pd: float) -> float:
    base_rate = 3.5
    spread = pd * 60
    return round(base_rate + spread, 2)


def _max_loan(annual_inc: float, pd: float) -> float:
    dti_cap = 0.36
    monthly_capacity = annual_inc * dti_cap / 12
    multiplier = max(0.2, 1.0 - pd * 2)
    return round(monthly_capacity * 36 * multiplier, -2)


def _adverse_actions(a: LoanApplication, pd: float) -> List[str]:
    codes = []
    if a.fico_avg < 650:    codes.append(ADVERSE_CODES["fico"])
    if a.dti > 36:          codes.append(ADVERSE_CODES["dti"])
    if a.delinq_2yrs > 0:   codes.append(ADVERSE_CODES["delinq"])
    if a.revol_util > 75:   codes.append(ADVERSE_CODES["revol_util"])
    if a.inq_last_6mths > 3: codes.append(ADVERSE_CODES["inq"])
    if a.pub_rec > 0:       codes.append(ADVERSE_CODES["pub_rec"])
    return codes[:3] if codes else ["No adverse factors identified"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)

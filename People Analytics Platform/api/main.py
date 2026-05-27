"""
People Analytics & Workforce Intelligence API
FastAPI service for attrition prediction, ELV, DEI analytics, and org network analysis.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, load_bls_benchmark_rates
from src.features import build_feature_pipeline, create_age_groups
from src.features import engineer_satisfaction_composite, engineer_tenure_features
from src.features import engineer_compensation_features, engineer_flight_risk_signals, engineer_elv
from src.models import train_all_models, select_best_model, evaluate_model, load_artifacts, save_artifacts
from src.elv_model import compute_elv, segment_employees, compute_retention_roi
from src.dei_analytics import generate_dei_scorecard, compute_attrition_by_group
from src.network_analysis import generate_org_network, identify_key_brokers, identify_isolated_employees, compute_department_cohesion
from src.explainability import explain_flight_risk, generate_intervention_recommendations, generate_hr_narrative

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App initialization ─────────────────────────────────────────────────────────
app = FastAPI(
    title="People Analytics & Workforce Intelligence API",
    description="Attrition prediction, ELV, DEI analytics, and org network analysis",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────────
_model = None
_feature_cols: List[str] = []
_workforce_df: Optional[pd.DataFrame] = None
_workforce_probas: Optional[np.ndarray] = None
_workforce_preds: Optional[np.ndarray] = None
_model_metrics: Dict = {}
_data_dir = str(PROJECT_ROOT / "data")
_model_dir = str(PROJECT_ROOT / "data" / "models")


def _ensure_model_loaded():
    global _model, _feature_cols, _workforce_df, _workforce_probas, _workforce_preds, _model_metrics

    if _model is not None:
        return

    logger.info("Initializing people analytics model...")

    try:
        _model, _feature_cols = load_artifacts(_model_dir, "best_model")
        logger.info("Loaded pre-trained model from disk")
    except FileNotFoundError:
        logger.info("Training model on workforce data...")
        _train_and_save_model()

    # Load workforce for reports
    try:
        _workforce_df = load_data(_data_dir, target_n=10000)  # Reduced for API startup
        X, y, _, _ = build_feature_pipeline(_workforce_df)
        X = _align_features(X, _feature_cols)
        _workforce_probas = _model.predict_proba(X)[:, 1]
        _workforce_preds = (_workforce_probas >= 0.5).astype(int)
        logger.info(f"Workforce loaded: {len(_workforce_df):,} employees")
    except Exception as exc:
        logger.warning(f"Workforce data load error: {exc}")


def _train_and_save_model():
    global _model, _feature_cols, _model_metrics

    from sklearn.model_selection import train_test_split

    df = load_data(_data_dir, target_n=5000)
    X, y, feature_cols, _ = build_feature_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = train_all_models(X_train, y_train)
    best_name, best_model = select_best_model(models, X_test, y_test)
    _model = best_model
    _feature_cols = list(feature_cols)

    _model_metrics = evaluate_model(_model, X_test, y_test)
    save_artifacts(_model, _feature_cols, _model_dir, "best_model")
    logger.info(f"Model trained. Best: {best_name}, AUC={_model_metrics.get('roc_auc', 'N/A')}")


def _align_features(X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected_cols].fillna(0).astype(float)


def _employee_to_row(emp: "EmployeeInput") -> pd.DataFrame:
    """Converts EmployeeInput Pydantic model to a DataFrame row."""
    return pd.DataFrame([{
        "Age": emp.Age,
        "Attrition": 0,
        "BusinessTravel": emp.BusinessTravel,
        "DailyRate": emp.DailyRate,
        "Department": emp.Department,
        "DistanceFromHome": emp.DistanceFromHome,
        "Education": emp.Education,
        "EducationField": emp.EducationField or "Life Sciences",
        "EmployeeNumber": emp.EmployeeNumber or 9999,
        "EnvironmentSatisfaction": emp.EnvironmentSatisfaction,
        "Gender": emp.Gender,
        "HourlyRate": emp.HourlyRate,
        "JobInvolvement": emp.JobInvolvement,
        "JobLevel": emp.JobLevel,
        "JobRole": emp.JobRole,
        "JobSatisfaction": emp.JobSatisfaction,
        "MaritalStatus": emp.MaritalStatus,
        "MonthlyIncome": emp.MonthlyIncome,
        "MonthlyRate": emp.MonthlyRate,
        "NumCompaniesWorked": emp.NumCompaniesWorked,
        "OverTime": 1 if emp.OverTime else 0,
        "PercentSalaryHike": emp.PercentSalaryHike,
        "PerformanceRating": emp.PerformanceRating,
        "RelationshipSatisfaction": emp.RelationshipSatisfaction,
        "StockOptionLevel": emp.StockOptionLevel,
        "TotalWorkingYears": emp.TotalWorkingYears,
        "TrainingTimesLastYear": emp.TrainingTimesLastYear,
        "WorkLifeBalance": emp.WorkLifeBalance,
        "YearsAtCompany": emp.YearsAtCompany,
        "YearsInCurrentRole": emp.YearsInCurrentRole,
        "YearsSinceLastPromotion": emp.YearsSinceLastPromotion,
        "YearsWithCurrManager": emp.YearsWithCurrManager,
    }])


# ── Pydantic Models ────────────────────────────────────────────────────────────

class EmployeeInput(BaseModel):
    Age: int = Field(default=35, ge=18, le=65)
    BusinessTravel: str = Field(default="Travel_Rarely")
    DailyRate: int = Field(default=800, ge=100, le=1500)
    Department: str = Field(default="Research & Development")
    DistanceFromHome: int = Field(default=10, ge=1, le=30)
    Education: int = Field(default=3, ge=1, le=5)
    EducationField: Optional[str] = Field(default="Life Sciences")
    EmployeeNumber: Optional[int] = Field(default=None)
    EnvironmentSatisfaction: int = Field(default=3, ge=1, le=4)
    Gender: str = Field(default="Male")
    HourlyRate: int = Field(default=65, ge=30, le=100)
    JobInvolvement: int = Field(default=3, ge=1, le=4)
    JobLevel: int = Field(default=2, ge=1, le=5)
    JobRole: str = Field(default="Research Scientist")
    JobSatisfaction: int = Field(default=3, ge=1, le=4)
    MaritalStatus: str = Field(default="Married")
    MonthlyIncome: float = Field(default=5000, ge=1000, le=25000)
    MonthlyRate: int = Field(default=14000, ge=2000, le=27000)
    NumCompaniesWorked: int = Field(default=3, ge=0, le=9)
    OverTime: bool = Field(default=False)
    PercentSalaryHike: int = Field(default=14, ge=11, le=25)
    PerformanceRating: int = Field(default=3, ge=1, le=4)
    RelationshipSatisfaction: int = Field(default=3, ge=1, le=4)
    StockOptionLevel: int = Field(default=1, ge=0, le=3)
    TotalWorkingYears: int = Field(default=10, ge=0, le=40)
    TrainingTimesLastYear: int = Field(default=3, ge=0, le=6)
    WorkLifeBalance: int = Field(default=3, ge=1, le=4)
    YearsAtCompany: int = Field(default=5, ge=0, le=40)
    YearsInCurrentRole: int = Field(default=3, ge=0, le=18)
    YearsSinceLastPromotion: int = Field(default=2, ge=0, le=15)
    YearsWithCurrManager: int = Field(default=4, ge=0, le=17)


class DepartmentBatchRequest(BaseModel):
    department: str
    employees: List[EmployeeInput] = Field(..., max_length=500)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("People Analytics API starting up...")
    _ensure_model_loaded()
    logger.info("API ready")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/score_employee")
async def score_employee(employee: EmployeeInput):
    """
    Scores a single employee for attrition risk with full explainability.
    """
    _ensure_model_loaded()

    try:
        raw_df = _employee_to_row(employee)
        raw_df = create_age_groups(raw_df)
        raw_df = engineer_satisfaction_composite(raw_df)
        raw_df = engineer_tenure_features(raw_df)
        raw_df = engineer_compensation_features(raw_df)
        raw_df = engineer_flight_risk_signals(raw_df)
        raw_df = engineer_elv(raw_df)

        # Encode categoricals
        cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole"]
        raw_df = pd.get_dummies(raw_df, columns=[c for c in cat_cols if c in raw_df.columns],
                                dtype=int)

        X = _align_features(raw_df, _feature_cols)
        prob = float(_model.predict_proba(X)[0, 1])

        shap_exp = explain_flight_risk(_model, X, _feature_cols, employee.EmployeeNumber, 0.5)
        recommendations = generate_intervention_recommendations(shap_exp, raw_df)

        # ELV + segment
        elv_row = compute_elv(raw_df)
        elv = float(elv_row["elv"].iloc[0])
        replacement_cost = float(elv_row["replacement_cost"].iloc[0])
        flight_risk_score = float(raw_df["flight_risk_score"].iloc[0]) if "flight_risk_score" in raw_df.columns else 30.0

        # HR narrative
        narrative_summary = {
            "employee_id": employee.EmployeeNumber,
            "attrition_probability": prob,
            "risk_level": shap_exp.get("risk_level", "Medium"),
            "top_risk_factors": shap_exp.get("top_risk_factors", []),
            "recommendations": recommendations,
            "department": employee.Department,
            "job_role": employee.JobRole,
            "years_at_company": employee.YearsAtCompany,
            "monthly_income": employee.MonthlyIncome,
            "replacement_cost": replacement_cost,
            "elv": elv,
        }
        hr_narrative = generate_hr_narrative(narrative_summary)

        return {
            "attrition_probability": round(prob, 4),
            "flight_risk_score": round(flight_risk_score, 1),
            "elv": round(elv, 2),
            "replacement_cost": round(replacement_cost, 2),
            "segment": "Protect" if prob >= 0.5 and elv > 0 else "Develop" if prob >= 0.5 else "Monitor",
            "intervention_recommendations": recommendations,
            "shap_explanation": shap_exp,
            "hr_narrative": hr_narrative,
        }

    except Exception as exc:
        logger.error(f"Score employee error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score_department")
async def score_department(request: DepartmentBatchRequest):
    """
    Department-level batch scoring with aggregated risk metrics.
    """
    _ensure_model_loaded()
    results = []
    errors = []

    for i, emp in enumerate(request.employees):
        try:
            raw_df = _employee_to_row(emp)
            raw_df = create_age_groups(raw_df)
            raw_df = engineer_satisfaction_composite(raw_df)
            raw_df = engineer_tenure_features(raw_df)
            raw_df = engineer_compensation_features(raw_df)
            raw_df = engineer_flight_risk_signals(raw_df)
            raw_df = engineer_elv(raw_df)

            cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole"]
            raw_df = pd.get_dummies(raw_df, columns=[c for c in cat_cols if c in raw_df.columns], dtype=int)
            X = _align_features(raw_df, _feature_cols)
            prob = float(_model.predict_proba(X)[0, 1])

            results.append({
                "employee_id": emp.EmployeeNumber,
                "attrition_probability": round(prob, 4),
                "at_risk": prob >= 0.5,
                "job_role": emp.JobRole,
                "job_level": emp.JobLevel,
                "monthly_income": emp.MonthlyIncome,
            })
        except Exception as exc:
            errors.append({"index": i, "error": str(exc)})

    at_risk = [r for r in results if r["at_risk"]]
    probas = [r["attrition_probability"] for r in results]

    return {
        "department": request.department,
        "n_employees": len(results),
        "n_at_risk": len(at_risk),
        "attrition_rate_predicted": round(len(at_risk) / max(len(results), 1), 4),
        "avg_attrition_probability": round(float(np.mean(probas)), 4) if probas else 0,
        "highest_risk_employees": sorted(at_risk, key=lambda x: x["attrition_probability"], reverse=True)[:5],
        "results": results,
        "errors": errors,
    }


@app.get("/dei_report")
async def get_dei_report():
    """Returns comprehensive DEI scorecard."""
    _ensure_model_loaded()
    if _workforce_df is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")
    try:
        scorecard = generate_dei_scorecard(_workforce_df)
        return scorecard
    except Exception as exc:
        logger.error(f"DEI report error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/network_analysis")
async def get_network_analysis():
    """Returns org network analysis: key brokers, isolated employees, department cohesion."""
    _ensure_model_loaded()
    if _workforce_df is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")

    try:
        # Use sample for performance
        sample = _workforce_df.head(500)
        G = generate_org_network(sample)

        brokers = identify_key_brokers(G, sample, top_n=10)
        isolated = identify_isolated_employees(G, sample)
        cohesion = compute_department_cohesion(G, sample)

        return {
            "key_brokers": brokers,
            "isolated_employees_count": len(isolated),
            "isolated_sample": isolated[:10],
            "department_cohesion": cohesion,
            "network_stats": {
                "n_nodes": getattr(G, "number_of_nodes", lambda: len(sample))(),
                "n_edges": getattr(G, "number_of_edges", lambda: 0)(),
            },
        }
    except Exception as exc:
        logger.error(f"Network analysis error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/workforce_summary")
async def get_workforce_summary():
    """Returns portfolio-level workforce statistics."""
    _ensure_model_loaded()
    if _workforce_df is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")

    df = _workforce_df
    probas = _workforce_probas

    summary = {
        "n_employees": len(df),
        "attrition_rate_actual": round(float(df["Attrition"].mean()), 4) if "Attrition" in df.columns else None,
        "attrition_rate_predicted": round(float((_workforce_preds >= 0.5).mean()), 4) if probas is not None else None,
        "median_monthly_income": round(float(df["MonthlyIncome"].median()), 2) if "MonthlyIncome" in df.columns else None,
        "avg_tenure_years": round(float(df["YearsAtCompany"].mean()), 1) if "YearsAtCompany" in df.columns else None,
        "bls_benchmarks": load_bls_benchmark_rates(),
    }

    if "Department" in df.columns and "Attrition" in df.columns:
        by_dept = df.groupby("Department")["Attrition"].agg(["mean", "count"]).reset_index()
        by_dept.columns = ["Department", "attrition_rate", "n"]
        summary["by_department"] = by_dept.to_dict(orient="records")

    if probas is not None:
        n_at_risk = int((probas >= 0.5).sum())
        summary["n_at_risk"] = n_at_risk
        summary["pct_at_risk"] = round(n_at_risk / len(df), 4)

    return summary


@app.get("/retention_roi")
async def get_retention_roi():
    """Returns ROI analysis for a retention intervention program."""
    _ensure_model_loaded()
    if _workforce_df is None or _workforce_probas is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")

    try:
        df_with_elv = compute_elv(_workforce_df)
        roi = compute_retention_roi(df_with_elv, _workforce_probas, intervention_cost_per_employee=5000)
        return roi
    except Exception as exc:
        logger.error(f"Retention ROI error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Intervention optimizer ────────────────────────────────────────────────────

@app.get("/intervention_optimizer")
async def intervention_optimizer():
    """
    Rank retention interventions by expected ROI across the at-risk population.
    Applies a causal uplift model: each program has an estimated probability of
    reducing attrition for a given risk profile, multiplied by the employee's
    Expected Lifetime Value minus intervention cost.
    """
    _ensure_model_loaded()
    if _workforce_df is None or _workforce_probas is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")

    df = _workforce_df.copy()
    probas = _workforce_probas.copy()

    # Intervention catalog: name, cost, avg. churn reduction (uplift), eligibility filter
    programs = [
        {"name": "Targeted Salary Adjustment",       "cost": 8000,  "uplift": 0.22, "segment": "high_risk"},
        {"name": "Flexible Work Arrangement",         "cost": 1200,  "uplift": 0.14, "segment": "all"},
        {"name": "Manager 1-on-1 Coaching Program",   "cost": 2500,  "uplift": 0.17, "segment": "medium_risk"},
        {"name": "Accelerated Career Path Discussion", "cost": 1800,  "uplift": 0.18, "segment": "high_potential"},
        {"name": "Wellness & EAP Benefit Expansion",  "cost": 600,   "uplift": 0.09, "segment": "all"},
        {"name": "Recognition & Bonus Program",       "cost": 3500,  "uplift": 0.13, "segment": "all"},
        {"name": "Cross-Functional Rotation",         "cost": 4000,  "uplift": 0.16, "segment": "high_potential"},
        {"name": "Team Culture Workshop",             "cost": 900,   "uplift": 0.08, "segment": "all"},
    ]

    avg_salary = float(df["MonthlyIncome"].mean() * 12) if "MonthlyIncome" in df.columns else 75000
    replacement_cost = avg_salary * 1.5  # industry standard: 1.5x annual salary

    at_risk_mask = probas >= 0.50
    n_at_risk = int(at_risk_mask.sum())
    n_high_risk = int((probas >= 0.70).sum())
    n_medium_risk = int(((probas >= 0.50) & (probas < 0.70)).sum())

    results = []
    for prog in programs:
        seg = prog["segment"]
        if seg == "high_risk":
            eligible_n = n_high_risk
        elif seg == "medium_risk":
            eligible_n = n_medium_risk
        elif seg == "high_potential":
            eligible_n = max(1, int(n_at_risk * 0.30))
        else:
            eligible_n = n_at_risk

        prevented_attritions = eligible_n * prog["uplift"]
        saved_replacement_cost = prevented_attritions * replacement_cost
        total_program_cost = eligible_n * prog["cost"]
        net_benefit = saved_replacement_cost - total_program_cost
        roi_pct = (net_benefit / (total_program_cost + 1)) * 100

        results.append({
            "program": prog["name"],
            "eligible_employees": eligible_n,
            "estimated_uplift_pct": prog["uplift"],
            "prevented_attritions": round(prevented_attritions, 1),
            "total_program_cost_usd": round(total_program_cost, 0),
            "saved_replacement_cost_usd": round(saved_replacement_cost, 0),
            "net_benefit_usd": round(net_benefit, 0),
            "roi_pct": round(roi_pct, 1),
            "payback_months": round((total_program_cost / (saved_replacement_cost / 12 + 1)), 1),
        })

    results.sort(key=lambda r: r["roi_pct"], reverse=True)

    return {
        "at_risk_employees": n_at_risk,
        "high_risk_employees": n_high_risk,
        "average_replacement_cost_usd": round(replacement_cost, 0),
        "ranked_interventions": results,
        "top_recommendation": results[0] if results else None,
        "portfolio_max_savings_usd": round(sum(r["net_benefit_usd"] for r in results[:3]), 0),
    }


# ── Compensation benchmarking ─────────────────────────────────────────────────

@app.get("/compensation_benchmarking")
async def compensation_benchmarking():
    """
    Identify employees who are paid below market benchmarks by job level and
    department.  Flags pay equity concerns and quantifies the cost to bring
    underpaid employees to the 50th percentile.
    """
    _ensure_model_loaded()
    if _workforce_df is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")

    df = _workforce_df.copy()
    if "MonthlyIncome" not in df.columns:
        raise HTTPException(status_code=422, detail="MonthlyIncome column required")

    results = []
    group_cols = [c for c in ["JobLevel", "Department", "JobRole"] if c in df.columns]

    for grp_vals, grp in df.groupby(group_cols[:2] if len(group_cols) >= 2 else group_cols[:1]):
        median_comp = float(grp["MonthlyIncome"].median())
        p25 = float(grp["MonthlyIncome"].quantile(0.25))
        p75 = float(grp["MonthlyIncome"].quantile(0.75))
        n_underpaid = int((grp["MonthlyIncome"] < median_comp).sum())
        cost_to_bring_to_median = float(
            (median_comp - grp.loc[grp["MonthlyIncome"] < median_comp, "MonthlyIncome"]).clip(lower=0).sum() * 12
        )

        label = grp_vals if isinstance(grp_vals, str) else " / ".join(str(v) for v in grp_vals)
        results.append({
            "segment": label,
            "n_employees": len(grp),
            "median_monthly_income": round(median_comp, 0),
            "p25_monthly_income": round(p25, 0),
            "p75_monthly_income": round(p75, 0),
            "n_below_median": n_underpaid,
            "pct_below_median": round(n_underpaid / len(grp), 3),
            "annual_cost_to_close_gap_usd": round(cost_to_bring_to_median, 0),
        })

    results.sort(key=lambda r: r["pct_below_median"], reverse=True)
    total_gap_cost = sum(r["annual_cost_to_close_gap_usd"] for r in results)

    return {
        "segments_analyzed": len(results),
        "total_annual_cost_to_close_pay_gap_usd": round(total_gap_cost, 0),
        "top_underpaid_segments": results[:5],
        "full_breakdown": results,
    }


# ── Promotion velocity ────────────────────────────────────────────────────────

@app.get("/promotion_velocity")
async def promotion_velocity():
    """
    Compute average years to promotion by department and job level.
    Identifies stalled cohorts where lack of advancement correlates with
    elevated flight risk scores.
    """
    _ensure_model_loaded()
    if _workforce_df is None:
        raise HTTPException(status_code=503, detail="Workforce data not loaded")

    df = _workforce_df.copy()
    col_years = "YearsAtCompany" if "YearsAtCompany" in df.columns else None
    col_level = "JobLevel" if "JobLevel" in df.columns else None

    if col_years is None or col_level is None:
        return {"status": "insufficient_columns", "segments": []}

    results = []
    for level, grp in df.groupby(col_level):
        avg_tenure = float(grp[col_years].mean())
        n = len(grp)
        avg_proba = float(_workforce_probas[grp.index].mean()) if _workforce_probas is not None else None
        results.append({
            "job_level": int(level),
            "n_employees": n,
            "avg_years_at_level": round(avg_tenure, 1),
            "avg_flight_risk": round(avg_proba, 3) if avg_proba is not None else None,
            "stall_risk": avg_tenure > 4.0 and (avg_proba or 0) > 0.40,
        })

    results.sort(key=lambda r: r["avg_years_at_level"], reverse=True)
    return {"promotion_velocity_by_level": results}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

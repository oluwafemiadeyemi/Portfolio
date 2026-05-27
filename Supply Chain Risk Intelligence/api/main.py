"""
api/main.py
===========
FastAPI REST API for the Enterprise Financial Health & Supply Chain Risk Platform.

Endpoints:
  POST /score_company           — financial ratios → distress probabilities (multi-horizon)
  POST /score_portfolio         — batch scoring + portfolio risk summary
  GET  /supply_chain_risk/{id}  — network contagion risk for a company
  GET  /sector_risk_summary     — average distress probability by sector
  GET  /systemic_risk_report    — top 20 systemic risk nodes
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import compute_altman_zscore, generate_synthetic_financial_data, load_data
from src.explainability import compute_counterfactual, explain_company_risk, generate_risk_narrative
from src.features import build_feature_pipeline, engineer_news_sentiment_features
from src.models import (
    evaluate_model,
    load_artifacts,
    save_artifacts,
    train_multi_horizon_models,
    train_xgboost_distress,
)
from src.network_analysis import (
    compute_contagion_risk,
    generate_supply_chain_network,
    identify_systemic_risk_nodes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Financial Health & Supply Chain Risk API",
    description=(
        "Financial distress prediction 3–18 months ahead using financial ratios, "
        "Altman Z-Score, and supply chain network analysis. "
        "Target: JPMorgan, Goldman Sachs, Deloitte, EY."
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
# Global state
# ---------------------------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "data" / "models"
_models: Dict[str, Any] = {}
_feature_cols: List[str] = []
_reference_df: Optional[pd.DataFrame] = None
_supply_chain_graph = None
_company_probas: Optional[pd.Series] = None


@app.on_event("startup")
async def startup_event() -> None:
    global _models, _feature_cols, _reference_df, _supply_chain_graph, _company_probas

    try:
        _models, _feature_cols = load_artifacts(MODEL_DIR)
        logger.info("Loaded artifacts: %s", list(_models.keys()))
    except Exception as exc:
        logger.warning("Could not load artifacts (%s) — training on synthetic data.", exc)
        _train_default_models()

    # Load reference data
    data_dir = PROJECT_ROOT / "data"
    _reference_df = load_data(str(data_dir))
    logger.info("Reference dataset loaded: %d rows.", len(_reference_df))

    # Pre-compute supply chain graph
    try:
        _supply_chain_graph = generate_supply_chain_network(
            _reference_df.groupby("company_id").last().reset_index(),
            n_suppliers_per_company=3,
        )
        logger.info("Supply chain graph built: %d nodes.", _supply_chain_graph.number_of_nodes())
    except Exception as exc:
        logger.warning("Could not build supply chain graph: %s", exc)

    # Pre-compute company probabilities
    try:
        X_ref, y_dict, _ = build_feature_pipeline(_reference_df)
        if "distress_12m" in _models:
            for col in _feature_cols:
                if col not in X_ref.columns:
                    X_ref[col] = 0.0
            X_ref = X_ref[_feature_cols].fillna(0)
            probas = _models["distress_12m"].predict_proba(X_ref)[:, 1]
            latest = _reference_df.groupby("company_id").last().reset_index()
            if len(probas) == len(_reference_df):
                temp_df = _reference_df.copy()
                temp_df["proba"] = probas
                _company_probas = temp_df.groupby("company_id")["proba"].last()
            logger.info("Company probabilities pre-computed.")
    except Exception as exc:
        logger.warning("Could not pre-compute company probabilities: %s", exc)


def _train_default_models() -> None:
    global _models, _feature_cols
    df = generate_synthetic_financial_data(n_companies=1000, years=5)
    from src.data_loader import compute_altman_zscore
    df = compute_altman_zscore(df)
    X, y_dict, feat_cols = build_feature_pipeline(df)
    _feature_cols = feat_cols
    _models = train_multi_horizon_models(X, y_dict)
    logger.info("Default multi-horizon models trained.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FinancialRatios(BaseModel):
    ROA: float = Field(..., ge=-0.50, le=0.50, description="Return on Assets")
    ROE: float = Field(..., ge=-1.0, le=1.0, description="Return on Equity")
    net_profit_margin: float = Field(..., ge=-0.50, le=0.60)
    EBITDA_margin: float = Field(..., ge=-0.50, le=0.60)
    debt_to_equity: float = Field(..., ge=0.0, le=50.0)
    interest_coverage: float = Field(5.0, ge=-10.0, le=100.0)
    debt_to_assets: float = Field(..., ge=0.0, le=1.5)
    current_ratio: float = Field(..., ge=0.0, le=20.0)
    quick_ratio: float = Field(..., ge=0.0, le=15.0)
    cash_ratio: float = Field(0.5, ge=0.0, le=10.0)
    asset_turnover: float = Field(1.0, ge=0.0, le=10.0)
    inventory_turnover: float = Field(6.0, ge=0.0, le=200.0)
    receivables_turnover: float = Field(8.0, ge=0.0, le=200.0)
    revenue_growth_yoy: float = Field(0.05, ge=-1.0, le=5.0)
    asset_growth_yoy: float = Field(0.03, ge=-1.0, le=3.0)
    news_sentiment_30d: float = Field(0.0, ge=-1.0, le=1.0)
    sector: str = Field("Technology", description="Company sector")
    altman_wc_ta: Optional[float] = Field(None, description="Working Capital / Total Assets")
    altman_re_ta: Optional[float] = Field(None, description="Retained Earnings / Total Assets")
    altman_ebit_ta: Optional[float] = Field(None, description="EBIT / Total Assets")
    altman_mve_tl: Optional[float] = Field(None, description="Market Value Equity / Total Liabilities")
    altman_sales_ta: Optional[float] = Field(None, description="Sales / Total Assets")


class CompanyScoreRequest(BaseModel):
    company_name: str
    company_id: Optional[str] = None
    financial_ratios: FinancialRatios
    exposure_usd_mm: float = Field(10.0, ge=0.0, description="Credit exposure in $MM")


class PortfolioScoreRequest(BaseModel):
    companies: List[CompanyScoreRequest]
    portfolio_name: str = "Portfolio"


# ---------------------------------------------------------------------------
# Helper: build feature row from ratios
# ---------------------------------------------------------------------------

def _ratios_to_features(ratios: FinancialRatios) -> pd.DataFrame:
    from src.features import (
        engineer_altman_components,
        engineer_news_sentiment_features,
        engineer_financial_ratios,
    )

    row = ratios.dict()
    df_row = pd.DataFrame([row])

    # Compute Altman Z-score
    if all(v is not None for v in [ratios.altman_wc_ta, ratios.altman_re_ta,
                                    ratios.altman_ebit_ta, ratios.altman_mve_tl,
                                    ratios.altman_sales_ta]):
        from src.data_loader import compute_altman_zscore_single
        z = compute_altman_zscore_single(
            ratios.altman_wc_ta, ratios.altman_re_ta,
            ratios.altman_ebit_ta, ratios.altman_mve_tl, ratios.altman_sales_ta,
        )
        df_row["altman_zscore"] = z
    else:
        # Estimate Z from available ratios
        df_row["altman_wc_ta"] = ratios.current_ratio * 0.15
        df_row["altman_re_ta"] = max(ratios.ROA * 2, 0)
        df_row["altman_ebit_ta"] = ratios.ROA
        df_row["altman_mve_tl"] = (1 - ratios.debt_to_assets) / max(ratios.debt_to_assets, 0.01)
        df_row["altman_sales_ta"] = ratios.asset_turnover
        from src.data_loader import compute_altman_zscore_single
        z = compute_altman_zscore_single(
            df_row["altman_wc_ta"].iloc[0],
            df_row["altman_re_ta"].iloc[0],
            df_row["altman_ebit_ta"].iloc[0],
            df_row["altman_mve_tl"].iloc[0],
            df_row["altman_sales_ta"].iloc[0],
        )
        df_row["altman_zscore"] = z

    df_row = engineer_altman_components(df_row)
    df_row = engineer_news_sentiment_features(df_row)
    df_row = engineer_financial_ratios(df_row)

    # Align to training feature columns
    for col in _feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0.0
    df_row = df_row[_feature_cols].fillna(0)
    return df_row


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/score_company")
async def score_company(req: CompanyScoreRequest) -> Dict[str, Any]:
    """Score a company's financial distress probability across multiple horizons."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    feat_row = _ratios_to_features(req.financial_ratios)
    altman_z = float(feat_row.get("altman_zscore", pd.Series([2.0])).iloc[0]) if "altman_zscore" in feat_row.columns else None

    # Multi-horizon predictions
    horizon_scores: Dict[str, float] = {}
    for horizon in ["distress_3m", "distress_6m", "distress_12m", "distress_18m"]:
        if horizon in _models:
            p = float(_models[horizon].predict_proba(feat_row)[:, 1][0])
            horizon_scores[horizon.replace("distress_", "")] = round(p, 4)

    # SHAP explanation (using 12m model as primary)
    primary_model = _models.get("distress_12m", next(iter(_models.values())))
    explanation = explain_company_risk(
        primary_model,
        feat_row,
        _feature_cols,
        company_name=req.company_name,
        altman_zscore=altman_z,
        sector=req.financial_ratios.sector,
    )

    # Altman zone
    if altman_z is not None:
        altman_zone = "Safe" if altman_z > 2.99 else "Grey" if altman_z >= 1.81 else "Distress"
    else:
        altman_zone = "Unknown"

    # Peer percentile (approximate from reference data)
    peer_percentile = None
    if _reference_df is not None and "sector" in _reference_df.columns:
        sector_df = _reference_df[_reference_df["sector"] == req.financial_ratios.sector]
        if len(sector_df) > 0 and "ROA" in sector_df.columns:
            peer_percentile = float(np.mean(sector_df["ROA"] <= req.financial_ratios.ROA) * 100)

    return {
        "company_name": req.company_name,
        "company_id": req.company_id,
        "distress_probability_3m": horizon_scores.get("3m"),
        "distress_probability_6m": horizon_scores.get("6m"),
        "distress_probability_12m": horizon_scores.get("12m"),
        "distress_probability_18m": horizon_scores.get("18m"),
        "altman_zscore": round(altman_z, 3) if altman_z is not None else None,
        "altman_zone": altman_zone,
        "risk_level": explanation["risk_level"],
        "shap_explanation": {
            "top_risk_factors": explanation["top_risk_factors"][:3],
            "top_protective_factors": explanation["top_protective_factors"][:3],
        },
        "risk_narrative": explanation["narrative"],
        "peer_percentile_roa": round(peer_percentile, 1) if peer_percentile is not None else None,
    }


@app.post("/score_portfolio")
async def score_portfolio(req: PortfolioScoreRequest) -> Dict[str, Any]:
    """Score multiple companies and compute portfolio-level risk summary."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    individual_scores = []
    for comp_req in req.companies:
        try:
            score = await score_company(comp_req)
            score["exposure_usd_mm"] = comp_req.exposure_usd_mm
            individual_scores.append(score)
        except Exception as exc:
            logger.warning("Failed to score %s: %s", comp_req.company_name, exc)

    # Portfolio summary
    probs_12m = [s["distress_probability_12m"] or 0 for s in individual_scores]
    exposures = [s["exposure_usd_mm"] for s in individual_scores]
    total_exposure = sum(exposures)
    weighted_risk = sum(p * e for p, e in zip(probs_12m, exposures)) / max(total_exposure, 1)
    expected_loss = weighted_risk * total_exposure * 0.60  # 60% LGD

    critical_companies = [
        s["company_name"] for s in individual_scores
        if s["risk_level"] in ("Critical", "High")
    ]

    # Count by zone
    zone_counts = {"Safe": 0, "Grey": 0, "Distress": 0, "Unknown": 0}
    for s in individual_scores:
        zone = s.get("altman_zone", "Unknown")
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    return {
        "portfolio_name": req.portfolio_name,
        "n_companies": len(individual_scores),
        "total_exposure_usd_mm": round(total_exposure, 2),
        "weighted_average_distress_prob_12m": round(weighted_risk, 4),
        "expected_portfolio_loss_usd_mm": round(expected_loss, 2),
        "critical_high_risk_companies": critical_companies,
        "altman_zone_distribution": zone_counts,
        "individual_scores": individual_scores,
    }


@app.get("/supply_chain_risk/{company_id}")
async def supply_chain_risk(company_id: str) -> Dict[str, Any]:
    """Network contagion risk for a specific company."""
    if _supply_chain_graph is None:
        raise HTTPException(status_code=503, detail="Supply chain graph not available.")
    if company_id not in _supply_chain_graph.nodes:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not in supply chain graph.")

    if _company_probas is None:
        raise HTTPException(status_code=503, detail="Company probabilities not computed.")

    contagion_df = compute_contagion_risk(_supply_chain_graph, _reference_df, _company_probas)
    comp_row = contagion_df[contagion_df["company_id"] == company_id]

    if len(comp_row) == 0:
        raise HTTPException(status_code=404, detail=f"No contagion data for '{company_id}'.")

    suppliers = list(_supply_chain_graph.predecessors(company_id))
    supplier_risks = [
        {"supplier_id": s, "distress_probability": round(float(_company_probas.get(s, 0)), 4)}
        for s in suppliers
    ]

    row = comp_row.iloc[0]
    return {
        "company_id": company_id,
        "direct_supplier_exposure": float(row["direct_supplier_exposure"]),
        "indirect_exposure": float(row["indirect_exposure"]),
        "total_contagion_risk": float(row["total_contagion_risk"]),
        "n_suppliers": int(row["n_suppliers"]),
        "n_distressed_suppliers": int(row["n_distressed_suppliers"]),
        "supplier_details": supplier_risks,
    }


@app.get("/sector_risk_summary")
async def sector_risk_summary() -> Dict[str, Any]:
    """Average distress probability by sector."""
    if _reference_df is None or _company_probas is None:
        raise HTTPException(status_code=503, detail="Reference data not loaded.")

    latest = _reference_df.groupby("company_id").last().reset_index()
    latest["distress_proba"] = latest["company_id"].map(_company_probas)

    if "sector" not in latest.columns:
        raise HTTPException(status_code=503, detail="Sector data not available.")

    summary = (
        latest.groupby("sector")["distress_proba"]
        .agg(["mean", "median", "std", "count"])
        .round(4)
        .reset_index()
        .rename(columns={"mean": "avg_distress_prob", "median": "median_distress_prob",
                          "std": "std_distress_prob", "count": "n_companies"})
        .sort_values("avg_distress_prob", ascending=False)
    )

    return {
        "sector_risk_summary": summary.to_dict(orient="records"),
        "highest_risk_sector": summary.iloc[0]["sector"] if len(summary) > 0 else None,
        "lowest_risk_sector": summary.iloc[-1]["sector"] if len(summary) > 0 else None,
    }


@app.get("/systemic_risk_report")
async def systemic_risk_report(top_n: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    """Top systemic risk nodes in the supply chain network."""
    if _supply_chain_graph is None or _company_probas is None:
        raise HTTPException(status_code=503, detail="Supply chain data not available.")

    systemic_df = identify_systemic_risk_nodes(
        _supply_chain_graph, _reference_df, _company_probas, top_n=top_n
    )

    return {
        "report_description": (
            f"Top {top_n} companies by systemic risk score "
            "(out-degree × distress probability)."
        ),
        "total_companies_in_network": _supply_chain_graph.number_of_nodes(),
        "total_supply_relationships": _supply_chain_graph.number_of_edges(),
        "systemic_risk_nodes": systemic_df.to_dict(orient="records"),
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "Financial Distress & Supply Chain Risk API"}

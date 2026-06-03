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


# ── Cascading failure simulator ───────────────────────────────────────────────

@app.get("/cascade_simulation/{company_id}")
async def cascade_simulation(
    company_id: str,
    failure_threshold: float = Query(0.60, ge=0.0, le=1.0, description="Distress probability to trigger cascade"),
    max_hops: int = Query(3, ge=1, le=5, description="Max propagation depth"),
) -> Dict[str, Any]:
    """
    Simulate cascading supply chain failure starting from a single company.
    Uses BFS propagation: a distressed company (prob > threshold) infects
    its downstream buyers with a fraction of its distress score.
    Returns the full contagion tree, impacted revenue estimate, and risk heatmap.
    """
    if _supply_chain_graph is None or _reference_df is None or _company_probas is None:
        raise HTTPException(status_code=503, detail="Supply chain data not loaded.")

    # Map company_id to graph node index
    id_col = "company_id" if "company_id" in _reference_df.columns else _reference_df.columns[0]
    mask = _reference_df[id_col].astype(str) == str(company_id)
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found.")

    origin_idx = int(_reference_df[mask].index[0])
    origin_prob = float(_company_probas[origin_idx]) if origin_idx < len(_company_probas) else 0.5

    if origin_prob < failure_threshold:
        return {
            "company_id": company_id,
            "origin_distress_probability": round(origin_prob, 4),
            "cascade_triggered": False,
            "message": f"Distress probability {origin_prob:.3f} below trigger threshold {failure_threshold}.",
        }

    # BFS cascade
    visited = {origin_idx: origin_prob}
    frontier = [origin_idx]
    cascade_tree = []
    hop = 0

    contagion_decay = 0.55  # each hop transmits 55% of distress signal

    while frontier and hop < max_hops:
        next_frontier = []
        for node in frontier:
            node_prob = visited[node]
            transmitted_prob = node_prob * contagion_decay
            if transmitted_prob < 0.15:
                continue
            neighbors = list(_supply_chain_graph.successors(node)) if _supply_chain_graph.is_directed() else list(_supply_chain_graph.neighbors(node))
            for nbr in neighbors[:10]:  # cap fan-out
                if nbr not in visited:
                    combined = min(1.0, (_company_probas[nbr] if nbr < len(_company_probas) else 0.3) + transmitted_prob * 0.4)
                    visited[nbr] = combined
                    next_frontier.append(nbr)
                    cascade_tree.append({
                        "from_node": int(node),
                        "to_node": int(nbr),
                        "transmitted_distress": round(transmitted_prob, 4),
                        "combined_distress": round(combined, 4),
                        "hop": hop + 1,
                    })
        frontier = next_frontier
        hop += 1

    n_impacted = len(visited) - 1
    high_risk_impacted = sum(1 for v in visited.values() if v > 0.60)

    # Revenue at risk (approximate from company_revenue if present)
    revenue_at_risk = 0.0
    if "revenue" in _reference_df.columns:
        impacted_revs = _reference_df.iloc[list(visited.keys())]["revenue"].sum()
        revenue_at_risk = float(impacted_revs * 0.35)  # 35% revenue disruption estimate

    return {
        "company_id": company_id,
        "origin_distress_probability": round(origin_prob, 4),
        "cascade_triggered": True,
        "total_companies_impacted": n_impacted,
        "high_risk_after_cascade": high_risk_impacted,
        "max_propagation_depth": max_hops,
        "estimated_revenue_at_risk_usd": round(revenue_at_risk, 0),
        "contagion_tree": cascade_tree[:100],
        "impacted_node_distress": {str(k): round(v, 4) for k, v in sorted(visited.items(), key=lambda x: x[1], reverse=True)[:20]},
    }


# ── ESG risk overlay ──────────────────────────────────────────────────────────

@app.get("/esg_risk_overlay")
async def esg_risk_overlay() -> Dict[str, Any]:
    """
    Overlay ESG (Environmental, Social, Governance) risk scores onto the
    financial distress model to produce a composite sustainability-adjusted
    risk score.  Companies with high financial + high ESG risk are flagged
    as double-exposure candidates.
    """
    if _reference_df is None or _company_probas is None:
        raise HTTPException(status_code=503, detail="Portfolio data not loaded.")

    df = _reference_df.copy()
    n = len(df)
    np.random.seed(42)

    # Simulate ESG scores (0=worst, 100=best) correlated with company size/sector
    # In production these would come from MSCI, Sustainalytics, or Bloomberg ESG feeds
    esg_e = np.random.beta(2, 3, n) * 100   # Environmental
    esg_s = np.random.beta(2.5, 2.5, n) * 100  # Social
    esg_g = np.random.beta(3, 2, n) * 100   # Governance
    composite_esg = (esg_e * 0.35 + esg_s * 0.30 + esg_g * 0.35)
    esg_risk = 100 - composite_esg  # invert: higher = more ESG risk

    financial_risk = _company_probas[:n] * 100 if len(_company_probas) >= n else np.random.beta(2, 5, n) * 100

    # Quadrant classification
    fin_median = np.median(financial_risk)
    esg_median = np.median(esg_risk)

    quadrants = {
        "double_exposure": int(((financial_risk > fin_median) & (esg_risk > esg_median)).sum()),
        "financial_only": int(((financial_risk > fin_median) & (esg_risk <= esg_median)).sum()),
        "esg_only": int(((financial_risk <= fin_median) & (esg_risk > esg_median)).sum()),
        "low_risk": int(((financial_risk <= fin_median) & (esg_risk <= esg_median)).sum()),
    }

    top_double = np.argsort(financial_risk + esg_risk)[::-1][:10]
    id_col = "company_id" if "company_id" in df.columns else df.columns[0]
    name_col = "company_name" if "company_name" in df.columns else id_col

    worst_companies = []
    for idx in top_double:
        if idx < len(df):
            worst_companies.append({
                "company": str(df.iloc[idx][name_col]),
                "financial_risk_score": round(float(financial_risk[idx]), 1),
                "esg_risk_score": round(float(esg_risk[idx]), 1),
                "composite_risk": round(float((financial_risk[idx] + esg_risk[idx]) / 2), 1),
                "environmental_score": round(float(esg_e[idx]), 1),
                "social_score": round(float(esg_s[idx]), 1),
                "governance_score": round(float(esg_g[idx]), 1),
            })

    return {
        "portfolio_size": n,
        "quadrant_breakdown": quadrants,
        "portfolio_avg_esg_risk": round(float(esg_risk.mean()), 1),
        "portfolio_avg_financial_risk": round(float(financial_risk.mean()), 1),
        "top_double_exposure_companies": worst_companies,
        "esg_data_source": "Simulated (MSCI/Sustainalytics integration point)",
        "note": "Double-exposure companies require enhanced due diligence and diversification review.",
    }


# ── Scenario stress test ──────────────────────────────────────────────────────

@app.get("/supply_shock_scenarios")
async def supply_shock_scenarios() -> Dict[str, Any]:
    """
    Model portfolio distress rate shifts under macro supply shock scenarios:
    - Geopolitical disruption (semiconductor shortage)
    - Energy price spike (+50%)
    - Logistics collapse (port blockage)
    - Credit tightening (rates +200bps)
    """
    if _company_probas is None:
        raise HTTPException(status_code=503, detail="Portfolio not loaded.")

    base_rate = float(_company_probas.mean())
    n = len(_company_probas)

    scenarios = {
        "baseline": {"shock": 0.00, "description": "Current conditions"},
        "geopolitical_disruption": {"shock": 0.08, "description": "Semiconductor/rare-earth export restrictions"},
        "energy_price_spike_50pct": {"shock": 0.06, "description": "Energy costs +50% — manufacturing sectors most exposed"},
        "logistics_collapse": {"shock": 0.11, "description": "Major port blockage — 3–6 week lead time spike"},
        "credit_tightening_200bps": {"shock": 0.09, "description": "Interest rate +200bps — highly leveraged suppliers most at risk"},
        "combined_severe": {"shock": 0.22, "description": "All shocks simultaneously (tail risk scenario)"},
    }

    results = {}
    for name, cfg in scenarios.items():
        shocked_probas = np.clip(_company_probas + cfg["shock"], 0.0, 1.0)
        distress_rate = float((shocked_probas > 0.50).mean())
        high_risk_n = int((shocked_probas > 0.70).sum())
        results[name] = {
            "description": cfg["description"],
            "distress_rate": round(distress_rate, 4),
            "distress_rate_delta": round(distress_rate - base_rate, 4),
            "n_high_risk_companies": high_risk_n,
            "avg_portfolio_distress_prob": round(float(shocked_probas.mean()), 4),
        }

    return {"baseline_distress_rate": round(base_rate, 4), "scenarios": results, "n_companies": n}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "Financial Distress & Supply Chain Risk API"}


# ---------------------------------------------------------------------------
# AI-Powered Endpoints (Llama via Ollama)
# ---------------------------------------------------------------------------

from src.llm_insights import generate_risk_narrative as _llm_narrative, analyze_filing_text as _llm_filing
from pydantic import BaseModel as _BM


class RiskNarrativeRequest(_BM):
    company_name: Optional[str] = "Unknown Company"
    distress_prob_3m: float = 0.0
    distress_prob_12m: float = 0.0
    altman_zscore: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    sector: Optional[str] = None


class FilingAnalysisRequest(_BM):
    company_name: Optional[str] = "Unknown Company"
    filing_text: str


@app.post(
    "/ai/risk_narrative",
    tags=["AI Insights"],
    summary="Plain-English risk assessment via Llama (local, free)",
)
def ai_risk_narrative(req: RiskNarrativeRequest):
    """
    Converts quantitative financial distress scores into a plain-English
    credit committee briefing using llama3.2:4k running locally via Ollama.
    """
    data = {
        "Company":                    req.company_name,
        "Sector":                     req.sector or "n/a",
        "3-month distress probability": f"{req.distress_prob_3m:.1%}",
        "12-month distress probability": f"{req.distress_prob_12m:.1%}",
        "Altman Z-Score":             req.altman_zscore or "n/a",
        "Debt-to-Equity":             req.debt_to_equity or "n/a",
        "Current Ratio":              req.current_ratio or "n/a",
        "Revenue Growth":             f"{req.revenue_growth_pct:.1%}" if req.revenue_growth_pct else "n/a",
    }
    narrative = _llm_narrative(data)
    return {
        "company":      req.company_name,
        "narrative":    narrative,
        "model_used":   "llama3.2:4k (Ollama local)",
    }


@app.post(
    "/ai/filing_analysis",
    tags=["AI Insights"],
    summary="Extract structured risk factors from SEC filing text via Llama",
)
def ai_filing_analysis(req: FilingAnalysisRequest):
    """
    Extracts supply chain dependencies, risk factors, and concentration risks
    from raw SEC 10-K/10-Q filing text using llama3.2:4k.
    """
    result = _llm_filing(req.filing_text)
    result["company"]    = req.company_name
    result["model_used"] = "llama3.2:4k (Ollama local)"
    return result

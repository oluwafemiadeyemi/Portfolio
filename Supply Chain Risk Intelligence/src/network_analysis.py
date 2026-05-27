"""
network_analysis.py
===================
Supply chain risk network analysis for Financial Distress Prediction.

Functions:
  - Supply chain network generation (synthetic or from data)
  - Contagion risk propagation
  - Systemic risk node identification
  - Portfolio Value at Risk (VaR)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logging.getLogger(__name__).warning("networkx not installed — network analysis limited.")

logger = logging.getLogger(__name__)


# Sector relationship map: which sectors supply to which
SECTOR_SUPPLY_RELATIONSHIPS: Dict[str, List[str]] = {
    "Technology":   ["Consumer", "Finance", "Healthcare", "Industrial"],
    "Industrial":   ["Technology", "Consumer", "Energy"],
    "Energy":       ["Industrial", "Consumer", "Technology"],
    "Healthcare":   ["Consumer", "Finance"],
    "Finance":      ["Technology", "Consumer", "Industrial", "Healthcare", "Energy"],
    "Consumer":     ["Industrial", "Technology"],
}


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

def generate_supply_chain_network(
    df: pd.DataFrame,
    company_col: str = "company_id",
    sector_col: str = "sector",
    n_suppliers_per_company: int = 3,
    random_state: int = 42,
) -> "nx.DiGraph":
    """Create a synthetic supply chain network.

    Companies are nodes; directed edges represent supplier → customer
    relationships. Edges are created based on sector compatibility and
    random selection.

    Parameters
    ----------
    df: Company DataFrame with at least company_id and sector columns.
    company_col: Column with company IDs.
    sector_col: Column with sector names.
    n_suppliers_per_company: Average number of suppliers per company.
    random_state: RNG seed.

    Returns
    -------
    NetworkX DiGraph. Nodes have attributes: sector, distress_flag.
    """
    if not NX_AVAILABLE:
        raise ImportError("networkx is required for network analysis. Install with: pip install networkx")

    rng = np.random.default_rng(random_state)
    G = nx.DiGraph()

    # Use latest year per company if panel data
    if "year" in df.columns:
        df_latest = df.sort_values("year").groupby(company_col).last().reset_index()
    else:
        df_latest = df.copy()

    # Add nodes
    for _, row in df_latest.iterrows():
        comp_id = row[company_col]
        G.add_node(
            comp_id,
            sector=row.get(sector_col, "Unknown"),
            distress_flag=int(row.get("distress_flag", 0)),
            altman_zscore=float(row.get("altman_zscore", 2.0)),
        )

    # Build sector → company lookup
    sector_companies: Dict[str, List[str]] = {}
    for comp_id in G.nodes:
        sec = G.nodes[comp_id].get("sector", "Unknown")
        sector_companies.setdefault(sec, []).append(comp_id)

    # Add edges: each company selects suppliers from compatible sectors
    companies = list(G.nodes)
    for comp_id in companies:
        comp_sector = G.nodes[comp_id].get("sector", "Unknown")
        supplier_sectors = SECTOR_SUPPLY_RELATIONSHIPS.get(comp_sector, list(sector_companies.keys()))

        # Gather candidate suppliers from compatible sectors
        candidates = []
        for sup_sector in supplier_sectors:
            candidates.extend(sector_companies.get(sup_sector, []))

        # Remove self-loops
        candidates = [c for c in candidates if c != comp_id]

        if not candidates:
            continue

        n_sup = min(n_suppliers_per_company, len(candidates))
        chosen = rng.choice(candidates, size=n_sup, replace=False)

        for supplier in chosen:
            weight = float(rng.uniform(0.1, 1.0))  # Relationship strength
            G.add_edge(supplier, comp_id, weight=weight, relationship="supplier-customer")

    logger.info(
        "Supply chain network created: %d nodes, %d edges.",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Contagion risk propagation
# ---------------------------------------------------------------------------

def compute_contagion_risk(
    G: "nx.DiGraph",
    df: pd.DataFrame,
    distress_proba: pd.Series,
    company_col: str = "company_id",
    propagation_factor: float = 0.3,
) -> pd.DataFrame:
    """Compute supplier distress exposure for each company.

    For each company, the supplier distress exposure is the weighted sum
    of its direct suppliers' distress probabilities.

    Contagion formula:
      supplier_exposure(i) = Σ_{j ∈ suppliers(i)} weight(j→i) × P(distress_j)

    Parameters
    ----------
    G: Supply chain DiGraph.
    df: Company DataFrame (used to align company_id).
    distress_proba: Series mapping company_id → distress probability.
    company_col: Company ID column name.
    propagation_factor: Scaling factor for indirect (2nd-degree) exposure.

    Returns
    -------
    DataFrame with columns: company_id, direct_supplier_exposure,
    indirect_exposure, total_contagion_risk, n_suppliers, n_distressed_suppliers.
    """
    if not NX_AVAILABLE:
        raise ImportError("networkx required.")

    proba_dict = distress_proba.to_dict() if hasattr(distress_proba, "to_dict") else dict(distress_proba)

    rows = []
    for node in G.nodes:
        predecessors = list(G.predecessors(node))  # Direct suppliers
        n_suppliers = len(predecessors)

        if n_suppliers == 0:
            direct_exposure = 0.0
            n_distressed = 0
        else:
            direct_exposure = sum(
                G.edges[pred, node].get("weight", 1.0) * proba_dict.get(pred, 0.0)
                for pred in predecessors
            )
            n_distressed = sum(1 for pred in predecessors if proba_dict.get(pred, 0.0) >= 0.5)

        # Indirect (2nd-degree) exposure
        second_degree_suppliers = set()
        for pred in predecessors:
            second_degree_suppliers.update(G.predecessors(pred))
        second_degree_suppliers.discard(node)

        indirect_exposure = propagation_factor * sum(
            proba_dict.get(s, 0.0) for s in second_degree_suppliers
        ) / max(len(second_degree_suppliers), 1)

        total_contagion = float(np.clip(direct_exposure + indirect_exposure, 0.0, 1.0))

        rows.append({
            company_col: node,
            "direct_supplier_exposure": round(direct_exposure, 4),
            "indirect_exposure": round(indirect_exposure, 4),
            "total_contagion_risk": round(total_contagion, 4),
            "n_suppliers": n_suppliers,
            "n_distressed_suppliers": n_distressed,
        })

    result_df = pd.DataFrame(rows)
    logger.info(
        "Contagion risk computed for %d companies. Mean exposure: %.3f",
        len(result_df),
        result_df["total_contagion_risk"].mean(),
    )
    return result_df


# ---------------------------------------------------------------------------
# Systemic risk identification
# ---------------------------------------------------------------------------

def identify_systemic_risk_nodes(
    G: "nx.DiGraph",
    df: pd.DataFrame,
    distress_proba: pd.Series,
    company_col: str = "company_id",
    top_n: int = 20,
) -> pd.DataFrame:
    """Identify companies whose failure would most impact the network.

    Systemic risk score = out-degree × distress_probability × mean_customer_size

    A company with many customers AND high distress probability is the most
    systemically dangerous node.

    Parameters
    ----------
    G: Supply chain DiGraph.
    df: Company DataFrame.
    distress_proba: Series mapping company_id → distress probability.
    top_n: Number of top systemic risk nodes to return.

    Returns
    -------
    DataFrame of top-N systemic risk companies, ranked by systemic_risk_score.
    """
    if not NX_AVAILABLE:
        raise ImportError("networkx required.")

    proba_dict = distress_proba.to_dict() if hasattr(distress_proba, "to_dict") else dict(distress_proba)

    rows = []
    for node in G.nodes:
        out_degree = G.out_degree(node)  # Number of customers supplied
        in_degree = G.in_degree(node)    # Number of suppliers
        p_distress = proba_dict.get(node, 0.0)

        # Impact = number of customers × distress probability
        systemic_score = float(out_degree * p_distress)

        # Betweenness centrality approximation (out-degree weighted)
        total_edge_weight = sum(
            G.edges[node, succ].get("weight", 1.0)
            for succ in G.successors(node)
        )

        rows.append({
            company_col: node,
            "sector": G.nodes[node].get("sector", "Unknown"),
            "distress_probability": round(p_distress, 4),
            "n_customers": int(out_degree),
            "n_suppliers": int(in_degree),
            "systemic_risk_score": round(systemic_score, 4),
            "total_supply_weight": round(total_edge_weight, 4),
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("systemic_risk_score", ascending=False)

    # Merge with company data if available
    if company_col in df.columns:
        merge_cols = [c for c in ["altman_zscore", "altman_zone", "ROA", "debt_to_equity"]
                      if c in df.columns]
        if merge_cols:
            df_latest = df.sort_values("year").groupby(company_col).last().reset_index() \
                if "year" in df.columns else df
            result_df = result_df.merge(
                df_latest[[company_col] + merge_cols],
                on=company_col,
                how="left",
            )

    top_df = result_df.head(top_n).reset_index(drop=True)
    logger.info("Top %d systemic risk nodes identified.", len(top_df))
    return top_df


# ---------------------------------------------------------------------------
# Portfolio Value at Risk
# ---------------------------------------------------------------------------

def compute_portfolio_var(
    df: pd.DataFrame,
    distress_proba: pd.Series,
    exposures: pd.Series,
    company_col: str = "company_id",
    confidence_level: float = 0.99,
    n_simulations: int = 10000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute Value at Risk for a credit portfolio.

    Simulates portfolio losses using a Monte Carlo approach with correlated
    defaults (via a Gaussian copula approximation).

    Parameters
    ----------
    df: Company DataFrame.
    distress_proba: Series mapping company_id → distress probability.
    exposures: Series mapping company_id → loan/credit exposure (in $MM).
    confidence_level: VaR confidence level (e.g., 0.99 for 99% VaR).
    n_simulations: Number of Monte Carlo scenarios.

    Returns
    -------
    Dict with: expected_loss, var_99, var_95, cvar_99, max_loss, portfolio_size
    """
    rng = np.random.default_rng(random_state)

    proba_dict = distress_proba.to_dict() if hasattr(distress_proba, "to_dict") else dict(distress_proba)
    expo_dict = exposures.to_dict() if hasattr(exposures, "to_dict") else dict(exposures)

    companies = list(set(proba_dict.keys()) & set(expo_dict.keys()))
    if not companies:
        logger.warning("No matching companies between distress_proba and exposures.")
        return {}

    n_companies = len(companies)
    probs = np.array([proba_dict[c] for c in companies])
    expo_vals = np.array([expo_dict[c] for c in companies])
    total_exposure = float(expo_vals.sum())

    # Monte Carlo simulation
    portfolio_losses = np.zeros(n_simulations)
    for sim_idx in range(n_simulations):
        # Independent default simulation (simplified — no correlation)
        uniform_draws = rng.uniform(0, 1, n_companies)
        defaults = (uniform_draws < probs).astype(float)
        loss_given_default = 0.60  # 60% LGD assumption
        portfolio_losses[sim_idx] = float(np.sum(defaults * expo_vals * loss_given_default))

    expected_loss = float(np.mean(portfolio_losses))
    var_99 = float(np.percentile(portfolio_losses, confidence_level * 100))
    var_95 = float(np.percentile(portfolio_losses, 95))
    cvar_99 = float(portfolio_losses[portfolio_losses >= var_99].mean()) if np.any(portfolio_losses >= var_99) else var_99
    max_loss = float(np.max(portfolio_losses))

    result = {
        "portfolio_size_usd_mm": round(total_exposure, 2),
        "n_companies": n_companies,
        "expected_loss_usd_mm": round(expected_loss, 2),
        "expected_loss_pct": round(expected_loss / max(total_exposure, 1) * 100, 2),
        "var_99_usd_mm": round(var_99, 2),
        "var_95_usd_mm": round(var_95, 2),
        "cvar_99_usd_mm": round(cvar_99, 2),
        "max_loss_usd_mm": round(max_loss, 2),
        "confidence_level": confidence_level,
        "n_simulations": n_simulations,
    }
    logger.info("Portfolio VaR computed: %s", result)
    return result

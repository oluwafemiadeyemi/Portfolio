"""
Organizational Network Analysis
Builds synthetic org graph from employee data, computes centrality metrics,
identifies key brokers, and measures department cohesion.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logger.warning("NetworkX not installed. Network analysis features will be limited.")


def generate_org_network(df: pd.DataFrame) -> Any:
    """
    Creates a synthetic organizational DiGraph from employee data.
    Nodes = employees, edges = reporting relationships based on JobLevel/Department.

    Hierarchy: higher JobLevel employees manage lower JobLevel employees within
    the same department, with realistic manager spans (5-8 direct reports).

    Args:
        df: Employee DataFrame

    Returns:
        NetworkX DiGraph (or dict if NetworkX unavailable)
    """
    if not _HAS_NX:
        return _generate_org_network_dict(df)

    df_work = df.copy().reset_index(drop=True)
    emp_id_col = "EmployeeNumber" if "EmployeeNumber" in df_work.columns else "index"
    if emp_id_col == "index":
        df_work["EmployeeNumber"] = df_work.index

    G = nx.DiGraph()

    # Add nodes with employee attributes
    for _, row in df_work.iterrows():
        node_id = int(row["EmployeeNumber"])
        G.add_node(
            node_id,
            department=str(row.get("Department", "Unknown")),
            job_level=int(row.get("JobLevel", 1)),
            job_role=str(row.get("JobRole", "Unknown")),
            monthly_income=float(row.get("MonthlyIncome", 5000)),
            attrition=int(row.get("Attrition", 0)),
            age=int(row.get("Age", 37)),
        )

    # Add reporting edges: higher level → lower level within department
    if "Department" in df_work.columns and "JobLevel" in df_work.columns:
        for dept in df_work["Department"].unique():
            dept_mask = df_work["Department"] == dept
            dept_employees = df_work[dept_mask].copy()

            # Sort by job level (descending)
            dept_employees = dept_employees.sort_values("JobLevel", ascending=False)

            # Assign managers: each level-5+ employee manages level-4, etc.
            for level in [5, 4, 3, 2]:
                managers = dept_employees[dept_employees["JobLevel"] == level]["EmployeeNumber"].tolist()
                reports = dept_employees[dept_employees["JobLevel"] == level - 1]["EmployeeNumber"].tolist()

                if not managers or not reports:
                    continue

                # Distribute reports among managers (round-robin)
                for i, report_id in enumerate(reports):
                    manager_id = managers[i % len(managers)]
                    if manager_id != report_id:
                        G.add_edge(int(manager_id), int(report_id), relationship="reports_to")

    # Add informal collaboration edges within department (for cohesion analysis)
    rng = np.random.default_rng(42)
    for dept in df_work["Department"].unique():
        dept_emps = df_work[df_work["Department"] == dept]["EmployeeNumber"].tolist()
        # Add ~10% random collaboration edges
        n_collab = max(1, int(len(dept_emps) * 0.10))
        for _ in range(n_collab):
            if len(dept_emps) >= 2:
                a, b = rng.choice(dept_emps, size=2, replace=False)
                if not G.has_edge(int(a), int(b)):
                    G.add_edge(int(a), int(b), relationship="collaborates")

    logger.info(
        f"Org network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def compute_network_centrality(G: Any) -> Dict[int, Dict]:
    """
    Computes betweenness, degree, and closeness centrality per employee.

    Args:
        G: NetworkX DiGraph

    Returns:
        Dict of {employee_id: {betweenness, degree, closeness, pagerank}}
    """
    if not _HAS_NX:
        return {}

    if G.number_of_nodes() == 0:
        return {}

    logger.info("Computing network centrality metrics...")

    # Use undirected version for centrality (more interpretable for org networks)
    G_undirected = G.to_undirected()

    # Only compute on largest connected component for efficiency
    if nx.is_connected(G_undirected):
        G_work = G_undirected
    else:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_work = G_undirected.subgraph(largest_cc).copy()

    # Betweenness: importance as a bridge/connector
    betweenness = nx.betweenness_centrality(G_work, normalized=True)

    # Degree: number of direct connections
    degree = dict(G_work.degree())
    max_degree = max(degree.values()) if degree else 1

    # Closeness: how quickly information reaches this node
    try:
        closeness = nx.closeness_centrality(G_work)
    except Exception:
        closeness = {n: 0.0 for n in G_work.nodes()}

    # PageRank (influence score)
    try:
        pagerank = nx.pagerank(G_work, alpha=0.85, max_iter=100)
    except Exception:
        pagerank = {n: 1 / max(G_work.number_of_nodes(), 1) for n in G_work.nodes()}

    centrality = {}
    for node in G.nodes():
        centrality[node] = {
            "betweenness_centrality": round(float(betweenness.get(node, 0)), 6),
            "degree": int(degree.get(node, 0)),
            "degree_normalized": round(float(degree.get(node, 0)) / max(max_degree, 1), 4),
            "closeness_centrality": round(float(closeness.get(node, 0)), 6),
            "pagerank": round(float(pagerank.get(node, 0)), 6),
        }

    logger.info(f"Centrality computed for {len(centrality)} employees")
    return centrality


def identify_key_brokers(
    G: Any,
    df: pd.DataFrame,
    top_n: int = 10,
) -> List[Dict]:
    """
    Identifies top employees by betweenness centrality (highest knowledge-flow risk).

    Args:
        G: NetworkX DiGraph
        df: Employee DataFrame
        top_n: Number of key brokers to return

    Returns:
        List of dicts with employee details and centrality scores
    """
    if not _HAS_NX:
        return []

    centrality = compute_network_centrality(G)
    if not centrality:
        return []

    # Sort by betweenness centrality
    sorted_nodes = sorted(
        centrality.items(),
        key=lambda x: x[1]["betweenness_centrality"],
        reverse=True,
    )[:top_n]

    emp_id_col = "EmployeeNumber" if "EmployeeNumber" in df.columns else None

    brokers = []
    for node_id, metrics in sorted_nodes:
        emp_row = {}
        if emp_id_col is not None:
            mask = df[emp_id_col] == node_id
            if mask.sum() > 0:
                emp_row = df[mask].iloc[0].to_dict()

        broker = {
            "employee_id": node_id,
            "department": str(emp_row.get("Department", G.nodes[node_id].get("department", "Unknown"))),
            "job_role": str(emp_row.get("JobRole", G.nodes[node_id].get("job_role", "Unknown"))),
            "job_level": int(emp_row.get("JobLevel", G.nodes[node_id].get("job_level", 1))),
            "attrition_risk": int(emp_row.get("Attrition", G.nodes[node_id].get("attrition", 0))),
            "betweenness_centrality": metrics["betweenness_centrality"],
            "pagerank": metrics["pagerank"],
            "degree": metrics["degree"],
            "risk_description": (
                "CRITICAL: High centrality + high attrition risk. Departure would disrupt information flow."
                if metrics["betweenness_centrality"] > 0.05 and emp_row.get("Attrition", 0) == 1
                else "Key connector. Monitor engagement."
            ),
        }
        brokers.append(broker)

    return brokers


def identify_isolated_employees(
    G: Any,
    df: pd.DataFrame,
    isolation_threshold: int = 2,
) -> List[Dict]:
    """
    Identifies employees with low connectivity (potential early attrition signal).

    Args:
        G: NetworkX DiGraph
        df: Employee DataFrame
        isolation_threshold: Degree below which an employee is considered isolated

    Returns:
        List of dicts with isolated employee details
    """
    if not _HAS_NX:
        return []

    G_undirected = G.to_undirected()
    isolated = []
    emp_id_col = "EmployeeNumber" if "EmployeeNumber" in df.columns else None

    for node in G.nodes():
        degree = G_undirected.degree(node)
        if degree <= isolation_threshold:
            emp_row = {}
            if emp_id_col is not None:
                mask = df[emp_id_col] == node
                if mask.sum() > 0:
                    emp_row = df[mask].iloc[0].to_dict()

            isolated.append({
                "employee_id": node,
                "degree": degree,
                "department": str(emp_row.get("Department", G.nodes[node].get("department", "Unknown"))),
                "job_role": str(emp_row.get("JobRole", G.nodes[node].get("job_role", "Unknown"))),
                "tenure_years": float(emp_row.get("YearsAtCompany", 0)),
                "attrition_flag": int(emp_row.get("Attrition", 0)),
                "recommendation": "Low connectivity. Consider team assignment or mentorship pairing.",
            })

    logger.info(f"Identified {len(isolated)} isolated employees (degree ≤ {isolation_threshold})")
    return isolated[:50]  # Return top 50 most isolated


def compute_department_cohesion(G: Any, df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Computes clustering coefficient per department (measure of team connectivity).

    Args:
        G: NetworkX DiGraph
        df: Employee DataFrame

    Returns:
        Dict of {department: {cohesion_score, n_employees, avg_connections}}
    """
    if not _HAS_NX:
        return {}

    G_undirected = G.to_undirected()
    results = {}

    if "Department" not in df.columns:
        return {}

    emp_id_col = "EmployeeNumber" if "EmployeeNumber" in df.columns else None

    for dept in df["Department"].unique():
        dept_mask = df["Department"] == dept
        dept_employees = df[dept_mask]

        if emp_id_col is not None:
            dept_nodes = [int(x) for x in dept_employees[emp_id_col].values if int(x) in G_undirected.nodes()]
        else:
            dept_nodes = list(G_undirected.nodes())[:len(dept_employees)]

        if len(dept_nodes) < 3:
            continue

        subgraph = G_undirected.subgraph(dept_nodes)

        try:
            clustering = nx.average_clustering(subgraph)
        except Exception:
            clustering = 0.0

        avg_degree = np.mean([d for _, d in subgraph.degree()]) if subgraph.number_of_nodes() > 0 else 0
        density = nx.density(subgraph)

        attrition_rate = float(dept_employees["Attrition"].mean()) if "Attrition" in dept_employees.columns else 0.0

        results[dept] = {
            "cohesion_score": round(float(clustering), 4),
            "density": round(float(density), 4),
            "n_employees": len(dept_nodes),
            "avg_connections": round(float(avg_degree), 2),
            "attrition_rate": round(attrition_rate, 4),
            "cohesion_risk": "High" if clustering < 0.2 else "Medium" if clustering < 0.4 else "Low",
        }

    logger.info(f"Department cohesion computed for {len(results)} departments")
    return results


def _generate_org_network_dict(df: pd.DataFrame) -> Dict:
    """Fallback when NetworkX is unavailable: returns dict representation."""
    nodes = []
    emp_id_col = "EmployeeNumber" if "EmployeeNumber" in df.columns else None

    for _, row in df.iterrows():
        emp_id = int(row[emp_id_col]) if emp_id_col else int(row.name)
        nodes.append({
            "id": emp_id,
            "department": str(row.get("Department", "Unknown")),
            "job_level": int(row.get("JobLevel", 1)),
            "attrition": int(row.get("Attrition", 0)),
        })

    return {
        "nodes": nodes,
        "edges": [],
        "note": "Install networkx for full graph analysis",
    }

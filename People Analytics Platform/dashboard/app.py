"""
People Analytics & Workforce Intelligence Dashboard
Streamlit dashboard for attrition prediction, DEI analytics, org network, and workforce planning.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, load_bls_benchmark_rates
from src.features import (
    build_feature_pipeline, create_age_groups,
    engineer_satisfaction_composite, engineer_tenure_features,
    engineer_compensation_features, engineer_flight_risk_signals, engineer_elv,
)
from src.elv_model import compute_elv, segment_employees, compute_retention_roi
from src.dei_analytics import (
    generate_dei_scorecard, compute_attrition_by_group,
    compute_pay_equity_gap, compute_promotion_velocity,
)
from src.network_analysis import (
    generate_org_network, compute_network_centrality,
    identify_key_brokers, identify_isolated_employees, compute_department_cohesion,
)
from src.explainability import explain_flight_risk, generate_intervention_recommendations, generate_hr_narrative

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="People Analytics Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
_portfolio_root = Path(__file__).resolve().parents[2]
if str(_portfolio_root) not in sys.path:
    sys.path.insert(0, str(_portfolio_root))
try:
    from shared.design_system import (
        apply_theme, kpi_card, section_header, chart_layout,
        page_hero, sidebar_brand, risk_badge, gauge_chart, COLORS,
    )
    apply_theme()
except ImportError:
    pass

st.markdown("""
<style>
.risk-critical { color: #FF5A65; font-weight: 700; }
.risk-high     { color: #FFB020; font-weight: 700; }
.risk-medium   { color: #4F8EF7; font-weight: 700; }
.risk-low      { color: #00C896; font-weight: 700; }
.segment-protect { background: #FF5A65; color: white; padding: 2px 10px; border-radius: 20px; font-size:0.72rem; font-weight:700; }
.segment-develop { background: #FFB020; color: #0E1117; padding: 2px 10px; border-radius: 20px; font-size:0.72rem; font-weight:700; }
.segment-monitor { background: #4F8EF7; color: white; padding: 2px 10px; border-radius: 20px; font-size:0.72rem; font-weight:700; }
.segment-maintain { background: #00C896; color: #0E1117; padding: 2px 10px; border-radius: 20px; font-size:0.72rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading workforce model and data...")
def load_model_and_data():
    """Loads model and workforce data (cached)."""
    from src.models import load_artifacts, train_all_models, select_best_model, save_artifacts
    from sklearn.model_selection import train_test_split

    data_dir = str(PROJECT_ROOT / "data")
    model_dir = str(PROJECT_ROOT / "data" / "models")

    df = load_data(data_dir, target_n=10000)

    X, y, feature_cols, protected_cols = build_feature_pipeline(df)

    model = None
    try:
        model, feature_cols_saved = load_artifacts(model_dir, "best_model")
        feature_cols = feature_cols_saved
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        models = train_all_models(X_train, y_train)
        best_name, model = select_best_model(models, X_test, y_test)
        save_artifacts(model, list(feature_cols), model_dir, "best_model")

    # Align and predict
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].fillna(0).astype(float)

    probas = model.predict_proba(X)[:, 1]
    preds = (probas >= 0.5).astype(int)

    # Add predictions and ELV to df
    df_enriched = df.copy()
    df_enriched = create_age_groups(df_enriched)
    df_enriched = engineer_satisfaction_composite(df_enriched)
    df_enriched = engineer_tenure_features(df_enriched)
    df_enriched = engineer_compensation_features(df_enriched)
    df_enriched = engineer_flight_risk_signals(df_enriched)
    df_enriched = compute_elv(df_enriched)

    n = min(len(probas), len(df_enriched))
    df_enriched = df_enriched.head(n).copy()
    df_enriched["attrition_probability"] = probas[:n]
    df_enriched["at_risk"] = preds[:n]

    return model, feature_cols, df_enriched, X, y, probas, preds, protected_cols


def _align_to_model(raw_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in raw_df.columns:
            raw_df[col] = 0.0
    return raw_df[feature_cols].fillna(0).astype(float)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("People Analytics Platform")
    st.sidebar.markdown("**Powered by IBM HR + LightGBM + SHAP**")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Flight Risk Dashboard",
            "Individual Employee",
            "DEI Analytics",
            "Org Network",
            "Workforce Planning",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("SHRM-aligned formulas | BLS JOLTS benchmarks")
    st.sidebar.caption("MIT Applied AI & Data Science")
    return page


# ── Flight Risk Dashboard ──────────────────────────────────────────────────────
def render_flight_risk_dashboard(df_enriched, probas, preds):
    st.title("Flight Risk Dashboard")
    st.markdown("Sortable employee risk table with ELV, segment, and intervention recommendations.")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Employees", f"{len(df_enriched):,}")
    k2.metric("At Risk (>50%)", f"{int(preds.sum()):,}", delta=f"{preds.mean():.1%}")
    k3.metric("Avg Attrition Prob", f"{probas.mean():.1%}")
    k4.metric("Total Replacement Exposure", f"${(df_enriched['replacement_cost'] * probas[:len(df_enriched)]).sum():,.0f}")

    st.divider()

    # Segment filter
    col1, col2, col3 = st.columns(3)
    with col1:
        segment_filter = st.multiselect("Filter by Segment", ["Protect", "Develop", "Monitor", "Maintain"], default=["Protect", "Develop"])
    with col2:
        dept_filter = st.multiselect("Filter by Department", sorted(df_enriched["Department"].unique().tolist()) if "Department" in df_enriched.columns else [], default=[])
    with col3:
        risk_threshold = st.slider("Minimum Attrition Probability", 0.0, 1.0, 0.3, 0.05)

    # Segment employees
    df_seg = segment_employees(df_enriched, probas[:len(df_enriched)])

    # Apply filters
    mask = df_seg["attrition_probability"] >= risk_threshold
    if segment_filter:
        mask &= df_seg["segment"].isin(segment_filter)
    if dept_filter and "Department" in df_seg.columns:
        mask &= df_seg["Department"].isin(dept_filter)

    df_display = df_seg[mask].copy()

    if not _HAS_PLOTLY:
        st.dataframe(df_display[["EmployeeNumber", "Department", "JobRole", "attrition_probability", "elv", "segment"]].head(50))
        return

    # Scatter: ELV vs. attrition probability
    if "elv" in df_display.columns:
        color_map = {"Protect": "#dc3545", "Develop": "#fd7e14", "Monitor": "#ffc107", "Maintain": "#28a745"}
        fig = px.scatter(
            df_display.head(500),
            x="attrition_probability",
            y="elv",
            color="segment",
            color_discrete_map=color_map,
            title="Employee Segmentation: ELV vs. Attrition Risk",
            labels={"attrition_probability": "Attrition Probability", "elv": "Employee Lifetime Value ($)"},
            hover_data=["JobRole", "Department"] if "JobRole" in df_display.columns else None,
            size_max=10,
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50% Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader(f"At-Risk Employees ({len(df_display):,} shown)")
    display_cols = ["EmployeeNumber", "Department", "JobRole", "JobLevel",
                    "attrition_probability", "flight_risk_score", "elv", "segment"]
    available_cols = [c for c in display_cols if c in df_display.columns]

    table_df = df_display[available_cols].sort_values("attrition_probability", ascending=False).head(100)
    if "attrition_probability" in table_df.columns:
        table_df["attrition_probability"] = table_df["attrition_probability"].map("{:.1%}".format)
    if "elv" in table_df.columns:
        table_df["elv"] = table_df["elv"].map("${:,.0f}".format)
    if "flight_risk_score" in table_df.columns:
        table_df["flight_risk_score"] = table_df["flight_risk_score"].map("{:.1f}".format)

    st.dataframe(table_df, use_container_width=True, hide_index=True)


# ── Individual Employee ────────────────────────────────────────────────────────
def render_individual_employee(model, feature_cols, df_enriched, probas):
    st.title("Individual Employee Analysis")

    if "EmployeeNumber" in df_enriched.columns:
        emp_ids = sorted(df_enriched["EmployeeNumber"].unique().tolist())
        selected_id = st.selectbox("Select Employee ID", emp_ids[:200], index=0)
        emp_row = df_enriched[df_enriched["EmployeeNumber"] == selected_id]
    else:
        idx = st.slider("Employee Index", 0, len(df_enriched) - 1, 0)
        emp_row = df_enriched.iloc[[idx]]
        selected_id = idx

    if emp_row.empty:
        st.warning("Employee not found.")
        return

    emp = emp_row.iloc[0]

    # Feature engineering for single employee
    raw = emp_row.copy()
    for f in ["create_age_groups", "engineer_satisfaction_composite", "engineer_tenure_features",
              "engineer_compensation_features", "engineer_flight_risk_signals"]:
        pass

    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole"]
    raw_enc = pd.get_dummies(raw, columns=[c for c in cat_cols if c in raw.columns], dtype=int)
    X_single = _align_to_model(raw_enc, feature_cols)

    prob = float(model.predict_proba(X_single)[0, 1])
    shap_exp = explain_flight_risk(model, X_single, feature_cols, selected_id, 0.5)
    recommendations = generate_intervention_recommendations(shap_exp, raw)

    # ── Display ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attrition Probability", f"{prob:.1%}")
    c2.metric("Flight Risk Score", f"{emp.get('flight_risk_score', 0):.1f}/100")
    c3.metric("ELV", f"${emp.get('elv', 0):,.0f}")
    c4.metric("Replacement Cost", f"${emp.get('replacement_cost', emp.get('MonthlyIncome', 5000) * 18):,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Employee Profile")
        profile_data = {
            "Department": emp.get("Department", "N/A"),
            "Job Role": emp.get("JobRole", "N/A"),
            "Job Level": emp.get("JobLevel", "N/A"),
            "Years at Company": emp.get("YearsAtCompany", "N/A"),
            "Monthly Income": f"${emp.get('MonthlyIncome', 0):,.0f}",
            "OverTime": "Yes" if emp.get("OverTime", 0) == 1 else "No",
            "Satisfaction Score": f"{emp.get('satisfaction_composite', 0):.2f}/4.0",
            "Stagnation": "Yes" if emp.get("stagnation_flag", 0) == 1 else "No",
        }
        for k, v in profile_data.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        # Risk gauge (plotly)
        if _HAS_PLOTLY:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Attrition Risk (%)"},
                delta={"reference": 16},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc3545" if prob >= 0.5 else "#28a745"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 50], "color": "#fff3cd"},
                        {"range": [50, 75], "color": "#f8d7da"},
                        {"range": [75, 100], "color": "#dc3545"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": 50},
                },
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

    # SHAP waterfall
    st.subheader("SHAP Explanation — What's Driving This Risk?")
    all_factors = shap_exp.get("top_risk_factors", []) + shap_exp.get("top_retention_factors", [])
    if all_factors and _HAS_PLOTLY:
        sorted_factors = sorted(all_factors, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:10]
        names = [f.get("feature", "").replace("_", " ").title() for f in sorted_factors]
        vals = [f.get("shap_value", 0) for f in sorted_factors]
        colors = ["#dc3545" if v > 0 else "#28a745" for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
        ))
        fig.update_layout(
            title="SHAP Feature Contributions (Red = increases attrition risk)",
            xaxis_title="SHAP Value",
            height=400,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ELV Breakdown
    st.subheader("Employee Lifetime Value Breakdown")
    elv_cols = st.columns(3)
    elv_cols[0].metric("Annual Contribution", f"${emp.get('annual_contribution', emp.get('MonthlyIncome', 5000) * 12):,.0f}")
    elv_cols[1].metric("Expected Tenure Remaining", f"{emp.get('expected_remaining_tenure', emp.get('expected_tenure_years', 10)):.1f} years")
    elv_cols[2].metric("Net ELV", f"${emp.get('elv', 0):,.0f}")

    # Intervention Recommendations
    st.subheader("Intervention Recommendations")
    if recommendations:
        for rec in recommendations:
            with st.expander(f"Priority {rec['priority']}: {rec['action'][:70]}..."):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Risk Reduction", f"{rec.get('expected_risk_reduction', 0):.0%}")
                col_b.metric("Cost Estimate", f"${rec.get('cost_estimate', 0):,.0f}")
                col_c.metric("Payback Period", str(rec.get("payback_period", "N/A")))
                st.caption(rec.get("root_cause", ""))
    else:
        st.info("No specific interventions recommended at this risk level.")

    # HR Narrative
    st.subheader("HR Narrative Report")
    narrative_data = {
        "employee_id": selected_id,
        "attrition_probability": prob,
        "risk_level": shap_exp.get("risk_level", "Medium"),
        "top_risk_factors": shap_exp.get("top_risk_factors", []),
        "recommendations": recommendations,
        "department": str(emp.get("Department", "N/A")),
        "job_role": str(emp.get("JobRole", "N/A")),
        "years_at_company": float(emp.get("YearsAtCompany", 0)),
        "monthly_income": float(emp.get("MonthlyIncome", 5000)),
        "replacement_cost": float(emp.get("replacement_cost", emp.get("MonthlyIncome", 5000) * 18)),
        "elv": float(emp.get("elv", 0)),
    }
    narrative = generate_hr_narrative(narrative_data)
    st.text_area("Generated HR Report", narrative, height=350)


# ── DEI Analytics ──────────────────────────────────────────────────────────────
def render_dei_analytics(df_enriched, probas):
    st.title("DEI Analytics")
    st.markdown("Attrition disparity, pay equity gap, and promotion velocity by demographic group.")

    with st.spinner("Computing DEI scorecard..."):
        scorecard = generate_dei_scorecard(df_enriched)

    # Summary score
    score = scorecard.get("overall_dei_score", 0)
    s_col, _, _ = st.columns(3)
    s_col.metric("Overall DEI Score", f"{score}/100", delta=f"{scorecard.get('n_passing', 0)}/{scorecard.get('n_checks', 0)} checks passing")

    if scorecard.get("flags"):
        st.warning("**Flags detected:**\n" + "\n".join(f"• {f}" for f in scorecard["flags"]))

    if not _HAS_PLOTLY:
        st.json(scorecard)
        return

    tabs = st.tabs(["Attrition by Group", "Pay Equity", "Promotion Velocity", "Full Scorecard"])

    with tabs[0]:
        st.subheader("Attrition Rate by Demographic Group")
        group_cols = ["Gender", "Age_group", "MaritalStatus", "Department"]
        available = [c for c in group_cols if c in df_enriched.columns]
        attr_data = compute_attrition_by_group(df_enriched, group_cols=available)

        sel_tab = st.radio("Select Grouping", available, horizontal=True) if available else None
        if sel_tab and sel_tab in attr_data:
            data = attr_data[sel_tab]
            fig = px.bar(
                data,
                x=sel_tab,
                y="attrition_rate",
                title=f"Attrition Rate by {sel_tab}",
                color="attrition_rate",
                color_continuous_scale="RdYlGn_r",
                text=[f"{r:.1%}" for r in data["attrition_rate"]],
                labels={"attrition_rate": "Attrition Rate"},
            )
            fig.update_traces(textposition="outside")
            fig.add_hline(y=df_enriched["Attrition"].mean(), line_dash="dash",
                          annotation_text="Overall Average")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(data, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Pay Equity Analysis")
        for group_col in ["Gender", "MaritalStatus"]:
            if group_col not in df_enriched.columns:
                continue
            gap_result = compute_pay_equity_gap(df_enriched, "MonthlyIncome", group_col)
            if "unexplained_gap_pct" in gap_result:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Unexplained Gap ({group_col})", f"{gap_result['unexplained_gap_pct']:.1f}%")
                c2.metric("P-value", f"{gap_result['p_value']:.4f}")
                c3.metric("Significant", "Yes" if gap_result["statistically_significant"] else "No")
                st.caption(gap_result.get("interpretation", ""))

        if "Gender" in df_enriched.columns and "MonthlyIncome" in df_enriched.columns:
            fig = px.violin(
                df_enriched,
                x="Gender",
                y="MonthlyIncome",
                box=True,
                title="Monthly Income Distribution by Gender",
                color="Gender",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Promotion Velocity by Group")
        group_cols_promo = [c for c in ["Gender", "Age_group", "MaritalStatus"] if c in df_enriched.columns]
        promo_data = compute_promotion_velocity(df_enriched, group_cols_promo)

        sel_attr = st.radio("Select Attribute", group_cols_promo, horizontal=True) if group_cols_promo else None
        if sel_attr and sel_attr in promo_data and not isinstance(promo_data[sel_attr], dict):
            pdata = promo_data[sel_attr]
            fig = px.bar(
                pdata,
                x=sel_attr,
                y="avg_years_since_promotion",
                title=f"Avg Years Since Last Promotion by {sel_attr}",
                color="avg_years_since_promotion",
                color_continuous_scale="RdYlGn_r",
                text=[f"{v:.1f}yr" for v in pdata["avg_years_since_promotion"]],
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Full DEI Scorecard")
        st.json(scorecard)


# ── Org Network ────────────────────────────────────────────────────────────────
def render_org_network(df_enriched, probas):
    st.title("Organizational Network Analysis")
    st.markdown("Key brokers, isolated employees, and department cohesion visualization.")

    sample_size = st.slider("Sample size for network (larger = slower)", 100, 1000, 300, 50)
    sample = df_enriched.head(sample_size).copy()

    with st.spinner("Building org network..."):
        G = generate_org_network(sample)
        brokers = identify_key_brokers(G, sample, top_n=10)
        isolated = identify_isolated_employees(G, sample)
        cohesion = compute_department_cohesion(G, sample)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes (Employees)", getattr(G, "number_of_nodes", lambda: sample_size)())
    c2.metric("Edges (Relationships)", getattr(G, "number_of_edges", lambda: 0)())
    c3.metric("Isolated Employees", len(isolated))

    if not _HAS_PLOTLY:
        st.subheader("Key Brokers")
        st.dataframe(pd.DataFrame(brokers))
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Knowledge Brokers")
        if brokers:
            broker_df = pd.DataFrame(brokers)
            cols_to_show = [c for c in ["employee_id", "department", "job_role", "betweenness_centrality", "attrition_risk"] if c in broker_df.columns]
            if cols_to_show:
                fig = px.bar(
                    broker_df.head(10),
                    x="employee_id" if "employee_id" in broker_df.columns else broker_df.index,
                    y="betweenness_centrality" if "betweenness_centrality" in broker_df.columns else broker_df.columns[0],
                    color="attrition_risk" if "attrition_risk" in broker_df.columns else None,
                    title="Top 10 Key Brokers by Betweenness Centrality",
                    labels={"betweenness_centrality": "Betweenness Centrality"},
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(broker_df[cols_to_show], use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Department Cohesion")
        if cohesion:
            cohesion_df = pd.DataFrame([
                {"Department": k, **v} for k, v in cohesion.items()
            ])
            if not cohesion_df.empty and "cohesion_score" in cohesion_df.columns:
                fig = px.bar(
                    cohesion_df.sort_values("cohesion_score"),
                    x="cohesion_score",
                    y="Department",
                    orientation="h",
                    title="Department Cohesion Score",
                    color="cohesion_score",
                    color_continuous_scale="RdYlGn",
                    text=[f"{v:.2f}" for v in cohesion_df.sort_values("cohesion_score")["cohesion_score"]],
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

    # Network visualization using Plotly (scatter-based for compatibility)
    try:
        import networkx as nx
        if isinstance(G, nx.DiGraph):
            st.subheader("Org Network Visualization (Sample: 100 employees)")
            G_sample = G.subgraph(list(G.nodes())[:100])
            pos = nx.spring_layout(G_sample, seed=42)

            centrality = compute_network_centrality(G_sample)
            node_attrs = nx.get_node_attributes(G_sample, "attrition")
            node_sizes = [centrality.get(n, {}).get("degree_normalized", 0.1) * 20 + 5 for n in G_sample.nodes()]
            node_colors = ["red" if node_attrs.get(n, 0) == 1 else "steelblue" for n in G_sample.nodes()]
            node_x = [pos[n][0] for n in G_sample.nodes()]
            node_y = [pos[n][1] for n in G_sample.nodes()]

            edge_x, edge_y = [], []
            for u, v in G_sample.edges():
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#aaa"), hoverinfo="none"))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=node_sizes, color=node_colors, opacity=0.8, line=dict(width=1, color="white")),
                text=[f"ID: {n} | Dept: {nx.get_node_attributes(G_sample, 'department').get(n, 'N/A')}" for n in G_sample.nodes()],
                hoverinfo="text",
            ))
            fig.update_layout(
                title="Org Network (Red = Attrition Risk, Size = Centrality)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.caption(f"Graph visualization unavailable: {exc}")


# ── Workforce Planning ─────────────────────────────────────────────────────────
def render_workforce_planning(df_enriched, probas):
    st.title("Workforce Planning & Monte Carlo Forecast")
    st.markdown("Project headcount under different retention scenarios.")

    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        n_months = st.slider("Forecast Horizon (months)", 6, 36, 24, 6)
    with c2:
        scenario = st.selectbox(
            "Retention Scenario",
            ["Baseline (No Action)", "Targeted Intervention (Top 20% Risk)", "Salary Adjustment (+10%)", "Remote Work Policy"],
        )
    with c3:
        n_simulations = st.slider("Monte Carlo Simulations", 100, 1000, 500, 100)

    # Scenario parameters
    scenario_params = {
        "Baseline (No Action)": {"risk_multiplier": 1.0, "cost": 0},
        "Targeted Intervention (Top 20% Risk)": {"risk_multiplier": 0.65, "cost": 5000},
        "Salary Adjustment (+10%)": {"risk_multiplier": 0.80, "cost": 8000},
        "Remote Work Policy": {"risk_multiplier": 0.85, "cost": 2000},
    }
    params = scenario_params[scenario]

    # Compute ROI
    df_with_elv = compute_elv(df_enriched)
    roi_result = compute_retention_roi(
        df_with_elv,
        probas[:len(df_enriched)],
        intervention_cost_per_employee=params["cost"] if params["cost"] > 0 else 5000,
    )

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected Attritions (12mo)", f"{roi_result['expected_attritions']:.0f}")
    k2.metric("Intervention ROI", f"{roi_result['roi_pct']:.0f}%")
    k3.metric("Expected Savings", f"${roi_result['expected_savings']:,.0f}")
    k4.metric("Payback Period", f"{roi_result['payback_months']:.0f} months")

    if not _HAS_PLOTLY:
        st.json(roi_result)
        return

    # Monte Carlo simulation
    rng = np.random.default_rng(42)
    n_employees = len(df_enriched)
    adjusted_probas = np.clip(probas[:n_employees] * params["risk_multiplier"], 0, 1)

    monthly_attrition_rate = adjusted_probas / 12  # Convert annual to monthly
    headcount_trajectories = np.zeros((n_simulations, n_months + 1))
    headcount_trajectories[:, 0] = n_employees

    for sim in range(n_simulations):
        current_headcount = n_employees
        for month in range(1, n_months + 1):
            departures = (rng.uniform(0, 1, current_headcount) < monthly_attrition_rate[:current_headcount]).sum()
            current_headcount = max(current_headcount - int(departures), 0)
            headcount_trajectories[sim, month] = current_headcount

    # Plot trajectories
    months_x = list(range(n_months + 1))
    mean_traj = headcount_trajectories.mean(axis=0)
    p10_traj = np.percentile(headcount_trajectories, 10, axis=0)
    p90_traj = np.percentile(headcount_trajectories, 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months_x, y=p90_traj, fill=None, mode="lines",
        line=dict(color="lightblue", dash="dot"), name="P90 (Optimistic)",
    ))
    fig.add_trace(go.Scatter(
        x=months_x, y=p10_traj, fill="tonexty", mode="lines",
        line=dict(color="lightblue", dash="dot"), name="P10 (Pessimistic)",
        fillcolor="rgba(173,216,230,0.3)",
    ))
    fig.add_trace(go.Scatter(
        x=months_x, y=mean_traj, mode="lines+markers",
        line=dict(color="#1f77b4", width=2.5), name="Mean Forecast",
    ))
    fig.add_hline(y=n_employees * 0.90, line_dash="dash", line_color="orange",
                  annotation_text="90% of Current Headcount")
    fig.update_layout(
        title=f"Monte Carlo Headcount Forecast — {scenario} ({n_simulations} simulations)",
        xaxis_title="Months",
        yaxis_title="Headcount",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Forecast Summary")
    summary_df = pd.DataFrame({
        "Month": [6, 12, 18, 24, 36] if n_months >= 36 else [6, 12, n_months],
        "Mean Headcount": [int(mean_traj[min(m, n_months)]) for m in ([6, 12, 18, 24, 36] if n_months >= 36 else [6, 12, n_months])],
        "P10 (Pessimistic)": [int(p10_traj[min(m, n_months)]) for m in ([6, 12, 18, 24, 36] if n_months >= 36 else [6, 12, n_months])],
        "P90 (Optimistic)": [int(p90_traj[min(m, n_months)]) for m in ([6, 12, 18, 24, 36] if n_months >= 36 else [6, 12, n_months])],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # BLS Benchmarks
    st.subheader("BLS JOLTS Industry Benchmarks (2023)")
    bls = load_bls_benchmark_rates()
    bls_df = pd.DataFrame(list(bls.items()), columns=["Industry", "Annual Attrition Rate"])
    bls_df["Annual Attrition Rate"] = bls_df["Annual Attrition Rate"].map("{:.1%}".format)
    company_rate = roi_result["expected_attrition_rate"]
    bls_df["vs. Company"] = bls_df["Industry"].apply(
        lambda x: f"↑ Company is {'higher' if company_rate > bls[x] else 'lower'}" if x in bls else ""
    )
    st.dataframe(bls_df, use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    try:
        model, feature_cols, df_enriched, X, y, probas, preds, protected_cols = load_model_and_data()
    except Exception as exc:
        st.error(f"Failed to load data/model: {exc}")
        st.stop()

    if page == "Flight Risk Dashboard":
        render_flight_risk_dashboard(df_enriched, probas, preds)
    elif page == "Individual Employee":
        render_individual_employee(model, feature_cols, df_enriched, probas)
    elif page == "DEI Analytics":
        render_dei_analytics(df_enriched, probas)
    elif page == "Org Network":
        render_org_network(df_enriched, probas)
    elif page == "Workforce Planning":
        render_workforce_planning(df_enriched, probas)


if __name__ == "__main__":
    main()

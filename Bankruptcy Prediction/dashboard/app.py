"""
dashboard/app.py
================
Streamlit dashboard for Enterprise Financial Health & Supply Chain Risk Platform.

Pages:
  1. Company Risk Assessment  — ratio input → risk scorecard, Altman gauge, SHAP waterfall
  2. Portfolio Monitor        — company table, sector heatmap, time-series
  3. Supply Chain Network     — Plotly network graph, contagion risk table
  4. Sector Analysis          — box plots, peer comparison
  5. Early Warning System     — companies entering distress zone
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    compute_altman_zscore,
    compute_altman_zscore_single,
    generate_synthetic_financial_data,
    load_data,
)
from src.explainability import compute_counterfactual, explain_company_risk
from src.features import build_feature_pipeline, engineer_altman_components, engineer_financial_ratios
from src.models import evaluate_model, train_multi_horizon_models, train_xgboost_distress
from src.network_analysis import (
    compute_contagion_risk,
    generate_supply_chain_network,
    identify_systemic_risk_nodes,
)

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Cached resources
# -------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training financial distress models…")
def get_models_and_data():
    df = load_data(str(PROJECT_ROOT / "data"))
    df = compute_altman_zscore(df)
    X, y_dict, feature_cols = build_feature_pipeline(df)
    df = df.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    models = train_multi_horizon_models(X, y_dict)
    return models, feature_cols, df, X, y_dict


@st.cache_data(show_spinner="Building supply chain network…")
def get_supply_chain(df: pd.DataFrame):
    latest = df.groupby("company_id").last().reset_index()
    G = generate_supply_chain_network(latest, n_suppliers_per_company=3)
    return G


models, feature_cols, ref_df, ref_X, ref_y_dict = get_models_and_data()

# Pre-compute distress probabilities for reference companies
primary_model = models.get("distress_12m", next(iter(models.values())))

@st.cache_data(show_spinner="Computing portfolio risk scores…")
def compute_all_probas(_X: pd.DataFrame, _feature_cols: List[str]):
    X_aligned = _X.copy()
    for col in _feature_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0
    X_aligned = X_aligned[_feature_cols].fillna(0)
    return primary_model.predict_proba(X_aligned)[:, 1]


all_probas = compute_all_probas(ref_X, feature_cols)

# Build portfolio view DataFrame
portfolio_df = ref_df.groupby("company_id").last().reset_index()
if len(all_probas) == len(ref_df):
    temp = ref_df.copy()
    temp["proba_12m"] = all_probas
    portfolio_df = portfolio_df.merge(
        temp.groupby("company_id")["proba_12m"].last().reset_index(),
        on="company_id", how="left",
    )
else:
    portfolio_df["proba_12m"] = 0.5

portfolio_df["risk_level"] = portfolio_df["proba_12m"].apply(
    lambda p: "Critical" if p >= 0.70 else "High" if p >= 0.45 else "Elevated" if p >= 0.25 else "Low"
)


def risk_color_map(risk: str) -> str:
    return {"Critical": "#d62728", "High": "#ff7f0e", "Elevated": "#bcbd22", "Low": "#2ca02c"}.get(risk, "#1f77b4")


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
st.sidebar.title("Financial Risk Intelligence")
st.sidebar.markdown("Enterprise Financial Health & Supply Chain Risk Platform")
page = st.sidebar.radio(
    "Navigate to",
    [
        "Company Risk Assessment",
        "Portfolio Monitor",
        "Supply Chain Network",
        "Sector Analysis",
        "Early Warning System",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Context: Altman Z-Score + ML + Sentiment Analysis")


# =========================================================================
# PAGE 1: Company Risk Assessment
# =========================================================================

if page == "Company Risk Assessment":
    st.title("Company Risk Assessment")
    st.markdown("Enter financial ratios to assess distress probability across 3–18 month horizons.")

    with st.form("risk_form"):
        company_name = st.text_input("Company Name", value="Acme Corp")
        sector = st.selectbox("Sector", ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrial"])

        st.subheader("Profitability")
        col1, col2 = st.columns(2)
        roa = col1.number_input("ROA", min_value=-0.50, max_value=0.50, value=0.05, format="%.3f")
        roe = col2.number_input("ROE", min_value=-1.0, max_value=1.0, value=0.10, format="%.3f")
        npm = col1.number_input("Net Profit Margin", min_value=-0.50, max_value=0.60, value=0.04, format="%.3f")
        ebitda_m = col2.number_input("EBITDA Margin", min_value=-0.50, max_value=0.60, value=0.10, format="%.3f")

        st.subheader("Leverage")
        col3, col4 = st.columns(2)
        d2e = col3.number_input("Debt/Equity", min_value=0.0, max_value=20.0, value=2.0, format="%.2f")
        d2a = col4.number_input("Debt/Assets", min_value=0.0, max_value=1.5, value=0.55, format="%.3f")
        int_cov = col3.number_input("Interest Coverage", min_value=-10.0, max_value=100.0, value=4.0, format="%.1f")

        st.subheader("Liquidity")
        col5, col6 = st.columns(2)
        cr = col5.number_input("Current Ratio", min_value=0.0, max_value=20.0, value=1.3, format="%.2f")
        qr = col6.number_input("Quick Ratio", min_value=0.0, max_value=15.0, value=0.9, format="%.2f")

        st.subheader("Activity & Sentiment")
        col7, col8 = st.columns(2)
        at = col7.number_input("Asset Turnover", min_value=0.0, max_value=10.0, value=0.8, format="%.2f")
        sentiment = col8.slider("News Sentiment (−1 to +1)", min_value=-1.0, max_value=1.0, value=0.1, step=0.05)

        st.subheader("Altman Z-Score Inputs (optional)")
        col9, col10 = st.columns(2)
        wc_ta = col9.number_input("Working Capital / Total Assets", value=0.15, format="%.3f")
        re_ta = col10.number_input("Retained Earnings / Total Assets", value=0.12, format="%.3f")
        ebit_ta = col9.number_input("EBIT / Total Assets", value=roa, format="%.3f")
        mve_tl = col10.number_input("Market Cap / Total Liabilities", value=1.0, format="%.2f")
        sales_ta = col9.number_input("Sales / Total Assets", value=at, format="%.2f")

        submitted = st.form_submit_button("Assess Risk", type="primary")

    if submitted:
        # Compute Altman Z-Score
        z_score = compute_altman_zscore_single(wc_ta, re_ta, ebit_ta, mve_tl, sales_ta)
        altman_zone = "Safe" if z_score > 2.99 else "Grey Zone" if z_score >= 1.81 else "Distress Zone"

        # Build feature row
        row_data = {
            "ROA": roa, "ROE": roe, "net_profit_margin": npm, "EBITDA_margin": ebitda_m,
            "debt_to_equity": d2e, "debt_to_assets": d2a, "interest_coverage": int_cov,
            "current_ratio": cr, "quick_ratio": qr, "cash_ratio": 0.3,
            "asset_turnover": at, "inventory_turnover": 6.0, "receivables_turnover": 8.0,
            "revenue_growth_yoy": 0.05, "asset_growth_yoy": 0.03,
            "news_sentiment_30d": sentiment,
            "altman_wc_ta": wc_ta, "altman_re_ta": re_ta, "altman_ebit_ta": ebit_ta,
            "altman_mve_tl": mve_tl, "altman_sales_ta": sales_ta,
            "altman_zscore": z_score,
        }
        df_row = pd.DataFrame([row_data])
        df_row = engineer_altman_components(df_row)
        for col in feature_cols:
            if col not in df_row.columns:
                df_row[col] = 0.0
        df_row = df_row[feature_cols].fillna(0)

        # Multi-horizon predictions
        horizon_probs: Dict[str, float] = {}
        for h in ["distress_3m", "distress_6m", "distress_12m", "distress_18m"]:
            if h in models:
                horizon_probs[h] = float(models[h].predict_proba(df_row)[:, 1][0])

        primary_prob = horizon_probs.get("distress_12m", 0.5)
        risk_level = ("Critical" if primary_prob >= 0.70 else "High" if primary_prob >= 0.45
                      else "Elevated" if primary_prob >= 0.25 else "Low")

        st.markdown("---")
        st.subheader(f"Risk Assessment: {company_name}")

        col_gauge, col_metrics = st.columns([1, 2])

        with col_gauge:
            color = risk_color_map(risk_level)
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=primary_prob * 100,
                title={"text": "12-Month Distress Risk", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 25], "color": "#c7e9c0"},
                        {"range": [25, 45], "color": "#fdae6b"},
                        {"range": [45, 70], "color": "#fc8d59"},
                        {"range": [70, 100], "color": "#d62728"},
                    ],
                },
                number={"suffix": "%"},
            ))
            gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_metrics:
            st.markdown(f"**Risk Level**: {risk_level}")

            # Horizon table
            horizon_display = {
                "3 Months": horizon_probs.get("distress_3m", 0),
                "6 Months": horizon_probs.get("distress_6m", 0),
                "12 Months": horizon_probs.get("distress_12m", 0),
                "18 Months": horizon_probs.get("distress_18m", 0),
            }
            h_df = pd.DataFrame(list(horizon_display.items()), columns=["Horizon", "Distress Probability"])
            h_df["Distress Probability"] = h_df["Distress Probability"].apply(lambda p: f"{p:.1%}")
            st.table(h_df)

            # Altman Z-Score
            z_color = "🟢" if z_score > 2.99 else "🟡" if z_score >= 1.81 else "🔴"
            st.markdown(f"**Altman Z-Score**: {z_score:.2f} {z_color} ({altman_zone})")

        # SHAP explanation
        explanation = explain_company_risk(
            primary_model, df_row, feature_cols, company_name=company_name,
            altman_zscore=z_score, sector=sector,
        )

        st.markdown("---")
        col_risk, col_protect = st.columns(2)
        with col_risk:
            st.subheader("Top Risk Drivers")
            risk_data = pd.DataFrame(explanation["top_risk_factors"])
            if len(risk_data) > 0:
                fig_risk = px.bar(risk_data, x="shap_contribution", y="factor",
                                   orientation="h", title="Risk-Increasing Factors",
                                   color="shap_contribution", color_continuous_scale="Reds")
                fig_risk.update_layout(height=300)
                st.plotly_chart(fig_risk, use_container_width=True)

        with col_protect:
            st.subheader("Protective Factors")
            protect_data = pd.DataFrame(explanation["top_protective_factors"])
            if len(protect_data) > 0:
                fig_prot = px.bar(protect_data, x="shap_contribution", y="factor",
                                   orientation="h", title="Risk-Reducing Factors",
                                   color="shap_contribution", color_continuous_scale="Greens_r")
                fig_prot.update_layout(height=300)
                st.plotly_chart(fig_prot, use_container_width=True)

        st.info(explanation["narrative"])

        # Counterfactual
        with st.expander("What Would Reduce This Company's Risk?"):
            cf = compute_counterfactual(primary_model, df_row, feature_cols)
            st.markdown(f"**{cf['message']}**")
            if cf["required_changes"]:
                cf_df = pd.DataFrame(cf["required_changes"])
                st.dataframe(cf_df, use_container_width=True)


# =========================================================================
# PAGE 2: Portfolio Monitor
# =========================================================================

elif page == "Portfolio Monitor":
    st.title("Portfolio Monitor")

    # Summary metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Companies", len(portfolio_df))
    critical_n = (portfolio_df["risk_level"].isin(["Critical", "High"])).sum()
    col_m2.metric("Critical/High Risk", critical_n, delta=None)
    col_m3.metric("Avg 12M Distress Prob", f"{portfolio_df['proba_12m'].mean():.1%}")
    if "altman_zscore" in portfolio_df.columns:
        distress_zone_n = (portfolio_df["altman_zscore"] < 1.81).sum()
        col_m4.metric("Altman Distress Zone", distress_zone_n)

    # Risk table
    st.subheader("Company Risk Scorecard")
    display_cols = ["company_id", "sector", "proba_12m", "risk_level"]
    if "altman_zscore" in portfolio_df.columns:
        display_cols.append("altman_zscore")
    if "ROA" in portfolio_df.columns:
        display_cols.extend(["ROA", "current_ratio", "debt_to_equity"])

    display_df = portfolio_df[display_cols].copy()
    display_df["proba_12m"] = display_df["proba_12m"].apply(lambda p: f"{p:.1%}")
    if "altman_zscore" in display_df.columns:
        display_df["altman_zscore"] = display_df["altman_zscore"].round(2)

    st.dataframe(display_df.head(100), use_container_width=True)

    # Sector × Risk heatmap
    st.subheader("Sector Risk Heatmap")
    if "sector" in portfolio_df.columns:
        heatmap_data = portfolio_df.pivot_table(
            values="proba_12m", index="sector", columns="risk_level",
            aggfunc="count", fill_value=0,
        )
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale="Reds",
            title="Company Count by Sector × Risk Level",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Portfolio risk over time (synthetic trend)
    st.subheader("Portfolio Average Distress Risk Trend")
    if "year" in ref_df.columns:
        year_risk = ref_df.groupby("year")["distress_flag"].mean().reset_index()
        fig_trend = px.line(year_risk, x="year", y="distress_flag",
                            title="Average Distress Rate by Year",
                            labels={"distress_flag": "Distress Rate", "year": "Year"},
                            markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)


# =========================================================================
# PAGE 3: Supply Chain Network
# =========================================================================

elif page == "Supply Chain Network":
    st.title("Supply Chain Network")

    try:
        G = get_supply_chain(ref_df)

        # Subsample for visualization
        n_display = st.slider("Companies to display (subsample)", 20, 200, 80)
        company_probas = portfolio_df.set_index("company_id")["proba_12m"]

        # Select top-risk companies for display
        top_risk_companies = portfolio_df.nlargest(n_display, "proba_12m")["company_id"].tolist()
        subgraph_nodes = set(top_risk_companies)
        # Add their neighbors
        for node in top_risk_companies[:30]:
            if node in G:
                subgraph_nodes.update(G.predecessors(node))
                subgraph_nodes.update(G.successors(node))
        subgraph_nodes = list(subgraph_nodes)[:n_display]

        # Build Plotly network
        try:
            import networkx as nx
            sub_G = G.subgraph(subgraph_nodes)
            pos = nx.spring_layout(sub_G, seed=42, k=0.5)

            edge_x, edge_y = [], []
            for u, v in sub_G.edges():
                x0, y0 = pos.get(u, (0, 0))
                x1, y1 = pos.get(v, (0, 0))
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x = [pos[n][0] for n in sub_G.nodes if n in pos]
            node_y = [pos[n][1] for n in sub_G.nodes if n in pos]
            node_proba = [float(company_probas.get(n, 0)) for n in sub_G.nodes if n in pos]
            node_ids = [n for n in sub_G.nodes if n in pos]

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.5, color="lightgray"),
                hoverinfo="none",
            ))
            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=[f"{n[:8]}..." for n in node_ids],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=node_proba,
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="Distress Prob"),
                    showscale=True,
                    cmin=0, cmax=1,
                ),
                hovertext=[f"{n}: {p:.1%}" for n, p in zip(node_ids, node_proba)],
                hoverinfo="text",
            ))
            fig_net.update_layout(
                title="Supply Chain Network (colored by 12M distress risk)",
                showlegend=False,
                height=600,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_net, use_container_width=True)

        except ImportError:
            st.warning("networkx required for network visualization.")

        # Contagion risk table
        st.subheader("Contagion Risk Table")
        contagion_df = compute_contagion_risk(G, ref_df, company_probas)
        contagion_df = contagion_df.sort_values("total_contagion_risk", ascending=False)
        st.dataframe(contagion_df.head(30), use_container_width=True)

        # Systemic risk
        st.subheader("Top 20 Systemic Risk Nodes")
        systemic_df = identify_systemic_risk_nodes(G, ref_df, company_probas, top_n=20)
        st.dataframe(systemic_df.round(4), use_container_width=True)

    except Exception as exc:
        st.error(f"Network analysis unavailable: {exc}")
        st.info("Install networkx: `pip install networkx`")


# =========================================================================
# PAGE 4: Sector Analysis
# =========================================================================

elif page == "Sector Analysis":
    st.title("Sector Analysis")

    if "sector" not in portfolio_df.columns:
        st.warning("No sector data available.")
    else:
        # Box plots by sector
        st.subheader("Distress Probability Distribution by Sector")
        fig_box = px.box(
            portfolio_df, x="sector", y="proba_12m",
            color="sector",
            title="12-Month Distress Probability by Sector",
            labels={"proba_12m": "Distress Probability", "sector": "Sector"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Peer comparison
        st.subheader("Peer Comparison")
        selected_sector = st.selectbox("Select Sector", portfolio_df["sector"].unique())
        sector_data = portfolio_df[portfolio_df["sector"] == selected_sector]

        numeric_display = ["proba_12m", "ROA", "current_ratio", "debt_to_equity", "altman_zscore"]
        available_display = [c for c in numeric_display if c in sector_data.columns]

        if available_display:
            sector_summary = sector_data[available_display].describe().round(4)
            st.dataframe(sector_summary, use_container_width=True)

        # Altman Z distribution by sector
        if "altman_zscore" in portfolio_df.columns:
            fig_alt = px.violin(
                portfolio_df, y="altman_zscore", x="sector",
                box=True, color="sector",
                title="Altman Z-Score Distribution by Sector",
            )
            fig_alt.add_hline(y=2.99, line_dash="dash", annotation_text="Safe threshold (2.99)",
                               line_color="green")
            fig_alt.add_hline(y=1.81, line_dash="dash", annotation_text="Distress threshold (1.81)",
                               line_color="red")
            st.plotly_chart(fig_alt, use_container_width=True)


# =========================================================================
# PAGE 5: Early Warning System
# =========================================================================

elif page == "Early Warning System":
    st.title("Early Warning System")
    st.markdown("Companies entering distress or grey zone in the latest monitoring period.")

    if "altman_zscore" in portfolio_df.columns:
        # Companies in distress zone
        distress_zone = portfolio_df[portfolio_df["altman_zscore"] < 1.81].sort_values("proba_12m", ascending=False)
        grey_zone = portfolio_df[(portfolio_df["altman_zscore"] >= 1.81) & (portfolio_df["altman_zscore"] <= 2.99)]
        grey_zone = grey_zone.sort_values("proba_12m", ascending=False)

        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Distress Zone Companies", len(distress_zone), delta=None)
        col_w2.metric("Grey Zone Companies", len(grey_zone))
        col_w3.metric("High ML Risk (>70%)", int((portfolio_df["proba_12m"] >= 0.70).sum()))

        tab_distress, tab_grey, tab_trends = st.tabs(["Distress Zone", "Grey Zone", "Trend Analysis"])

        with tab_distress:
            st.subheader("Companies in Altman Distress Zone (Z < 1.81)")
            display_cols = ["company_id", "sector", "altman_zscore", "proba_12m", "risk_level"]
            if "ROA" in distress_zone.columns:
                display_cols.extend(["ROA", "current_ratio", "debt_to_equity"])
            available = [c for c in display_cols if c in distress_zone.columns]
            if len(distress_zone) > 0:
                st.dataframe(distress_zone[available].round(4), use_container_width=True)
            else:
                st.success("No companies in distress zone.")

        with tab_grey:
            st.subheader("Companies in Altman Grey Zone (1.81 ≤ Z ≤ 2.99)")
            if len(grey_zone) > 0:
                available_grey = [c for c in display_cols if c in grey_zone.columns]
                st.dataframe(grey_zone[available_grey].head(50).round(4), use_container_width=True)
            else:
                st.success("No companies in grey zone.")

        with tab_trends:
            st.subheader("Altman Z-Score Trend Analysis")
            if "year" in ref_df.columns and "altman_zscore" in ref_df.columns:
                zone_trend = ref_df.groupby("year").apply(
                    lambda g: pd.Series({
                        "distress_pct": (g["altman_zscore"] < 1.81).mean() * 100,
                        "grey_pct": ((g["altman_zscore"] >= 1.81) & (g["altman_zscore"] <= 2.99)).mean() * 100,
                        "safe_pct": (g["altman_zscore"] > 2.99).mean() * 100,
                    })
                ).reset_index()
                fig_trend = px.area(
                    zone_trend.melt(id_vars="year", var_name="Zone", value_name="Percentage"),
                    x="year", y="Percentage", color="Zone",
                    title="Altman Zone Distribution Over Time (%)",
                    color_discrete_map={"distress_pct": "red", "grey_pct": "orange", "safe_pct": "green"},
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Altman Z-Score data not available in the current dataset.")
        st.info("The early warning system requires Altman Z-Score computation.")

"""P06 Supply Chain Risk Intelligence — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


@st.cache_data
def load_distress_data(proj_path: Path) -> pd.DataFrame:
    try:
        for fname in ["distress_scores.csv", "company_scores.csv", "risk_scores.csv"]:
            p = proj_path / "data" / "processed" / fname
            if p.exists():
                return pd.read_csv(p)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    n = 500
    sectors = ["Technology", "Retail", "Manufacturing", "Energy", "Healthcare", "Finance"]
    return pd.DataFrame({
        "company": [f"Company_{i:03d}" for i in range(n)],
        "sector": rng.choice(sectors, n),
        "distress_prob": rng.beta(2, 5, n).round(4),
        "altman_z": rng.uniform(-2, 8, n).round(3),
    })


def call_ollama(prompt: str) -> str:
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 200},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None


def render():
    proj = PROJECT_BY_ID["supply_chain"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Companies Monitored", metrics.get("n_companies", "2,847"), color="#1ABC9C")
    with cols[1]: metric_card("High-Risk Count", metrics.get("high_risk_count", "341"), color="#E74C3C")
    with cols[2]: metric_card("Avg Distress Prob", metrics.get("avg_distress", "12.4%"), color="#F39C12")
    with cols[3]: metric_card("Llama-Powered", "Yes", color="#8E44AD")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Financial Distress Probability Distribution")
        df = load_distress_data(proj["path"])
        fig = go.Figure(go.Histogram(
            x=df["distress_prob"], nbinsx=40,
            marker_color="#1ABC9C", opacity=0.8
        ))
        dark_fig(fig, height=320)
        fig.update_layout(title="Distribution of Distress Probabilities (All Companies)",
                           xaxis_title="Distress Probability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        high_risk = df[df["distress_prob"] > 0.5].shape[0]
        st.info(f"**{high_risk}** companies above 50% distress threshold ({high_risk/len(df)*100:.1f}% of monitored universe)")

    with tab2:
        section_header("Distress Probability Calculator")
        st.caption("Enter financial ratios to estimate distress probability.")
        c1, c2 = st.columns(2)
        with c1:
            working_capital_ratio = st.slider("Working Capital / Total Assets", -0.5, 0.8, 0.15, step=0.01)
            retained_earnings_ratio = st.slider("Retained Earnings / Total Assets", -1.0, 0.8, 0.12, step=0.01)
            ebit_ratio = st.slider("EBIT / Total Assets", -0.3, 0.4, 0.08, step=0.01)
        with c2:
            equity_debt_ratio = st.slider("Market Value Equity / Book Value Debt", 0.1, 10.0, 1.5, step=0.1)
            sales_ratio = st.slider("Sales / Total Assets", 0.1, 3.0, 0.9, step=0.05)
            current_ratio = st.slider("Current Ratio", 0.5, 4.0, 1.8, step=0.1)

        if st.button("Calculate Distress Risk", type="primary"):
            altman_z = (1.2 * working_capital_ratio + 1.4 * retained_earnings_ratio +
                        3.3 * ebit_ratio + 0.6 * equity_debt_ratio + 1.0 * sales_ratio)
            if altman_z > 2.99:
                zone, color, prob = "Safe Zone", "#27AE60", max(0.02, 1 - altman_z / 10)
            elif altman_z > 1.81:
                zone, color, prob = "Grey Zone", "#F39C12", 0.35
            else:
                zone, color, prob = "Distress Zone", "#E74C3C", min(0.95, 1 - altman_z / 5)

            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Altman Z-Score", f"{altman_z:.2f}", color=color)
            with c2: metric_card("Risk Zone", zone, color=color)
            with c3: metric_card("Distress Probability", f"{prob:.1%}", color=color)

            # Ollama narrative
            narrative = call_ollama(
                f"In 2 sentences, explain what an Altman Z-score of {altman_z:.2f} means for a company's financial health."
            )
            if narrative:
                st.markdown(f"> **Llama Analysis:** {narrative}")
            else:
                st.caption("Llama narrative unavailable (Ollama not running). Start with `ollama serve`.")

    with tab3:
        section_header("Risk Heatmap by Sector")
        df = load_distress_data(proj["path"])
        sector_risk = df.groupby("sector")["distress_prob"].agg(
            Avg="mean", High_Risk=lambda x: (x > 0.5).sum(), Count="count"
        ).reset_index()
        sector_risk["Avg"] = sector_risk["Avg"].round(3)

        fig = px.bar(sector_risk, x="sector", y="Avg", color="Avg",
                     color_continuous_scale=[[0, "#27AE60"], [0.5, "#F39C12"], [1, "#E74C3C"]],
                     text="Avg")
        dark_fig(fig, height=320)
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(title="Average Distress Probability by Sector", xaxis_title="Sector",
                           yaxis_title="Avg Distress Prob", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

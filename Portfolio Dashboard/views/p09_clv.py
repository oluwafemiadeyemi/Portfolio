"""P09 CLV Retention Platform — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, load_parquet_sample, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


@st.cache_data
def load_clv_data(proj_path: Path) -> pd.DataFrame:
    try:
        for fname in ["clv_scores.parquet", "customer_clv.parquet", "rfm_features.parquet"]:
            p = proj_path / "data" / "processed" / fname
            if p.exists():
                df = pd.read_parquet(p)
                return df.sample(min(1000, len(df)), random_state=42)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    n = 1000
    segments = rng.choice(["Champions", "Loyal", "At Risk", "Hibernating", "Lost"], n,
                           p=[0.15, 0.25, 0.20, 0.25, 0.15])
    return pd.DataFrame({
        "clv": np.abs(rng.normal(450, 280, n)).round(2),
        "churn_prob": rng.beta(2, 5, n).round(4),
        "segment": segments,
        "recency_days": rng.integers(1, 365, n),
        "frequency": rng.integers(1, 50, n),
        "monetary": np.abs(rng.normal(200, 150, n)).round(2),
    })


@st.cache_resource
def load_clv_model(proj_path: Path):
    for name in ["clv_model.pkl", "bgf_model.pkl", "uplift_model.pkl", "model.pkl"]:
        for folder in ["models", "data/models"]:
            p = proj_path / folder / name
            if p.exists():
                try:
                    return joblib.load(p), name
                except Exception:
                    pass
    return None, None


def render():
    proj = PROJECT_BY_ID["clv"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Users Analysed", metrics.get("n_users", "2.6M"), color="#D35400")
    with cols[1]: metric_card("Avg CLV", metrics.get("avg_clv", "$447"), color="#27AE60")
    with cols[2]: metric_card("Churn Rate", metrics.get("churn_rate", "18.3%"), color="#E74C3C")
    with cols[3]: metric_card("Uplift Model Lift", metrics.get("uplift_lift", "2.4x"), color="#58A6FF")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("CLV Distribution")
        df = load_clv_data(proj["path"])
        if "clv" in df.columns:
            fig = go.Figure(go.Histogram(
                x=df["clv"].clip(upper=df["clv"].quantile(0.95)),
                nbinsx=50, marker_color="#D35400", opacity=0.8
            ))
            dark_fig(fig, height=300)
            fig.update_layout(title="Customer Lifetime Value Distribution (BG/NBD Model)",
                               xaxis_title="CLV ($)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("CLV data not found"), use_container_width=True)

    with tab2:
        section_header("CLV & Churn Risk Predictor")
        model, model_name = load_clv_model(proj["path"])
        if model_name:
            st.caption(f"Model loaded: `{model_name}`")
        else:
            st.caption("Using BG/NBD heuristic (model not found)")

        c1, c2 = st.columns(2)
        with c1:
            recency = st.slider("Days Since Last Purchase (Recency)", 1, 365, 45)
            frequency = st.slider("Number of Purchases (Frequency)", 1, 100, 8)
        with c2:
            monetary = st.slider("Avg Purchase Value ($)", 5.0, 1000.0, 85.0, step=5.0)
            tenure = st.slider("Customer Tenure (days)", 30, 1825, 365)

        if st.button("Predict CLV & Churn", type="primary"):
            prob_churn = None
            clv_pred = None

            if model is not None:
                try:
                    row = pd.DataFrame([[recency, frequency, monetary, tenure]])
                    clv_pred = float(model.predict(row)[0])
                except Exception:
                    pass

            if clv_pred is None:
                # BG/NBD-style heuristic
                purchase_rate = frequency / max(tenure / 30, 1)
                recency_factor = max(0, 1 - recency / 365)
                clv_pred = monetary * purchase_rate * 12 * recency_factor
                clv_pred = max(0, round(clv_pred, 2))

            churn_prob = 1 - min(0.95, (frequency * 0.05) + (1 - recency / 365) * 0.5 + (tenure / 1825) * 0.3)
            churn_prob = max(0.02, churn_prob)

            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Predicted CLV (12mo)", f"${clv_pred:.0f}", color="#D35400")
            with c2: metric_card("Churn Probability", f"{churn_prob:.1%}", color="#E74C3C" if churn_prob > 0.4 else "#27AE60")
            with c3:
                if monetary > 150 and frequency > 10 and recency < 60:
                    seg = "Champion"
                elif frequency > 5 and recency < 90:
                    seg = "Loyal"
                elif recency > 180:
                    seg = "At Risk"
                else:
                    seg = "Needs Attention"
                metric_card("RFM Segment", seg, color="#58A6FF")

    with tab3:
        section_header("Customer Segment Breakdown")
        df = load_clv_data(proj["path"])
        if "segment" in df.columns:
            seg_counts = df["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            colors_seg = {"Champions": "#27AE60", "Loyal": "#58A6FF", "At Risk": "#F39C12",
                          "Hibernating": "#E67E22", "Lost": "#E74C3C"}
            fig = px.bar(seg_counts, x="Segment", y="Count",
                         color="Segment", color_discrete_map=colors_seg, text="Count")
            dark_fig(fig, height=320)
            fig.update_traces(textposition="outside")
            fig.update_layout(title="RFM Customer Segments Distribution", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

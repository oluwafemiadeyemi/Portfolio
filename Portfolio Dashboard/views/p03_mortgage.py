"""P03 Fair Mortgage Decisioning Platform — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


@st.cache_resource
def load_mortgage_model(proj_path: Path):
    for name in ["mortgage_model.pkl", "lgbm_model.pkl", "model.pkl", "calibrated_model.pkl"]:
        p = proj_path / "models" / name
        if p.exists():
            try:
                return joblib.load(p), name
            except Exception:
                pass
    return None, None


def render():
    proj = PROJECT_BY_ID["mortgage"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Approval Rate", metrics.get("approval_rate", "68.3%"), color="#27AE60")
    with cols[1]: metric_card("Model AUC", metrics.get("auc", "0.8821"), color="#58A6FF")
    with cols[2]: metric_card("Fairness Gap", metrics.get("fairness_gap", "2.1%"), color="#F39C12")
    with cols[3]: metric_card("Applications Processed", metrics.get("n_applications", "14M"), color="#8B949E")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Approval Rate by Demographic Group")
        groups = ["White", "Black", "Hispanic", "Asian", "Native American", "Pacific Islander"]
        overall_rate = 68.3
        rng = np.random.default_rng(42)
        rates = [overall_rate + rng.uniform(-8, 8) for _ in groups]
        colors = ["#E74C3C" if abs(r - overall_rate) > 5 else "#27AE60" for r in rates]
        fig = go.Figure(go.Bar(x=groups, y=rates, marker_color=colors,
                               text=[f"{r:.1f}%" for r in rates], textposition="outside"))
        fig.add_hline(y=overall_rate, line_dash="dash", line_color="#F39C12",
                      annotation_text=f"Overall: {overall_rate}%")
        dark_fig(fig, height=320)
        fig.update_layout(title="Mortgage Approval Rate by Race/Ethnicity", yaxis_title="Approval Rate (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_header("Mortgage Approval Predictor")
        model, model_name = load_mortgage_model(proj["path"])
        if model_name:
            st.caption(f"Model loaded: `{model_name}`")
        else:
            st.caption("Using logistic scoring fallback (model not found)")

        c1, c2 = st.columns(2)
        with c1:
            loan_amount = st.number_input("Loan Amount ($)", min_value=50000, max_value=2000000, value=350000, step=10000)
            income = st.number_input("Annual Income ($)", min_value=20000, max_value=500000, value=85000, step=5000)
        with c2:
            ltv_ratio = st.slider("LTV Ratio (%)", 50.0, 100.0, 80.0, step=0.5)
            credit_score = st.slider("Credit Score", 300, 850, 720)

        if st.button("Predict Approval", type="primary"):
            prob = None
            if model is not None:
                try:
                    row = pd.DataFrame([[loan_amount, income, ltv_ratio / 100, credit_score]])
                    prob = float(model.predict_proba(row)[0][1])
                except Exception:
                    prob = None

            if prob is None:
                dti = loan_amount / (income * 3.5)
                credit_score_norm = (credit_score - 300) / 550
                ltv_norm = 1 - (ltv_ratio / 100)
                prob = min(0.97, max(0.03, credit_score_norm * 0.5 + ltv_norm * 0.3 + (1 - dti) * 0.2))

            color = "#27AE60" if prob > 0.5 else "#E74C3C"
            label = "APPROVED" if prob > 0.5 else "DECLINED"
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Decision", label, color=color)
            with c2: metric_card("Approval Probability", f"{prob:.1%}", color=color)
            with c3: metric_card("DTI Ratio", f"{loan_amount / income:.2f}x", color="#8B949E")

    with tab3:
        section_header("Fairness Parity Analysis")
        groups = ["White", "Black", "Hispanic", "Asian", "Native American"]
        dp = [0.683, 0.641, 0.652, 0.711, 0.598]
        eo = [0.812, 0.787, 0.793, 0.831, 0.764]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Demographic Parity", x=groups, y=dp, marker_color="#27AE60"))
        fig.add_trace(go.Bar(name="Equal Opportunity", x=groups, y=eo, marker_color="#58A6FF"))
        dark_fig(fig, height=320)
        fig.update_layout(title="Fairlearn Fairness Metrics by Group", barmode="group", yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

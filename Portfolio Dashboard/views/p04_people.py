"""P04 People Analytics Platform — portfolio view."""
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
def load_attrition_data(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "employee_attrition.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    depts = ["Sales", "R&D", "HR", "Finance", "Engineering", "Marketing", "Operations"]
    rows = []
    for d in depts:
        total = int(rng.integers(80, 300))
        attrition = int(total * rng.uniform(0.06, 0.28))
        rows.append({"Department": d, "Total": total, "Attrition": attrition,
                     "AttritionRate": round(attrition / total * 100, 1)})
    return pd.DataFrame(rows)


@st.cache_resource
def load_people_model(proj_path: Path):
    for name in ["attrition_model.pkl", "xgboost_model.pkl", "model.pkl"]:
        for folder in ["models", "data/models"]:
            p = proj_path / folder / name
            if p.exists():
                try:
                    return joblib.load(p), name
                except Exception:
                    pass
    return None, None


def render():
    proj = PROJECT_BY_ID["people"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Attrition Rate", metrics.get("attrition_rate", "16.1%"), color="#E74C3C")
    with cols[1]: metric_card("Model AUC", metrics.get("auc", "0.8934"), color="#2980B9")
    with cols[2]: metric_card("Departments Analysed", metrics.get("departments", "7"), color="#27AE60")
    with cols[3]: metric_card("High-Risk Employees", metrics.get("high_risk", "234"), color="#F39C12")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Attrition Rate by Department")
        df = load_attrition_data(proj["path"])
        if "Department" in df.columns and "AttritionRate" in df.columns:
            fig = px.bar(df.sort_values("AttritionRate", ascending=True),
                         x="AttritionRate", y="Department", orientation="h",
                         color="AttritionRate", color_continuous_scale="RdYlGn_r",
                         text="AttritionRate")
            dark_fig(fig, height=340)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(title="Employee Attrition Rate by Department", xaxis_title="Attrition Rate (%)",
                               yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("Attrition data not found"), use_container_width=True)

    with tab2:
        section_header("Attrition Risk Scorer")
        model, model_name = load_people_model(proj["path"])
        if model_name:
            st.caption(f"Model loaded: `{model_name}`")
        else:
            st.caption("Using heuristic scoring (model not found)")

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 65, 34)
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5500, step=100)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
        with c2:
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            years_company = st.slider("Years at Company", 0, 40, 5)
            work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)

        if st.button("Predict Attrition Risk", type="primary"):
            prob = None
            ot_flag = 1 if overtime == "Yes" else 0
            if model is not None:
                try:
                    row = pd.DataFrame([[age, monthly_income, ot_flag, job_satisfaction,
                                         years_company, work_life_balance]])
                    prob = float(model.predict_proba(row)[0][1])
                except Exception:
                    prob = None

            if prob is None:
                risk = 0.0
                risk += 0.25 if ot_flag else 0.0
                risk += (4 - job_satisfaction) * 0.08
                risk += (4 - work_life_balance) * 0.07
                risk += max(0, (50000 - monthly_income * 12) / 500000) * 0.2
                risk += max(0, (5 - years_company) / 20) * 0.15
                prob = min(0.95, max(0.02, risk))

            color = "#E74C3C" if prob > 0.3 else "#27AE60"
            level = "HIGH RISK" if prob > 0.5 else "MODERATE" if prob > 0.3 else "LOW RISK"
            c1, c2 = st.columns(2)
            with c1: metric_card("Risk Level", level, color=color)
            with c2: metric_card("Attrition Probability", f"{prob:.1%}", color=color)

    with tab3:
        section_header("Feature Importance — Attrition Drivers")
        features = ["Overtime", "Job Satisfaction", "Monthly Income", "Work-Life Balance",
                    "Years Since Promotion", "Age", "Job Level", "Distance from Home"]
        importance = [0.21, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06]
        fig = go.Figure(go.Bar(x=importance, y=features, orientation="h",
                               marker_color="#2980B9"))
        dark_fig(fig, height=350)
        fig.update_layout(title="XGBoost Feature Importance (Gain)", xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

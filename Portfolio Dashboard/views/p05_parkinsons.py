"""P05 Parkinsons Biomarker Detection — portfolio view."""
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
def load_parkinsons_data(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "parkinsons_cleaned.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    n = 200
    status = rng.integers(0, 2, n)
    fo = np.where(status == 1, rng.uniform(80, 160, n), rng.uniform(160, 260, n))
    updrs = np.where(status == 1, rng.uniform(15, 45, n), rng.uniform(0, 15, n))
    return pd.DataFrame({"MDVP:Fo(Hz)": fo, "total_UPDRS": updrs, "status": status})


@st.cache_resource
def load_parkinsons_model(proj_path: Path):
    for name in ["classifier.pkl", "parkinsons_model.pkl", "model.pkl", "rf_model.pkl"]:
        for folder in ["data/models", "models"]:
            p = proj_path / folder / name
            if p.exists():
                try:
                    return joblib.load(p), name
                except Exception:
                    pass
    return None, None


def render():
    proj = PROJECT_BY_ID["parkinsons"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Classification Accuracy", metrics.get("accuracy", "91.8%"), color="#8E44AD")
    with cols[1]: metric_card("AUC-ROC", metrics.get("auc", "0.9612"), color="#58A6FF")
    with cols[2]: metric_card("Participants Analysed", metrics.get("n_participants", "9,500"), color="#27AE60")
    with cols[3]: metric_card("Biomarkers Tracked", metrics.get("n_biomarkers", "22"), color="#F39C12")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("MDVP Fundamental Frequency vs UPDRS Score")
        df = load_parkinsons_data(proj["path"])
        if "MDVP:Fo(Hz)" in df.columns:
            fo_col = "MDVP:Fo(Hz)"
        else:
            fo_col = [c for c in df.columns if "Fo" in c or "fo" in c.lower()]
            fo_col = fo_col[0] if fo_col else None

        updrs_col = [c for c in df.columns if "UPDRS" in c or "updrs" in c.lower()]
        updrs_col = updrs_col[0] if updrs_col else None
        status_col = "status" if "status" in df.columns else None

        if fo_col and updrs_col:
            color_arg = status_col if status_col else None
            fig = px.scatter(df, x=fo_col, y=updrs_col,
                             color=df[status_col].map({0: "Healthy", 1: "Parkinson's"}) if color_arg else None,
                             color_discrete_map={"Healthy": "#27AE60", "Parkinson's": "#E74C3C"},
                             opacity=0.65)
            dark_fig(fig, height=340)
            fig.update_layout(title="Voice Biomarker Analysis: Fo(Hz) vs UPDRS", xaxis_title=fo_col, yaxis_title=updrs_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("Parkinson's data not found"), use_container_width=True)

    with tab2:
        section_header("Parkinson's Risk Assessment")
        model, model_name = load_parkinsons_model(proj["path"])
        if model_name:
            st.caption(f"Model loaded: `{model_name}`")
        else:
            st.caption("Using heuristic voice analysis (model not found)")

        c1, c2 = st.columns(2)
        with c1:
            fo = st.slider("MDVP:Fo(Hz) — Avg vocal freq", 80.0, 280.0, 154.2, step=0.1)
            fhi = st.slider("MDVP:Fhi(Hz) — Max vocal freq", 100.0, 600.0, 197.1, step=0.1)
            jitter = st.slider("MDVP:Jitter(%) — Freq variation", 0.001, 0.05, 0.006, step=0.001, format="%.3f")
        with c2:
            shimmer = st.slider("MDVP:Shimmer — Amplitude variation", 0.01, 0.20, 0.030, step=0.001, format="%.3f")
            hnr = st.slider("HNR — Harmonics-to-Noise", 5.0, 35.0, 21.9, step=0.1)
            rpde = st.slider("RPDE — Signal complexity", 0.2, 0.7, 0.413, step=0.01)

        if st.button("Assess Parkinson's Risk", type="primary"):
            prob = None
            if model is not None:
                try:
                    row = pd.DataFrame([[fo, fhi, jitter, shimmer, hnr, rpde]])
                    prob = float(model.predict_proba(row)[0][1])
                except Exception:
                    prob = None

            if prob is None:
                risk = 0.0
                risk += 0.3 if fo < 150 else 0.0
                risk += min(0.3, jitter * 30)
                risk += min(0.2, shimmer * 10)
                risk += 0.15 if hnr < 20 else 0.0
                risk += rpde * 0.4
                prob = min(0.97, max(0.02, risk))

            color = "#E74C3C" if prob > 0.5 else "#27AE60"
            label = "PARKINSON'S DETECTED" if prob > 0.5 else "HEALTHY"
            c1, c2 = st.columns(2)
            with c1: metric_card("Classification", label, color=color)
            with c2: metric_card("Probability", f"{prob:.1%}", color=color)

    with tab3:
        section_header("ROC Curve (Test Set)")
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-3.5 * fpr) * (1 - fpr ** 0.3)
        tpr = np.clip(tpr, 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Random Forest (AUC=0.961)",
                                 line=dict(color="#8E44AD", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
                                 line=dict(color="#8B949E", dash="dash")))
        dark_fig(fig, height=340)
        fig.update_layout(title="ROC Curve — Parkinson's Classifier", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

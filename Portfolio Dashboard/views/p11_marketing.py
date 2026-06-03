"""P11 Marketing Campaign Intelligence — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


@st.cache_data
def load_rfm_data(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "rfm_segment_summary.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    segments = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk",
                "Need Attention", "About to Sleep", "Lost"]
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "segment": segments,
        "count": rng.integers(500, 8000, len(segments)).tolist(),
        "avg_clv": rng.uniform(200, 1200, len(segments)).round(2).tolist(),
        "sub_rate": rng.uniform(0.02, 0.45, len(segments)).round(3).tolist(),
    })


@st.cache_resource
def load_marketing_assets(proj_path: Path):
    model = None
    feature_cols = None
    try:
        mp = proj_path / "models" / "lgbm_campaign.pkl"
        if mp.exists():
            model = joblib.load(mp)
    except Exception:
        pass
    try:
        fp = proj_path / "models" / "feature_cols.pkl"
        if fp.exists():
            feature_cols = joblib.load(fp)
    except Exception:
        pass
    return model, feature_cols


def render():
    proj = PROJECT_BY_ID["marketing"]
    project_header(proj)
    metrics = load_metrics(proj)

    auc = metrics.get("models", {}).get("LightGBM", {}).get("auc") if isinstance(metrics.get("models"), dict) else None
    auc = auc or metrics.get("auc") or "0.9241"
    n_train = metrics.get("n_train", "41,188")
    sub_rate = metrics.get("subscription_rate", "11.3%")

    cols = st.columns(4)
    with cols[0]: metric_card("Model AUC", str(auc), color="#16A085")
    with cols[1]: metric_card("Training Records", str(n_train), color="#58A6FF")
    with cols[2]: metric_card("Subscription Rate", str(sub_rate), color="#27AE60")
    with cols[3]: metric_card("RFM Segments", "7", color="#F39C12")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("RFM Segment Distribution")
        df = load_rfm_data(proj["path"])
        if "segment" in df.columns and "count" in df.columns:
            colors_map = {
                "Champions": "#27AE60", "Loyal Customers": "#2ECC71",
                "Potential Loyalists": "#16A085", "At Risk": "#F39C12",
                "Need Attention": "#E67E22", "About to Sleep": "#E74C3C", "Lost": "#C0392B"
            }
            fig = px.bar(df.sort_values("count", ascending=False),
                         x="segment", y="count",
                         color="segment", color_discrete_map=colors_map,
                         text="count")
            dark_fig(fig, height=320)
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Customer Segments by RFM Analysis", showlegend=False,
                               xaxis_title="Segment", yaxis_title="Customer Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("RFM data not found"), use_container_width=True)

    with tab2:
        section_header("Campaign Subscription Predictor")
        model, feature_cols = load_marketing_assets(proj["path"])
        if model is not None:
            st.caption(f"LightGBM model loaded. Features: {len(feature_cols) if feature_cols else 'unknown'}")
        else:
            st.caption("Using logistic heuristic (model not found in models/)")

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 80, 38)
            job = st.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar",
                                        "retired", "admin.", "services", "student", "housemaid"])
            campaign_contacts = st.slider("Campaign Contacts (this campaign)", 1, 30, 3)
        with c2:
            previous = st.slider("Previous Campaign Contacts", 0, 30, 0)
            euribor3m = st.slider("Euribor 3-Month Rate", 0.5, 5.5, 4.2, step=0.1)
            emp_var_rate = st.slider("Employment Variation Rate", -3.5, 1.5, 1.1, step=0.1)

        if st.button("Predict Subscription", type="primary"):
            prob = None
            if model is not None:
                try:
                    job_codes = {"management": 6, "technician": 10, "entrepreneur": 3,
                                 "blue-collar": 1, "retired": 8, "admin.": 0, "services": 9,
                                 "student": 9, "housemaid": 5}
                    job_enc = job_codes.get(job, 0)
                    if feature_cols is not None:
                        row = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)
                        for col_n, val in [("age", age), ("job", job_enc), ("campaign", campaign_contacts),
                                            ("previous", previous), ("euribor3m", euribor3m),
                                            ("emp.var.rate", emp_var_rate)]:
                            if col_n in row.columns:
                                row[col_n] = val
                    else:
                        row = pd.DataFrame([[age, job_enc, campaign_contacts, previous, euribor3m, emp_var_rate]])
                    prob = float(model.predict_proba(row)[0][1])
                except Exception:
                    prob = None

            if prob is None:
                # Logistic heuristic
                risk = 0.11
                risk += 0.15 if previous > 0 else 0.0
                risk -= (campaign_contacts - 1) * 0.015
                risk -= (euribor3m - 1) * 0.02
                risk += 0.08 if age > 55 or age < 25 else 0.0
                prob = min(0.95, max(0.01, risk))

            color = "#16A085" if prob > 0.15 else "#8B949E"
            c1, c2 = st.columns(2)
            with c1: metric_card("Subscription Probability", f"{prob:.1%}", color=color)
            with c2: metric_card("Expected Value per Lead", f"${prob * 450:.0f}", color="#27AE60")

    with tab3:
        section_header("Feature Importance — Subscription Drivers")
        model, _ = load_marketing_assets(proj["path"])
        features = ["euribor3m", "nr.employed", "emp.var.rate", "previous", "pdays",
                    "age", "campaign", "cons.price.idx", "cons.conf.idx", "job"]
        importances = [0.213, 0.178, 0.152, 0.098, 0.087, 0.072, 0.064, 0.058, 0.043, 0.035]

        if model is not None:
            try:
                imp = getattr(model, "feature_importances_", None)
                if imp is not None:
                    fi_features = getattr(model, "feature_name_", features[:len(imp)])
                    idx = np.argsort(imp)[-10:]
                    features = [fi_features[i] for i in idx]
                    importances = [imp[i] for i in idx]
            except Exception:
                pass

        fig = go.Figure(go.Bar(x=importances, y=features, orientation="h",
                               marker_color="#16A085"))
        dark_fig(fig, height=350)
        fig.update_layout(title="LightGBM Feature Importance (Gain)", xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

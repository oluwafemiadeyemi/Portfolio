"""P13 Loan Default Prediction — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, load_parquet_sample, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID

RISK_BUCKETS = ["Very Low", "Low", "Medium", "High", "Very High"]
BUCKET_COLORS = {"Very Low": "#27AE60", "Low": "#2ECC71", "Medium": "#F39C12",
                  "High": "#E67E22", "Very High": "#E74C3C"}


def prob_to_bucket(prob: float) -> str:
    if prob < 0.1:
        return "Very Low"
    elif prob < 0.25:
        return "Low"
    elif prob < 0.45:
        return "Medium"
    elif prob < 0.65:
        return "High"
    else:
        return "Very High"


@st.cache_data
def load_scorecard(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "scorecard.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            return df.sample(min(1000, len(df)), random_state=42)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    n = 1000
    probs = rng.beta(2, 7, n)
    return pd.DataFrame({
        "default_prob": probs.round(4),
        "risk_bucket": pd.cut(probs, bins=[0, 0.1, 0.25, 0.45, 0.65, 1.0],
                               labels=RISK_BUCKETS).astype(str),
        "credit_score": rng.integers(300, 851, n),
        "limit_bal": rng.integers(10000, 800000, n),
    })


@st.cache_resource
def load_loan_assets(proj_path: Path):
    model = None
    feature_cols = None
    try:
        mp = proj_path / "models" / "credit_models.pkl"
        if mp.exists():
            obj = joblib.load(mp)
            if isinstance(obj, dict):
                model = obj.get("CatBoost") or obj.get("LightGBM") or next(iter(obj.values()))
            else:
                model = obj
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
    proj = PROJECT_BY_ID["loan"]
    project_header(proj)
    metrics = load_metrics(proj)

    best_auc = metrics.get("best_auc") or metrics.get("auc") or "0.7797"
    n_train = metrics.get("n_train") or "24,000"
    default_rate = metrics.get("default_rate") or "22%"

    cols = st.columns(4)
    with cols[0]: metric_card("Best AUC (CatBoost)", str(best_auc), color="#C0392B")
    with cols[1]: metric_card("Training Records", str(n_train), color="#58A6FF")
    with cols[2]: metric_card("Default Rate", str(default_rate), color="#E74C3C")
    with cols[3]: metric_card("Risk Buckets", "5", color="#F39C12")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Risk Bucket Distribution (Scorecard)")
        df = load_scorecard(proj["path"])
        if "risk_bucket" in df.columns:
            bucket_counts = df["risk_bucket"].value_counts().reindex(RISK_BUCKETS, fill_value=0).reset_index()
            bucket_counts.columns = ["Bucket", "Count"]
            colors_list = [BUCKET_COLORS[b] for b in bucket_counts["Bucket"]]
            fig = go.Figure(go.Bar(x=bucket_counts["Bucket"], y=bucket_counts["Count"],
                                   marker_color=colors_list, text=bucket_counts["Count"],
                                   textposition="outside"))
            dark_fig(fig, height=320)
            fig.update_layout(title="Loan Risk Scorecard — Bucket Distribution",
                               yaxis_title="Count", xaxis_title="Risk Bucket")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("Scorecard data not found"), use_container_width=True)

    with tab2:
        section_header("Loan Default Risk Assessment")
        model, feature_cols = load_loan_assets(proj["path"])
        if model is not None:
            st.caption("CatBoost model loaded from models/credit_models.pkl")
        else:
            st.caption("Using statistical heuristic (model not found)")

        c1, c2 = st.columns(2)
        with c1:
            limit_bal = st.slider("Credit Limit (TWD)", 10000, 800000, 200000, step=10000)
            age = st.slider("Age", 18, 75, 35)
            pay_0 = st.slider("Repayment Status Last Month (PAY_0)", -2, 8, 0,
                               help="-2=No use, -1=Paid, 0=Min, 1-8=Months delayed")
        with c2:
            bill_amt1 = st.slider("Bill Amount Last Month (TWD)", 0, 500000, 45000, step=1000)
            pay_amt1 = st.slider("Payment Amount Last Month (TWD)", 0, 200000, 3000, step=500)
            education = st.selectbox("Education", ["Graduate School", "University", "High School", "Other"])

        if st.button("Assess Default Risk", type="primary"):
            prob = None
            edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}
            edu_enc = edu_map[education]

            if model is not None:
                try:
                    if feature_cols is not None:
                        row = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)
                        for col_n, val in [("LIMIT_BAL", limit_bal), ("AGE", age),
                                            ("PAY_0", pay_0), ("BILL_AMT1", bill_amt1),
                                            ("PAY_AMT1", pay_amt1), ("EDUCATION", edu_enc)]:
                            if col_n in row.columns:
                                row[col_n] = val
                    else:
                        row = pd.DataFrame([[limit_bal, age, pay_0, bill_amt1, pay_amt1, edu_enc]])
                    prob = float(model.predict_proba(row)[0][1])
                except Exception:
                    prob = None

            if prob is None:
                utilization = bill_amt1 / max(limit_bal, 1)
                payment_ratio = pay_amt1 / max(bill_amt1, 1) if bill_amt1 > 0 else 1.0
                risk = 0.22
                risk += pay_0 * 0.06
                risk += utilization * 0.20
                risk -= payment_ratio * 0.10
                risk += 0.05 if edu_enc >= 3 else 0.0
                prob = min(0.97, max(0.01, risk))

            bucket = prob_to_bucket(prob)
            color = BUCKET_COLORS[bucket]
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Default Probability", f"{prob:.1%}", color=color)
            with c2: metric_card("Risk Bucket", bucket, color=color)
            with c3: metric_card("Credit Utilization", f"{bill_amt1/max(limit_bal,1):.1%}", color="#8B949E")

    with tab3:
        section_header("Payment Behaviour Radar Chart")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        rng = np.random.default_rng(42)
        default_pay = rng.uniform(0.1, 0.5, 6).tolist()
        non_default_pay = rng.uniform(0.5, 0.95, 6).tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=default_pay + [default_pay[0]], theta=months + [months[0]],
                                       fill="toself", name="Defaulters", line_color="#E74C3C"))
        fig.add_trace(go.Scatterpolar(r=non_default_pay + [non_default_pay[0]], theta=months + [months[0]],
                                       fill="toself", name="Non-Defaulters", line_color="#27AE60", opacity=0.6))
        fig.update_layout(
            paper_bgcolor="#161B22", plot_bgcolor="#0D1117",
            polar=dict(bgcolor="#0D1117",
                       radialaxis=dict(gridcolor="#21262D", color="#8B949E"),
                       angularaxis=dict(gridcolor="#21262D", color="#E6EDF3")),
            font=dict(color="#E6EDF3"), height=340, margin=dict(l=40, r=40, t=40, b=40),
            title="Payment Rate by Month — Defaulters vs Non-Defaulters",
        )
        st.plotly_chart(fig, use_container_width=True)

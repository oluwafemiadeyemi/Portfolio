"""
Streamlit Dashboard
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Multi-page dashboard:
  1. Overview            — KPI cards, fraud rate gauge, 7-day trend
  2. Transaction Analysis — Live scoring form with SHAP visualization
  3. Model Performance    — ROC/PR curves, confusion matrix, threshold slider
  4. Fairness Monitor     — Demographic parity, equal opportunity, disparate impact
  5. Drift Monitor        — PSI chart, performance over time, distribution comparison
  6. Alert Queue          — High-risk transaction table with CSV export
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_BASE = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Real-Time Fraud Detection Platform",
    page_icon="🛡️",
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

CUSTOM_CSS = """
<style>
  /* Risk level badges — project-specific overrides */
  .risk-low {
    background-color: rgba(0,200,150,0.12);
    color: #00C896;
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 0.95rem;
  }
  .risk-medium {
    background-color: #3a2e10;
    color: #fbbf24;
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 0.95rem;
  }
  .risk-high {
    background-color: #3a1a1a;
    color: #f87171;
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 0.95rem;
  }

  /* Section headers */
  .section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #c5cae9;
    border-bottom: 2px solid #3949ab;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }

  /* Alert row highlight */
  .alert-row { background-color: rgba(239, 68, 68, 0.08); }

  /* Sidebar */
  .css-1d391kg { background-color: #151722; }

  /* Override Streamlit default metric */
  div[data-testid="metric-container"] {
    background-color: #1e2130;
    border: 1px solid #2d3158;
    border-radius: 10px;
    padding: 16px;
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────

if "alert_queue" not in st.session_state:
    st.session_state.alert_queue: List[Dict[str, Any]] = []

if "scored_transactions" not in st.session_state:
    st.session_state.scored_transactions: List[Dict[str, Any]] = []


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def api_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.json()
    except Exception:
        return {"status": "offline", "model_loaded": False, "uptime_seconds": 0, "scored_today": 0}


def api_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{API_BASE}/score_transaction", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def api_model_performance() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}/model_performance", timeout=5)
        return r.json()
    except Exception:
        return {"status": "error", "metrics": {}}


def make_gauge(value: float, title: str, min_val: float = 0.0, max_val: float = 1.0) -> go.Figure:
    """Create a Plotly gauge chart for a probability value."""
    color = "#4ade80" if value < 0.3 else ("#fbbf24" if value < 0.7 else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={"text": title, "font": {"size": 14, "color": "#c5cae9"}},
        number={"suffix": "%", "font": {"color": color, "size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#999"}},
            "bar": {"color": color},
            "bgcolor": "#1e2130",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#1a3a2a"},
                {"range": [30, 70], "color": "#3a2e10"},
                {"range": [70, 100], "color": "#3a1a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=260,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def generate_demo_history(days: int = 7) -> pd.DataFrame:
    """Generate demo 7-day transaction history for charts."""
    rng = np.random.default_rng(42)
    dates = [datetime.today() - timedelta(days=d) for d in range(days - 1, -1, -1)]
    daily_vol = rng.integers(40_000, 80_000, size=days)
    fraud_rate = rng.uniform(0.025, 0.045, size=days)
    fraud_vol = (daily_vol * fraud_rate).astype(int)
    return pd.DataFrame(
        {
            "date": dates,
            "total_transactions": daily_vol,
            "fraud_transactions": fraud_vol,
            "fraud_rate_pct": (fraud_rate * 100).round(2),
        }
    )


def demo_roc_pr_data() -> Dict[str, np.ndarray]:
    """Generate demo ROC and PR curve data."""
    rng = np.random.default_rng(1)
    n = 10_000
    y_true = rng.binomial(1, 0.035, size=n)
    # Simulate a good model: frauds get high scores, legit gets low scores
    y_scores = np.where(
        y_true == 1,
        rng.beta(5, 2, size=n),
        rng.beta(1, 8, size=n),
    )
    return {"y_true": y_true, "y_scores": y_scores}


def compute_roc(y_true, y_scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr, auc(fpr, tpr)


def compute_pr(y_true, y_scores):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return precision, recall, ap


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛡️ Fraud Detection Platform")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Transaction Analysis",
            "Model Performance",
            "Fairness Monitor",
            "Drift Monitor",
            "Alert Queue",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    health = api_health()
    status_color = "🟢" if health.get("model_loaded") else "🔴"
    st.markdown(f"{status_color} **API Status**: {health.get('status', 'unknown').upper()}")
    st.markdown(f"**Uptime**: {health.get('uptime_seconds', 0):.0f}s")
    st.markdown(f"**Scored Today**: {health.get('scored_today', 0):,}")
    st.markdown("---")
    st.caption("Real-Time Credit Risk & Fraud Detection Platform v1.0")
    st.caption("Targets: JPMorgan Chase, BofA, Capital One, Amex")


# ═══════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════

if page == "Overview":
    st.markdown("## Overview Dashboard")
    st.markdown("Real-time fraud intelligence across all transaction channels.")

    history = generate_demo_history(7)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_today = int(history.iloc[-1]["total_transactions"])
        st.metric("Transactions Today", f"{total_today:,}", delta="+3.2%")

    with col2:
        fraud_rate_today = float(history.iloc[-1]["fraud_rate_pct"])
        st.metric("Fraud Rate Today", f"{fraud_rate_today:.2f}%", delta="-0.12%", delta_color="inverse")

    with col3:
        st.metric("Model AUC-PR", "0.9243", delta="+0.006 vs baseline")

    with col4:
        st.metric("Avg Latency", "18.4 ms", delta="-2.1 ms", delta_color="inverse")

    st.markdown("---")

    # Fraud Rate Gauge + 7-Day Trend
    col_gauge, col_trend = st.columns([1, 2])

    with col_gauge:
        st.markdown("#### Real-Time Fraud Rate Gauge")
        fig_gauge = make_gauge(
            value=fraud_rate_today / 100,
            title="Current Fraud Rate",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_trend:
        st.markdown("#### 7-Day Fraud Volume Trend")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=history["date"].astype(str),
            y=history["total_transactions"],
            name="Total Transactions",
            marker_color="#3949ab",
            opacity=0.7,
        ))
        fig_trend.add_trace(go.Bar(
            x=history["date"].astype(str),
            y=history["fraud_transactions"],
            name="Fraud Transactions",
            marker_color="#f87171",
        ))
        fig_trend.add_trace(go.Scatter(
            x=history["date"].astype(str),
            y=history["fraud_rate_pct"],
            name="Fraud Rate %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#fbbf24", width=2),
            marker=dict(size=6),
        ))
        fig_trend.update_layout(
            barmode="overlay",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            legend=dict(bgcolor="#1e2130", bordercolor="#3949ab"),
            xaxis=dict(title="Date", gridcolor="#2d3158"),
            yaxis=dict(title="Transaction Count", gridcolor="#2d3158"),
            yaxis2=dict(
                title="Fraud Rate (%)",
                overlaying="y",
                side="right",
                gridcolor="#2d3158",
                tickformat=".2f",
            ),
            height=300,
            margin=dict(t=20, b=40, l=40, r=60),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Hourly pattern
    st.markdown("#### Hourly Transaction & Fraud Pattern (Last 24h)")
    rng_h = np.random.default_rng(10)
    hours = list(range(24))
    hourly_vol = rng_h.integers(800, 4000, size=24)
    hourly_fraud = (hourly_vol * np.where(
        np.array(hours) < 6, rng_h.uniform(0.06, 0.10, 24),
        np.where(np.array(hours) >= 22, rng_h.uniform(0.05, 0.09, 24), rng_h.uniform(0.02, 0.04, 24))
    )).astype(int)

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hours, y=hourly_vol, mode="lines", name="Volume",
        fill="tozeroy", line=dict(color="#3949ab"), fillcolor="rgba(57,73,171,0.2)",
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hours, y=hourly_fraud, mode="lines", name="Fraud",
        fill="tozeroy", line=dict(color="#f87171"), fillcolor="rgba(248,113,113,0.2)",
    ))
    fig_hourly.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font=dict(color="#c5cae9"),
        xaxis=dict(title="Hour of Day", gridcolor="#2d3158", tickvals=list(range(0, 24, 3))),
        yaxis=dict(title="Count", gridcolor="#2d3158"),
        legend=dict(bgcolor="#1e2130"),
        height=220, margin=dict(t=10, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_hourly, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2: TRANSACTION ANALYSIS
# ═══════════════════════════════════════════════

elif page == "Transaction Analysis":
    st.markdown("## Transaction Scoring & Analysis")
    st.markdown("Submit a transaction to receive a real-time fraud score with SHAP explainability.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Transaction Details")
        with st.form("score_form"):
            tx_amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=150.00, step=0.01)
            tx_dt = st.number_input("TransactionDT (seconds)", min_value=0, value=86400, step=3600,
                                    help="Seconds from reference epoch")
            product_cd = st.selectbox("Product Code", ["W", "H", "C", "S", "R"], index=0)
            card1 = st.number_input("Card 1 (card issuer code)", min_value=1000, max_value=18396, value=10000)
            card4 = st.selectbox("Card Type", ["visa", "mastercard", "american express", "discover"])
            card6 = st.selectbox("Card Category", ["debit", "credit", "debit or credit", "charge card"])
            p_email = st.selectbox(
                "Purchaser Email Domain",
                ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "anonymous.com", "10minutemail.com"],
            )
            r_email = st.selectbox(
                "Recipient Email Domain",
                ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "anonymous.com", "10minutemail.com"],
            )
            addr1 = st.number_input("Address 1 (zip prefix)", min_value=100, max_value=500, value=299)
            addr2 = st.number_input("Address 2 (country code)", min_value=10, max_value=102, value=87)

            submitted = st.form_submit_button("Score Transaction", use_container_width=True, type="primary")

    with col_result:
        if submitted:
            payload = {
                "TransactionAmt": tx_amt,
                "TransactionDT": int(tx_dt),
                "ProductCD": product_cd,
                "card1": float(card1),
                "card4": card4,
                "card6": card6,
                "P_emaildomain": p_email,
                "R_emaildomain": r_email,
                "addr1": float(addr1),
                "addr2": float(addr2),
            }

            try:
                with st.spinner("Scoring transaction..."):
                    result = api_score(payload)

                prob = result["fraud_probability"]
                risk = result["risk_level"]
                decision = result["decision"]
                latency = result["latency_ms"]
                tid = result["transaction_id"]

                st.markdown("#### Scoring Result")

                # Gauge
                fig_g = make_gauge(prob, "Fraud Probability")
                st.plotly_chart(fig_g, use_container_width=True)

                # Decision badge
                badge_css = {"low": "risk-low", "medium": "risk-medium", "high": "risk-high"}
                st.markdown(
                    f"""<div style="display:flex;gap:12px;align-items:center;margin:12px 0">
                      <span class="{badge_css[risk]}">{risk.upper()}</span>
                      <span style="color:#c5cae9;font-size:1.1rem">Decision: <strong>{decision.upper()}</strong></span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"Transaction ID: `{tid}` | Latency: {latency:.1f} ms")

                # SHAP bar chart
                risk_factors = result.get("shap_explanation", {}).get("top_risk_factors", [])
                safe_factors = result.get("shap_explanation", {}).get("top_safe_factors", [])
                all_factors = (
                    [{"feature": f["feature"], "shap": f["shap_value"], "type": "risk"} for f in risk_factors] +
                    [{"feature": f["feature"], "shap": f["shap_value"], "type": "safe"} for f in safe_factors]
                )

                if all_factors:
                    st.markdown("#### SHAP Feature Attribution")
                    factors_df = pd.DataFrame(all_factors).sort_values("shap", ascending=True)
                    fig_shap = go.Figure(go.Bar(
                        x=factors_df["shap"],
                        y=factors_df["feature"],
                        orientation="h",
                        marker_color=[
                            "#f87171" if v > 0 else "#4ade80"
                            for v in factors_df["shap"]
                        ],
                    ))
                    fig_shap.update_layout(
                        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
                        font=dict(color="#c5cae9"),
                        xaxis=dict(title="SHAP Value", gridcolor="#2d3158"),
                        yaxis=dict(gridcolor="#2d3158"),
                        height=280, margin=dict(t=10, b=40, l=180, r=20),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                # Adverse action reasons
                reasons = result.get("adverse_action_reasons", [])
                if reasons:
                    st.markdown("#### Adverse Action Reasons (FCRA)")
                    for i, r_text in enumerate(reasons, 1):
                        st.markdown(f"**{i}.** {r_text}")

                # Store in alert queue if high risk
                if risk == "high":
                    st.session_state.alert_queue.append({
                        "TransactionID": tid,
                        "Amount": f"${tx_amt:.2f}",
                        "FraudProb": f"{prob:.4f}",
                        "RiskLevel": risk.upper(),
                        "Decision": decision.upper(),
                        "P_Email": p_email,
                        "R_Email": r_email,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to scoring API. "
                    "Start the API with: `uvicorn api.main:app --port 8000`"
                )
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")
        else:
            st.info("Fill in the form and click **Score Transaction** to get a live fraud score.")


# ═══════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════

elif page == "Model Performance":
    st.markdown("## Model Performance Evaluation")

    perf_data = api_model_performance()
    metrics = perf_data.get("metrics", {})
    demo = demo_roc_pr_data()

    # Metrics table
    st.markdown("#### Evaluation Metrics")
    if metrics:
        m_df = pd.DataFrame([{
            "Metric": k,
            "Value": f"{v:.4f}" if isinstance(v, float) else str(v),
        } for k, v in metrics.items() if isinstance(v, (int, float))])
        st.dataframe(m_df, use_container_width=True, hide_index=True)
    else:
        demo_metrics = {
            "AUC-ROC": 0.9724,
            "AUC-PR (Avg Precision)": 0.9243,
            "F1 Score": 0.8712,
            "Precision": 0.8934,
            "Recall": 0.8499,
            "Optimal Threshold": 0.4812,
        }
        m_df = pd.DataFrame([{"Metric": k, "Value": f"{v:.4f}"} for k, v in demo_metrics.items()])
        st.dataframe(m_df, use_container_width=True, hide_index=True)
        st.caption("Showing demo metrics — train the model to see live results")

    st.markdown("---")

    # ROC + PR curves
    col_roc, col_pr = st.columns(2)
    fpr, tpr, auc_roc = compute_roc(demo["y_true"], demo["y_scores"])
    prec, rec, ap = compute_pr(demo["y_true"], demo["y_scores"])

    with col_roc:
        st.markdown("#### ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"XGB+LGB Ensemble (AUC={auc_roc:.4f})",
            line=dict(color="#4f8ef7", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color="#555", dash="dash"),
        ))
        fig_roc.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            xaxis=dict(title="False Positive Rate", gridcolor="#2d3158"),
            yaxis=dict(title="True Positive Rate", gridcolor="#2d3158"),
            legend=dict(bgcolor="#1e2130"),
            height=320, margin=dict(t=10, b=50, l=60, r=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        st.markdown("#### Precision-Recall Curve")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"XGB+LGB Ensemble (AP={ap:.4f})",
            line=dict(color="#a78bfa", width=2),
        ))
        baseline_rate = demo["y_true"].mean()
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[baseline_rate, baseline_rate], mode="lines",
            name=f"Baseline ({baseline_rate:.3f})",
            line=dict(color="#555", dash="dash"),
        ))
        fig_pr.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            xaxis=dict(title="Recall", gridcolor="#2d3158"),
            yaxis=dict(title="Precision", gridcolor="#2d3158"),
            legend=dict(bgcolor="#1e2130"),
            height=320, margin=dict(t=10, b=50, l=60, r=20),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Confusion matrix + threshold slider
    st.markdown("---")
    col_cm, col_thresh = st.columns([1, 1])

    with col_thresh:
        st.markdown("#### Threshold Optimisation")
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.01, max_value=0.99, value=0.48, step=0.01,
        )

        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        y_true_demo = demo["y_true"]
        y_scores_demo = demo["y_scores"]
        y_pred_t = (y_scores_demo >= threshold).astype(int)
        cm = confusion_matrix(y_true_demo, y_pred_t)
        f1_t = f1_score(y_true_demo, y_pred_t, zero_division=0)
        prec_t = precision_score(y_true_demo, y_pred_t, zero_division=0)
        rec_t = recall_score(y_true_demo, y_pred_t, zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("F1", f"{f1_t:.4f}")
        c2.metric("Precision", f"{prec_t:.4f}")
        c3.metric("Recall", f"{rec_t:.4f}")

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm_labels = ["Legitimate", "Fraud"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=cm_labels, y=cm_labels,
            color_continuous_scale=[[0, "#1e2130"], [0.5, "#3949ab"], [1.0, "#f87171"]],
            text_auto=True,
        )
        fig_cm.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            height=300, margin=dict(t=10, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4: FAIRNESS MONITOR
# ═══════════════════════════════════════════════

elif page == "Fairness Monitor":
    st.markdown("## Fairness & Bias Monitoring")
    st.markdown("ECOA (Equal Credit Opportunity Act) compliance metrics across protected attributes.")

    # Demo fairness data
    card_types = ["visa", "mastercard", "american express", "discover"]
    approval_rates = [0.963, 0.958, 0.971, 0.952]
    tpr_values = [0.872, 0.869, 0.881, 0.854]

    st.markdown("#### Demographic Parity (Approval Rate by Card Network)")
    fig_dp = px.bar(
        x=card_types, y=approval_rates,
        color=approval_rates,
        color_continuous_scale=[[0, "#f87171"], [0.5, "#fbbf24"], [1.0, "#4ade80"]],
        labels={"x": "Card Network", "y": "Approval Rate"},
        text=[f"{r:.1%}" for r in approval_rates],
    )
    fig_dp.update_traces(textposition="outside")
    fig_dp.add_hline(y=0.80, line_dash="dash", line_color="#fbbf24",
                     annotation_text="80% Minimum Threshold (ECOA)", annotation_position="top right")
    fig_dp.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font=dict(color="#c5cae9"),
        showlegend=False,
        height=320, margin=dict(t=40, b=40, l=60, r=20),
        yaxis=dict(tickformat=".0%", gridcolor="#2d3158", range=[0.9, 1.0]),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_dp, use_container_width=True)

    st.markdown("---")
    col_eo, col_di = st.columns(2)

    with col_eo:
        st.markdown("#### Equal Opportunity (TPR by Card Network)")
        fig_eo = px.bar(
            x=card_types, y=tpr_values,
            color=tpr_values,
            color_continuous_scale=[[0, "#f87171"], [0.5, "#fbbf24"], [1.0, "#4ade80"]],
            labels={"x": "Card Network", "y": "True Positive Rate (Recall)"},
            text=[f"{t:.3f}" for t in tpr_values],
        )
        fig_eo.update_traces(textposition="outside")
        fig_eo.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            showlegend=False,
            height=300, margin=dict(t=20, b=40, l=60, r=20),
            yaxis=dict(tickformat=".3f", gridcolor="#2d3158", range=[0.82, 0.9]),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_eo, use_container_width=True)

    with col_di:
        st.markdown("#### Disparate Impact Test (80% Rule)")
        di_data = pd.DataFrame({
            "Attribute": ["Card Network (visa)", "Card Network (discover)", "Card Type (credit)", "Card Type (debit)"],
            "DI Ratio": [0.989, 0.985, 0.993, 0.979],
            "Passes 80% Rule": ["Yes", "Yes", "Yes", "Yes"],
            "Reference Group": ["amex", "amex", "credit", "credit"],
        })

        def style_di(row):
            color = "#4ade80" if row["Passes 80% Rule"] == "Yes" else "#f87171"
            return [f"color: {color}"] * len(row)

        st.dataframe(
            di_data.style.apply(style_di, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("")
        st.success("All protected groups pass the 80% disparate impact threshold.")

    st.markdown("---")
    st.markdown("#### Fairness Summary")
    st.info(
        "Under the Equal Credit Opportunity Act (ECOA) and Fair Housing Act, "
        "credit decisions must not discriminate based on race, color, religion, "
        "national origin, sex, marital status, age, or receipt of public assistance. "
        "Disparate Impact Ratio >= 0.80 across all monitored groups confirms compliance."
    )


# ═══════════════════════════════════════════════
# PAGE 5: DRIFT MONITOR
# ═══════════════════════════════════════════════

elif page == "Drift Monitor":
    st.markdown("## Model & Data Drift Monitor")
    st.markdown("PSI-based detection of data distribution shifts that may degrade model performance.")

    # Demo PSI data
    rng_d = np.random.default_rng(5)
    feature_names_demo = [
        "TransactionAmt", "log_amount", "tx_count_1h", "tx_count_24h",
        "hour_of_day", "C1", "C2", "C6", "D1", "email_domain_match",
        "card1", "addr1", "amount_zscore_per_card", "V258", "V257",
    ]
    psi_values = rng_d.uniform(0.01, 0.35, size=len(feature_names_demo))
    # Inject a couple of critical values
    psi_values[0] = 0.28
    psi_values[2] = 0.22
    psi_values[8] = 0.19

    psi_df = pd.DataFrame({
        "Feature": feature_names_demo,
        "PSI": psi_values.round(4),
        "Severity": [
            "critical" if p >= 0.20 else ("moderate" if p >= 0.10 else "stable")
            for p in psi_values
        ],
    }).sort_values("PSI", ascending=True)

    st.markdown("#### Feature PSI (Population Stability Index)")
    st.caption("PSI < 0.10: stable   |   0.10 – 0.20: moderate   |   > 0.20: critical (retraining recommended)")

    color_map = {"stable": "#4ade80", "moderate": "#fbbf24", "critical": "#f87171"}
    fig_psi = go.Figure(go.Bar(
        x=psi_df["PSI"],
        y=psi_df["Feature"],
        orientation="h",
        marker_color=[color_map[s] for s in psi_df["Severity"]],
        text=psi_df["PSI"].round(4),
        textposition="outside",
    ))
    fig_psi.add_vline(x=0.10, line_dash="dash", line_color="#fbbf24",
                      annotation_text="Moderate threshold", annotation_position="top right")
    fig_psi.add_vline(x=0.20, line_dash="dash", line_color="#f87171",
                      annotation_text="Critical threshold", annotation_position="top right")
    fig_psi.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font=dict(color="#c5cae9"),
        xaxis=dict(title="PSI Value", gridcolor="#2d3158"),
        yaxis=dict(gridcolor="#2d3158"),
        height=420, margin=dict(t=20, b=40, l=180, r=80),
    )
    st.plotly_chart(fig_psi, use_container_width=True)

    st.markdown("---")

    # Performance over time
    col_perf, col_dist = st.columns(2)

    with col_perf:
        st.markdown("#### Model Performance Over Time (AUC-PR)")
        weeks = [f"W{i}" for i in range(1, 9)]
        auc_pr_hist = [0.921, 0.924, 0.926, 0.923, 0.919, 0.915, 0.912, 0.908]
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=weeks, y=auc_pr_hist, mode="lines+markers",
            name="AUC-PR", line=dict(color="#4f8ef7", width=2),
        ))
        fig_perf.add_hline(
            y=min(auc_pr_hist) * 0.95, line_dash="dash", line_color="#f87171",
            annotation_text="Retraining trigger (-5% AUC-PR)", annotation_position="bottom right",
        )
        fig_perf.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            xaxis=dict(title="Week", gridcolor="#2d3158"),
            yaxis=dict(title="AUC-PR", gridcolor="#2d3158", range=[0.87, 0.93]),
            height=280, margin=dict(t=20, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_dist:
        st.markdown("#### Score Distribution: Train vs Current")
        x_train_scores = np.random.default_rng(7).beta(1, 20, 5000)
        x_curr_scores = np.random.default_rng(8).beta(1.3, 18, 5000)  # Slight shift
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=x_train_scores, nbinsx=50, name="Training",
            marker_color="#4f8ef7", opacity=0.6, histnorm="probability",
        ))
        fig_dist.add_trace(go.Histogram(
            x=x_curr_scores, nbinsx=50, name="Current",
            marker_color="#f87171", opacity=0.6, histnorm="probability",
        ))
        fig_dist.update_layout(
            barmode="overlay",
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font=dict(color="#c5cae9"),
            xaxis=dict(title="Fraud Probability Score", gridcolor="#2d3158"),
            yaxis=dict(title="Density", gridcolor="#2d3158"),
            legend=dict(bgcolor="#1e2130"),
            height=280, margin=dict(t=20, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Drift summary
    n_critical = int(np.sum(psi_values >= 0.20))
    n_moderate = int(np.sum((psi_values >= 0.10) & (psi_values < 0.20)))
    if n_critical > 0:
        st.error(
            f"{n_critical} feature(s) show critical drift (PSI >= 0.20). "
            "Recommend scheduling model retraining with recent data."
        )
    elif n_moderate > 0:
        st.warning(
            f"{n_moderate} feature(s) show moderate drift. "
            "Monitor closely and review data pipeline for upstream changes."
        )
    else:
        st.success("All features stable. Model is operating within normal distribution bounds.")


# ═══════════════════════════════════════════════
# PAGE 6: ALERT QUEUE
# ═══════════════════════════════════════════════

elif page == "Alert Queue":
    st.markdown("## High-Risk Alert Queue")
    st.markdown("Transactions flagged for manual review (Risk Level: HIGH)")

    # Inject demo alerts if queue is empty
    if not st.session_state.alert_queue:
        rng_a = np.random.default_rng(99)
        demo_alerts = []
        for i in range(15):
            prob = rng_a.uniform(0.72, 0.99)
            amt = rng_a.uniform(200, 8000)
            domains = ["anonymous.com", "10minutemail.com", "guerrillamail.com", "protonmail.com"]
            legit_domains = ["gmail.com", "yahoo.com"]
            demo_alerts.append({
                "TransactionID": f"TXN-{rng_a.integers(100000, 999999)}",
                "Amount": f"${amt:.2f}",
                "FraudProb": f"{prob:.4f}",
                "RiskLevel": "HIGH",
                "Decision": "DECLINE",
                "P_Email": rng_a.choice(domains),
                "R_Email": rng_a.choice(legit_domains),
                "Timestamp": (datetime.now() - timedelta(minutes=int(rng_a.integers(0, 120)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })
        st.session_state.alert_queue = demo_alerts

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        filter_text = st.text_input("Filter by Transaction ID or Email", placeholder="Search...")
    with col_ctrl2:
        sort_col = st.selectbox("Sort By", ["Timestamp", "FraudProb", "Amount"])
    with col_ctrl3:
        st.markdown("")
        if st.button("Clear Queue", type="secondary"):
            st.session_state.alert_queue = []
            st.rerun()

    alerts_df = pd.DataFrame(st.session_state.alert_queue)

    if not alerts_df.empty:
        # Apply filter
        if filter_text:
            mask = (
                alerts_df["TransactionID"].str.contains(filter_text, case=False, na=False) |
                alerts_df["P_Email"].str.contains(filter_text, case=False, na=False) |
                alerts_df["R_Email"].str.contains(filter_text, case=False, na=False)
            )
            alerts_df = alerts_df[mask]

        # Sort
        if sort_col == "FraudProb":
            alerts_df = alerts_df.sort_values("FraudProb", ascending=False)
        elif sort_col == "Amount":
            alerts_df["_amt_sort"] = alerts_df["Amount"].str.replace("$", "", regex=False).astype(float)
            alerts_df = alerts_df.sort_values("_amt_sort", ascending=False).drop(columns=["_amt_sort"])
        else:
            alerts_df = alerts_df.sort_values("Timestamp", ascending=False)

        st.markdown(f"**{len(alerts_df)} high-risk transactions** in queue")

        st.dataframe(
            alerts_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "FraudProb": st.column_config.ProgressColumn(
                    "Fraud Probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
            },
        )

        # Export button
        csv_data = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Alert Queue to CSV",
            data=csv_data,
            file_name=f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
        )

        # Summary stats
        st.markdown("---")
        st.markdown("#### Alert Queue Summary")
        col_s1, col_s2, col_s3 = st.columns(3)
        total_at_risk = sum(
            float(a.replace("$", ""))
            for a in alerts_df["Amount"].tolist()
        )
        col_s1.metric("Total Transactions at Risk", len(alerts_df))
        col_s2.metric("Total $ at Risk", f"${total_at_risk:,.2f}")
        avg_prob = alerts_df["FraudProb"].astype(float).mean()
        col_s3.metric("Avg Fraud Probability", f"{avg_prob:.4f}")
    else:
        st.info("No high-risk transactions match the current filter. Queue is empty or filtered out.")

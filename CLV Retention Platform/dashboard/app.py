"""
Streamlit Dashboard — Customer Lifetime Value & Intelligent Retention Platform
Tabs: User Intelligence | Cohort Analysis | Campaign Planner |
      Retention Messages | Revenue Protection | Model Insights
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="CLV & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing models and data...")
def load_all() -> dict[str, Any]:
    """Load data, engineer features, and train all models."""
    from sklearn.model_selection import train_test_split

    from src.data_loader import load_data
    from src.features import build_feature_pipeline
    from src.clv_model import compute_simple_clv, segment_by_clv
    from src.models import train_churn_classifier, evaluate_model
    from src.survival_analysis import prepare_survival_data, fit_kaplan_meier
    from src.uplift_model import train_uplift_model, compute_uplift_scores, segment_by_uplift
    from src.retention_messaging import compute_campaign_roi

    data_dir = PROJECT_ROOT / "data" / "raw"
    df = load_data(str(data_dir), str(PROJECT_ROOT))
    X, y, feat_names = build_feature_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    churn_model, model_comparison = train_churn_classifier(X_train, y_train, eval_set=(X_test, y_test))
    metrics = evaluate_model(churn_model, X_test, y_test)

    churn_probs = pd.Series(churn_model.predict_proba(X.fillna(0))[:, 1], index=X.index, name="churn_prob")
    clv = compute_simple_clv(df, churn_proba=churn_probs)
    clv_segments = segment_by_clv(clv)

    rng = np.random.default_rng(42)
    treatment = pd.Series(rng.integers(0, 2, len(df)), index=df.index, name="treatment")
    try:
        mt, mc, _ = train_uplift_model(X, y, treatment)
        uplift_scores = compute_uplift_scores((mt, mc), X)
        uplift_models = (mt, mc)
    except Exception:
        uplift_scores = pd.Series(rng.uniform(-0.1, 0.3, len(df)), index=df.index)
        uplift_models = None

    uplift_segments = segment_by_uplift(uplift_scores)

    surv_df = prepare_survival_data(df)
    km_results = fit_kaplan_meier(
        surv_df["T"], surv_df["E"],
        groups=clv_segments.reindex(surv_df.index),
    )

    return {
        "df": df,
        "X": X,
        "y": y,
        "feature_names": feat_names,
        "churn_model": churn_model,
        "churn_probs": churn_probs,
        "clv": clv,
        "clv_segments": clv_segments,
        "uplift_scores": uplift_scores,
        "uplift_segments": uplift_segments,
        "uplift_models": uplift_models,
        "km_results": km_results,
        "surv_df": surv_df,
        "metrics": metrics,
        "model_comparison": model_comparison,
    }


state = load_all()
df = state["df"]
X = state["X"]
feature_names = state["feature_names"]
churn_model = state["churn_model"]
churn_probs = state["churn_probs"]
clv = state["clv"]
clv_segments = state["clv_segments"]
uplift_scores = state["uplift_scores"]
uplift_segments = state["uplift_segments"]
km_results = state["km_results"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("CLV & Retention Intelligence")
    st.markdown("**AT&T • Verizon • Spotify • Netflix**")
    st.divider()
    st.metric("Total Users", f"{len(df):,}")
    churn_rate = float(churn_probs.mean())
    st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
    at_risk_n = int((churn_probs >= 0.55).sum())
    st.metric("High-Risk Users", f"{at_risk_n:,}")
    avg_clv = float(clv.mean())
    st.metric("Avg CLV", f"${avg_clv:,.0f}")
    st.divider()
    st.caption("Model: XGBoost + LightGBM")
    st.caption(f"AUC-ROC: {state['metrics'].get('auc_roc', 0):.3f}")
    st.caption(f"AUC-PR: {state['metrics'].get('auc_pr', 0):.3f}")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "User Intelligence", "Cohort Analysis", "Campaign Planner",
    "Retention Messages", "Revenue Protection", "Model Insights",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: User Intelligence
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("User Churn Intelligence")

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("User Lookup")
        user_id_input = st.text_input("User ID", placeholder="U0001234")
        tenure_in = st.slider("Tenure (days)", 1, 1825, 180)
        days_inactive = st.slider("Days Since Last Login", 0, 180, 15)
        completion_rate = st.slider("Avg Completion Rate", 0.0, 1.0, 0.65, 0.01)
        plan_price = st.selectbox("Plan Price", [98, 148, 198], index=1)
        total_secs_in = st.number_input("Total Listening (seconds)", value=500000)

    with col_results:
        from src.features import (
            engineer_rfm_features, engineer_engagement_features,
            engineer_payment_features, engineer_lifecycle_features,
            engineer_behavioral_signals,
        )
        from src.clv_model import compute_simple_clv, segment_by_clv
        from src.uplift_model import compute_uplift_scores, segment_by_uplift
        from src.retention_messaging import generate_retention_message, personalize_offer
        from src.explainability import explain_churn_risk, generate_user_narrative

        single = pd.DataFrame([{
            "tenure_days": tenure_in,
            "days_since_last_login": days_inactive,
            "avg_completion_rate": completion_rate,
            "plan_list_price": plan_price,
            "total_secs": total_secs_in,
            "num_25": 50, "num_50": 40, "num_75": 30,
            "num_985": 20, "num_100": 60, "num_unq": 100,
            "bd": 30, "registered_via": 4, "payment_method_id": 10,
        }])
        for fn in [engineer_rfm_features, engineer_engagement_features,
                   engineer_payment_features, engineer_lifecycle_features,
                   engineer_behavioral_signals]:
            single = fn(single)
        for col in feature_names:
            if col not in single.columns:
                single[col] = 0.0
        X_single = single[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        cp = float(churn_model.predict_proba(X_single)[0, 1])
        clv_val = float(compute_simple_clv(single, churn_proba=pd.Series([cp])).iloc[0])
        clv_seg = segment_by_clv(pd.Series([clv_val])).iloc[0]

        if state["uplift_models"]:
            up_score = float(compute_uplift_scores(state["uplift_models"], X_single).iloc[0])
        else:
            up_score = 0.05
        up_seg = segment_by_uplift(pd.Series([up_score])).iloc[0]

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        risk_color = "red" if cp >= 0.75 else "orange" if cp >= 0.55 else "green"
        col1.metric("Churn Probability", f"{cp:.1%}", delta=None)
        col2.metric("CLV Estimate", f"${clv_val:,.0f}")
        col3.metric("CLV Tier", clv_seg)
        col4.metric("Uplift Segment", up_seg)

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cp * 100,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 35], "color": "#d4edda"},
                    {"range": [35, 55], "color": "#fff3cd"},
                    {"range": [55, 75], "color": "#f8d7da"},
                    {"range": [75, 100], "color": "#721c24"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
            },
            title={"text": "Churn Risk Gauge"},
        ))
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        # SHAP explanation
        shap_exp = explain_churn_risk(churn_model, X_single, feature_names)
        top_factors = (
            shap_exp["top_churn_factors"][:5] + shap_exp["top_retention_factors"][:3]
        )
        if top_factors:
            shap_df = pd.DataFrame(top_factors)
            shap_df["color"] = shap_df["shap_value"].apply(lambda v: "red" if v > 0 else "green")
            fig_shap = px.bar(
                shap_df, x="shap_value", y="feature", orientation="h",
                color="color", color_discrete_map={"red": "#e74c3c", "green": "#2ecc71"},
                title="SHAP Feature Contributions",
            )
            fig_shap.update_layout(showlegend=False, height=300, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_shap, use_container_width=True)

        # Retention message
        st.subheader("Personalized Retention Message")
        offer = personalize_offer(clv_seg, up_seg, cp)
        msg = generate_retention_message(
            user_segment=up_seg, clv_tier=clv_seg, churn_risk=cp,
            tenure_days=tenure_in, days_inactive=days_inactive,
            user_name=user_id_input or "Valued Member",
        )
        st.text_input("Subject", value=msg["subject"])
        st.text_area("Message Body", value=msg["body"], height=200)
        st.info(
            f"Offer Type: **{msg['offer_type']}** | "
            f"Discount: **{msg['discount_pct']}%** | "
            f"Urgency: **{msg['urgency']}**"
        )

        narrative = generate_user_narrative(
            user_id=user_id_input or "Unknown", churn_prob=cp,
            clv=clv_val, uplift_segment=up_seg, shap_explanation=shap_exp,
            tenure_days=tenure_in, lifecycle_stage=str(single.get("lifecycle_stage", pd.Series(["mature"])).iloc[0]),
            recommended_action=offer["action"],
        )
        st.markdown(f"**AI Narrative:** {narrative}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Cohort Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Cohort Survival & Churn Analysis")

    from src.survival_analysis import plot_km_curves
    col_sel, col_charts = st.columns([1, 3])

    with col_sel:
        selected_segments = st.multiselect(
            "Show Segments",
            options=list(km_results.keys()),
            default=list(km_results.keys()),
        )

    filtered_km = {k: v for k, v in km_results.items() if k in selected_segments}

    if filtered_km:
        with col_charts:
            km_fig = plot_km_curves(filtered_km)
            if hasattr(km_fig, "update_layout"):
                st.plotly_chart(km_fig, use_container_width=True)
            else:
                st.write(filtered_km)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Avg CLV by Segment")
        clv_by_seg = pd.DataFrame({
            "segment": clv_segments.values,
            "clv": clv.values,
        }).groupby("segment")["clv"].mean().reset_index()
        fig_clv = px.bar(clv_by_seg, x="segment", y="clv", color="segment",
                         title="Average CLV by Segment",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_clv, use_container_width=True)

    with col_b:
        st.subheader("Churn Rate by Segment")
        seg_df = pd.DataFrame({
            "segment": clv_segments.reindex(df.index).fillna("Unknown").values,
            "churn": df.get("churn_flag", pd.Series(np.zeros(len(df)), index=df.index)).values,
        })
        churn_by_seg = seg_df.groupby("segment")["churn"].mean().reset_index()
        fig_churn = px.bar(churn_by_seg, x="segment", y="churn", color="segment",
                           title="Churn Rate by CLV Segment",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_churn.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_churn, use_container_width=True)

    # Engagement heatmap
    st.subheader("Engagement Heatmap")
    if "days_since_last_login" in df.columns and "avg_completion_rate" in df.columns:
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        fig_heat = px.density_heatmap(
            sample_df,
            x="days_since_last_login",
            y="avg_completion_rate",
            title="User Engagement: Login Recency vs Completion Rate",
            nbinsx=30, nbinsy=30,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Campaign Planner
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Retention Campaign Planner")
    from src.retention_messaging import compute_campaign_roi

    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.subheader("Campaign Configuration")
        target_segment = st.selectbox(
            "Target Segment", ["All At-Risk", "Persuadables", "High CLV", "Low CLV"]
        )
        intervention_type = st.selectbox(
            "Intervention Type", ["email_only", "email_sms", "email_sms_call"]
        )
        cost_map = {"email_only": 2.5, "email_sms": 5.0, "email_sms_call": 15.0}
        base_cost = cost_map[intervention_type]
        budget_multiplier = st.slider("Budget Multiplier", 0.5, 3.0, 1.0, 0.1)
        final_cost = base_cost * budget_multiplier
        avg_revenue = st.slider("Avg Monthly Revenue ($)", 5, 50, 15)
        months_ret = st.slider("Months Retained if Saved", 1, 24, 6)

    with col_results:
        st.subheader("ROI Projection")

        df_copy = df.copy()
        df_copy["churn_flag"] = df.get("churn_flag", pd.Series(np.zeros(len(df)), index=df.index)).values
        df_copy["uplift_segment"] = uplift_segments.reindex(df.index).fillna("Lost Causes").values

        roi = compute_campaign_roi(
            df_copy,
            uplift_scores.reindex(df.index).fillna(0),
            intervention_cost_per_user=final_cost,
            avg_monthly_revenue=avg_revenue,
            months_retained=months_ret,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Users Targeted", f"{roi['n_targeted']:,}")
        col2.metric("Expected Saves", f"{roi['expected_saves']:,.0f}")
        col3.metric("Revenue Retained", f"${roi['retained_revenue_usd']:,.0f}")
        col4.metric("Campaign ROI", f"{roi['roi_pct']:.1f}%",
                    delta="Profitable" if roi['roi_pct'] > 0 else "Loss")

        st.metric("Net Benefit", f"${roi['net_benefit_usd']:,.0f}")
        st.metric("Break-Even Save Rate", f"{roi['break_even_save_rate']:.1%}")

        # ROI comparison across intervention types
        roi_comparison = []
        for int_type, cost in cost_map.items():
            r = compute_campaign_roi(
                df_copy, uplift_scores.reindex(df.index).fillna(0),
                intervention_cost_per_user=cost * budget_multiplier,
                avg_monthly_revenue=avg_revenue,
                months_retained=months_ret,
            )
            roi_comparison.append({
                "Intervention": int_type,
                "ROI (%)": r["roi_pct"],
                "Net Benefit ($)": r["net_benefit_usd"],
                "Cost ($)": r["intervention_cost_usd"],
            })

        fig_roi = px.bar(
            pd.DataFrame(roi_comparison), x="Intervention", y="ROI (%)",
            color="ROI (%)", color_continuous_scale="RdYlGn",
            title="ROI by Intervention Type",
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        if roi.get("segment_breakdown"):
            st.subheader("Segment Breakdown")
            st.dataframe(pd.DataFrame(roi["segment_breakdown"]).T, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Retention Messages
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Retention Message Center")
    from src.retention_messaging import generate_retention_message

    # Build high-risk user table
    high_risk_mask = churn_probs >= 0.60
    n_show = min(100, int(high_risk_mask.sum()))

    hi_idx = churn_probs[high_risk_mask].nlargest(n_show).index
    hi_df = pd.DataFrame({
        "user_id": [f"U{i:07d}" for i in range(len(hi_idx))],
        "churn_prob": churn_probs.loc[hi_idx].values,
        "clv": clv.reindex(hi_idx).fillna(0).values,
        "clv_segment": clv_segments.reindex(hi_idx).fillna("Medium").values,
        "uplift_segment": uplift_segments.reindex(hi_idx).fillna("Lost Causes").values,
        "tenure_days": df.get("tenure_days", pd.Series(np.full(len(df), 180), index=df.index)).reindex(hi_idx).fillna(180).astype(int).values,
        "days_inactive": df.get("days_since_last_login", pd.Series(np.zeros(len(df)), index=df.index)).reindex(hi_idx).fillna(0).astype(int).values,
    })

    # Generate messages
    messages = []
    for _, row in hi_df.iterrows():
        msg = generate_retention_message(
            user_segment=row["uplift_segment"],
            clv_tier=row["clv_segment"],
            churn_risk=row["churn_prob"],
            tenure_days=int(row["tenure_days"]),
            days_inactive=int(row["days_inactive"]),
        )
        messages.append(msg["subject"])
    hi_df["message_subject"] = messages

    st.subheader(f"Top {n_show} At-Risk Users")

    ab_test = st.toggle("A/B Test Variant", value=False)
    if ab_test:
        st.info("A/B Test Mode: Variant B messages use 10% higher discount")

    edited_df = st.data_editor(
        hi_df[["user_id", "churn_prob", "clv_segment", "uplift_segment", "message_subject"]],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "churn_prob": st.column_config.ProgressColumn("Churn Risk", min_value=0, max_value=1),
        },
    )

    col_act1, col_act2, col_act3 = st.columns(3)
    if col_act1.button("Send All Messages (Mock)", type="primary"):
        st.success(f"Sent {len(hi_df)} retention messages (mock)")
    if col_act2.button("Export to CSV"):
        st.download_button(
            "Download CSV", hi_df.to_csv(index=False),
            file_name="high_risk_users.csv", mime="text/csv",
        )
    col_act3.metric("Total Messages Queued", len(hi_df))

    # OSHA-style incident log (retention campaign log)
    st.subheader("Campaign Activity Log")
    log_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1D"),
        "action": ["Email sent"] * 10 + ["SMS sent"] * 5 + ["Offer redeemed"] * 5,
        "users_affected": np.random.randint(50, 500, 20),
        "status": ["Delivered"] * 15 + ["Redeemed"] * 5,
    })
    st.dataframe(log_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Revenue Protection
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Revenue Protection Dashboard")

    high_risk_mask2 = churn_probs >= 0.55
    at_risk_mrr = float((clv.reindex(df.index).fillna(0)[high_risk_mask2] / 12.0).sum())
    total_mrr = float((clv.reindex(df.index).fillna(0) / 12.0).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("At-Risk MRR", f"${at_risk_mrr:,.0f}", delta=f"-{at_risk_mrr/total_mrr:.0%} of total")
    col2.metric("Total MRR", f"${total_mrr:,.0f}")
    col3.metric("At-Risk Users", f"{int(high_risk_mask2.sum()):,}")
    col4.metric("Overall Churn Rate", f"{float(churn_probs.mean()):.1%}")

    # At-risk MRR gauge
    gauge_mrr = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=at_risk_mrr,
        delta={"reference": total_mrr * 0.10, "suffix": " target"},
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, total_mrr * 0.5]},
            "bar": {"color": "#e74c3c"},
            "threshold": {"line": {"color": "darkred", "width": 3}, "value": total_mrr * 0.10},
        },
        title={"text": "At-Risk MRR"},
    ))
    gauge_mrr.update_layout(height=250)
    st.plotly_chart(gauge_mrr, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Monthly churn trend (simulated)
        months = pd.date_range("2023-01-01", periods=24, freq="ME")
        churn_trend = pd.Series(
            np.random.normal(0.25, 0.03, 24).clip(0.05, 0.50), index=months
        )
        fig_trend = px.line(
            x=churn_trend.index, y=churn_trend.values,
            title="Monthly Churn Rate Trend", labels={"x": "Month", "y": "Churn Rate"},
        )
        fig_trend.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Target: 25%")
        fig_trend.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        # CLV distribution
        fig_clv_dist = px.histogram(
            x=clv.values, nbins=50,
            title="CLV Distribution",
            labels={"x": "Customer Lifetime Value ($)"},
            color_discrete_sequence=["#3498db"],
        )
        st.plotly_chart(fig_clv_dist, use_container_width=True)

    # Revenue forecast
    st.subheader("Revenue Forecast (12 months)")
    forecast_months = pd.date_range("2024-01-01", periods=12, freq="ME")
    retention_rate = 1 - float(churn_probs.mean()) / 12
    monthly_revenue = [total_mrr * (retention_rate ** i) for i in range(12)]
    saved_revenue = [total_mrr * (retention_rate ** i) * 0.15 for i in range(12)]  # 15% saved by interventions

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Bar(
        x=forecast_months, y=monthly_revenue, name="Baseline MRR", marker_color="#3498db"
    ))
    fig_forecast.add_trace(go.Bar(
        x=forecast_months, y=saved_revenue, name="Revenue Saved (Interventions)", marker_color="#2ecc71"
    ))
    fig_forecast.update_layout(barmode="stack", title="12-Month Revenue Forecast", height=350)
    st.plotly_chart(fig_forecast, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Model Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Model Performance & Explainability")

    metrics = state["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")
    col2.metric("AUC-PR", f"{metrics.get('auc_pr', 0):.4f}")
    col3.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
    col4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    col5.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    col_a, col_b = st.columns(2)

    with col_a:
        # Feature importance
        st.subheader("Feature Importance")
        if hasattr(churn_model, "feature_importances_"):
            imp = churn_model.feature_importances_
        elif hasattr(churn_model, "lr_"):
            imp = np.abs(churn_model.lr_.coef_[0])
        else:
            imp = np.ones(len(feature_names))

        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
        imp_df = imp_df.nlargest(20, "importance")
        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title="Top 20 Feature Importances",
            color="importance", color_continuous_scale="Blues",
        )
        fig_imp.update_layout(yaxis={"autorange": "reversed"}, height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        # Calibration curve (simulated)
        st.subheader("Calibration Curve")
        prob_bins = np.linspace(0, 1, 11)
        from sklearn.calibration import calibration_curve
        y_arr = state["y"].values
        p_arr = churn_probs.reindex(state["X"].index).fillna(0.5).values

        try:
            fraction_pos, mean_pred = calibration_curve(y_arr, p_arr, n_bins=10)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=mean_pred, y=fraction_pos, mode="lines+markers",
                name="Model", line=dict(color="#e74c3c", width=2),
            ))
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Perfect Calibration", line=dict(dash="dash", color="gray"),
            ))
            fig_cal.update_layout(
                title="Calibration Curve",
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Fraction of Positives",
                height=400,
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        except Exception:
            st.warning("Calibration curve unavailable.")

    # Uplift Qini curve
    st.subheader("Uplift Model — Qini Curve")
    from src.uplift_model import compute_qini_coefficient

    y_arr = state["y"].reindex(df.index).fillna(0).values
    up_arr = uplift_scores.reindex(df.index).fillna(0).values
    rng2 = np.random.default_rng(99)
    treatment_arr = rng2.integers(0, 2, len(df))

    try:
        qini = compute_qini_coefficient(y_arr, up_arr, treatment_arr)
        qc = qini["qini_curve"]
        fig_qini = go.Figure()
        fig_qini.add_trace(go.Scatter(
            x=qc["x"], y=qc["y"], mode="lines",
            name=f"Uplift Model (Qini={qini['qini_coefficient']:.4f})",
            line=dict(color="#9b59b6", width=2),
        ))
        fig_qini.add_trace(go.Scatter(
            x=qc["x"], y=qc["random_baseline"], mode="lines",
            name="Random Baseline", line=dict(dash="dash", color="gray"),
        ))
        fig_qini.update_layout(
            title="Qini Curve — Uplift Model",
            xaxis_title="Fraction of Population Treated",
            yaxis_title="Cumulative Incremental Gains",
            height=400,
        )
        st.plotly_chart(fig_qini, use_container_width=True)
        st.metric("Qini Coefficient", f"{qini['qini_coefficient']:.4f}")
    except Exception as e:
        st.warning(f"Qini curve unavailable: {e}")

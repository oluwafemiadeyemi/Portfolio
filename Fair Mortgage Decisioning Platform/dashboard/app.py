"""
Fair Mortgage Underwriting Dashboard
Streamlit dashboard for underwriting decisions, portfolio fairness monitoring,
and geographic analysis.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, generate_synthetic_mortgage_data
from src.features import build_feature_pipeline, engineer_affordability_features
from src.features import engineer_risk_features, engineer_geographic_features, encode_categoricals
from src.fairness import (
    generate_fairness_report,
    compute_demographic_parity_difference,
    compute_disparate_impact_ratio,
    run_ecoa_compliance_test,
    PROTECTED_ATTRIBUTES,
)
from src.explainability import explain_single_application, generate_counterfactual

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fair Mortgage Decisioning Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
if str(PORTFOLIO_ROOT) not in sys.path:
    sys.path.insert(0, str(PORTFOLIO_ROOT))
try:
    from shared.design_system import (
        apply_theme, kpi_card, section_header, chart_layout,
        page_hero, sidebar_brand, risk_badge, gauge_chart, COLORS,
    )
    apply_theme()
except ImportError:
    pass

# ── Project-specific CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.pass-badge { color: #00C896; font-weight: 700; font-size: 1.05em; }
.fail-badge { color: #FF5A65; font-weight: 700; font-size: 1.05em; }
.warning-badge { color: #FFB020; font-weight: 700; font-size: 1.05em; }
.approved { color: #00C896; font-size: 1.4em; font-weight: 700; }
.denied   { color: #FF5A65; font-size: 1.4em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and data...")
def load_model_and_data():
    """Loads model and portfolio data (cached)."""
    from src.models import load_artifacts, train_lightgbm, train_logistic_regression, save_artifacts

    data_dir = str(PROJECT_ROOT / "data")
    model_dir = str(PROJECT_ROOT / "data" / "models")

    # Load data
    df = load_data(data_dir)

    # Build features (filters to actionable records — df must be realigned to X's index)
    X, y, feature_cols, protected_cols = build_feature_pipeline(df)
    df = df.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Try loading saved model
    model = None
    try:
        model, feature_cols_saved, threshold = load_artifacts(model_dir, "lightgbm")
        feature_cols = feature_cols_saved
    except (FileNotFoundError, Exception):
        st.info("Training model on portfolio data...")
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        try:
            model = train_lightgbm(X_train, y_train)
        except Exception:
            model = train_logistic_regression(X_train, y_train)
        threshold = 0.5
        save_artifacts(model, list(feature_cols), threshold, model_dir, "lightgbm")

    # Align features
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].fillna(0).astype(float)

    probas = model.predict_proba(X)[:, 1]
    preds = (probas >= 0.5).astype(int)

    return model, feature_cols, df, X, y, probas, preds, protected_cols


def _align_to_model(raw_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Aligns feature DataFrame to model's expected columns."""
    for col in feature_cols:
        if col not in raw_df.columns:
            raw_df[col] = 0.0
    return raw_df[feature_cols].fillna(0).astype(float)


# ── Sidebar Navigation ─────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/HMDA_Logo.svg/200px-HMDA_Logo.svg.png",
        use_container_width=True,
        caption="HMDA Compliant",
    )
    st.sidebar.title("Fair Mortgage Platform")
    st.sidebar.markdown("**ECOA | Fair Housing Act | HMDA 2022**")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Underwriting Decision",
            "Portfolio Overview",
            "Fairness Monitor",
            "Model Performance",
            "Geographic Risk",
        ],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.caption("Powered by LightGBM + SHAP")
    st.sidebar.caption("MIT Applied AI & Data Science")
    return page


# ── Underwriting Page ──────────────────────────────────────────────────────────
def render_underwriting(model, feature_cols, df):
    st.title("Mortgage Underwriting Decision Engine")
    st.markdown("Submit an application for instant AI-assisted underwriting with full explainability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=50000, max_value=2000000, value=320000, step=5000)
        income = st.number_input("Annual Income ($)", min_value=20000, max_value=1000000, value=95000, step=1000)
        dti = st.slider("Debt-to-Income Ratio (%)", min_value=5.0, max_value=65.0, value=35.0, step=0.5)
        ltv = st.slider("Loan-to-Value Ratio (%)", min_value=50.0, max_value=100.0, value=82.0, step=0.5)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 240, 120], index=0)

    with col2:
        st.subheader("Property & Loan Type")
        loan_purpose = st.selectbox(
            "Loan Purpose",
            [(1, "Home Purchase"), (31, "Refinancing"), (32, "Cash-out Refi"), (2, "Home Improvement")],
            format_func=lambda x: x[1],
        )[0]
        property_type = st.selectbox(
            "Property Type",
            [(1, "1-4 Family"), (2, "Manufactured Home"), (3, "Multifamily")],
            format_func=lambda x: x[1],
        )[0]
        occupancy_type = st.selectbox(
            "Occupancy Type",
            [(1, "Primary Residence"), (2, "Second Residence"), (3, "Investment Property")],
            format_func=lambda x: x[1],
        )[0]
        lien_status = st.selectbox(
            "Lien Status",
            [(1, "First Lien"), (2, "Subordinate Lien")],
            format_func=lambda x: x[1],
        )[0]
        co_applicant = st.checkbox("Co-Applicant Present", value=False)

    with col3:
        st.subheader("Applicant Demographics")
        st.info("ECOA-protected info collected for compliance monitoring only — NOT used in credit decision.")
        applicant_sex = st.selectbox(
            "Sex (optional)",
            [(3, "Information Not Provided"), (1, "Male"), (2, "Female")],
            format_func=lambda x: x[1],
        )[0]
        applicant_race = st.selectbox(
            "Race (optional)",
            [
                (6, "Information Not Provided"),
                (1, "American Indian/Alaska Native"),
                (2, "Asian"),
                (3, "Black/African American"),
                (4, "Native Hawaiian/Pacific Islander"),
                (5, "White"),
            ],
            format_func=lambda x: x[1],
        )[0]
        applicant_age = st.selectbox(
            "Age Group (optional)",
            ["35-44", "18-24", "25-34", "45-54", "55-64", "65+"],
        )

    st.divider()
    if st.button("Score Application", type="primary", use_container_width=True):
        with st.spinner("Analyzing application..."):
            row = {
                "loan_amount": loan_amount,
                "income": income,
                "debt_to_income_ratio": dti,
                "loan_to_value_ratio": ltv,
                "loan_purpose": loan_purpose,
                "property_type": property_type,
                "occupancy_type": occupancy_type,
                "lien_status": lien_status,
                "applicant_sex": applicant_sex,
                "applicant_race_1": applicant_race,
                "applicant_ethnicity_1": 3,
                "applicant_age": applicant_age,
                "applicant_age_raw": 38,
                "co_applicant_present": int(co_applicant),
                "co_applicant_sex": 5,
                "co_applicant_race_1": 8,
                "loan_term": loan_term,
                "county_code": "0600001",
                "census_tract": "0600001000",
                "action_taken": 1,
                "preapproval": 2,
                "construction_method": 1,
                "negative_amortization": 2,
                "interest_only_payments": 2,
                "balloon_payment": 2,
                "open_end_line_of_credit": 2,
                "reverse_mortgage": 2,
                "business_or_commercial_purpose": 2,
                "hoepa_status": 3,
            }
            raw_df = pd.DataFrame([row])
            raw_df = engineer_affordability_features(raw_df)
            raw_df = engineer_risk_features(raw_df)
            raw_df = engineer_geographic_features(raw_df)
            raw_df = encode_categoricals(raw_df)
            X_single = _align_to_model(raw_df, feature_cols)

            prob = float(model.predict_proba(X_single)[0, 1])
            decision = "APPROVED" if prob >= 0.5 else "DENIED"

            shap_exp = explain_single_application(model, X_single, feature_cols, 0.5)
            counterfactual = generate_counterfactual(shap_exp, X_single, feature_cols, model, 0.5)

        # ── Result display ──────────────────────────────────────────────────
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if decision == "APPROVED":
                st.markdown(f'<p class="approved">APPROVED ✓</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="denied">DENIED ✗</p>', unsafe_allow_html=True)
            st.metric("Approval Probability", f"{prob:.1%}")

        with res_col2:
            st.metric("Confidence", f"{abs(prob - 0.5) * 200:.1f}%")
            risk_tier = raw_df["risk_tier"].iloc[0] if "risk_tier" in raw_df.columns else "medium"
            st.metric("Risk Tier", risk_tier.upper())

        with res_col3:
            monthly_pmt = raw_df["monthly_payment_estimate"].iloc[0] if "monthly_payment_estimate" in raw_df.columns else loan_amount / loan_term
            st.metric("Est. Monthly Payment", f"${monthly_pmt:,.0f}")
            affordability = raw_df["affordability_index"].iloc[0] if "affordability_index" in raw_df.columns else 100
            st.metric("Affordability Index", f"{affordability:.1f}")

        # ── SHAP Waterfall ─────────────────────────────────────────────────
        st.subheader("Decision Explanation (SHAP)")
        factors = shap_exp.get("top_approval_factors", []) + shap_exp.get("top_denial_factors", [])
        factors_sorted = sorted(factors, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:10]

        if factors_sorted and _HAS_PLOTLY:
            feat_names = [f.get("feature", "").replace("_", " ").title() for f in factors_sorted]
            shap_vals = [f.get("shap_value", 0) for f in factors_sorted]
            colors = ["#28a745" if v > 0 else "#dc3545" for v in shap_vals]

            fig = go.Figure(go.Bar(
                x=shap_vals,
                y=feat_names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition="outside",
            ))
            fig.update_layout(
                title="SHAP Feature Contributions",
                xaxis_title="SHAP Value (impact on approval probability)",
                height=400,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Adverse Action Codes ──────────────────────────────────────────
        if decision == "DENIED":
            st.subheader("Adverse Action Notice (ECOA Required)")
            from src.fairness import generate_adverse_action_codes
            adverse_codes = generate_adverse_action_codes(shap_exp, top_n=4)
            for i, code in enumerate(adverse_codes, 1):
                st.markdown(f"**Reason {i}:** {code['description']} (Code {code['code']})")

            # Counterfactual
            st.subheader("How to Improve Your Application")
            suggestions = counterfactual.get("suggestions", [])
            if suggestions:
                for s in suggestions[:3]:
                    with st.expander(f"Option: {s.get('description', '')[:80]}..."):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Current Value", f"{s.get('current_value', 0):,.2f}")
                        with col_b:
                            st.metric("Suggested Value", f"{s.get('suggested_value', 0):,.2f}")
                        new_prob = s.get("expected_approval_probability", prob)
                        st.progress(new_prob, text=f"New approval probability: {new_prob:.1%}")
                        if s.get("would_approve"):
                            st.success("This change would likely result in approval.")
            else:
                st.info(counterfactual.get("message", "Complex risk profile. Consult a loan officer."))


# ── Portfolio Overview Page ────────────────────────────────────────────────────
def render_portfolio_overview(df, y, probas, preds):
    st.title("Portfolio Overview")
    st.markdown("Portfolio-level statistics from HMDA 2022 mortgage applications.")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Applications", f"{len(df):,}")
    k2.metric("Approval Rate", f"{preds.mean():.1%}")
    k3.metric("Avg Loan Amount", f"${df['loan_amount'].median():,.0f}")
    k4.metric("Avg Income", f"${df['income'].median():,.0f}")

    st.divider()

    if not _HAS_PLOTLY:
        st.warning("Plotly not installed. Showing tabular data.")
        st.dataframe(df.describe())
        return

    col1, col2 = st.columns(2)

    with col1:
        # Approval rate over loan amount buckets
        df_plot = df.copy()
        df_plot["approved"] = preds
        df_plot["loan_bucket"] = pd.cut(df_plot["loan_amount"], bins=10)
        approval_by_loan = df_plot.groupby("loan_bucket", observed=True)["approved"].mean().reset_index()
        approval_by_loan["loan_bucket"] = approval_by_loan["loan_bucket"].astype(str)

        fig = px.bar(
            approval_by_loan,
            x="loan_bucket",
            y="approved",
            title="Approval Rate by Loan Amount",
            labels={"approved": "Approval Rate", "loan_bucket": "Loan Amount Range"},
            color="approved",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Loan amount distribution
        fig = px.histogram(
            df,
            x="loan_amount",
            nbins=50,
            title="Loan Amount Distribution",
            labels={"loan_amount": "Loan Amount ($)"},
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Approval probability distribution
        fig = px.histogram(
            x=probas,
            nbins=50,
            title="Approval Probability Distribution",
            labels={"x": "Approval Probability"},
            color_discrete_sequence=["#2ca02c"],
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # DTI vs approval rate scatter
        df_plot2 = df.copy()
        df_plot2["approved"] = preds
        df_plot2["dti_bucket"] = pd.cut(df_plot2["debt_to_income_ratio"], bins=6)
        by_dti = df_plot2.groupby("dti_bucket", observed=True)["approved"].mean().reset_index()
        by_dti["dti_bucket"] = by_dti["dti_bucket"].astype(str)

        fig = px.bar(
            by_dti,
            x="dti_bucket",
            y="approved",
            title="Approval Rate by DTI Ratio",
            labels={"approved": "Approval Rate", "dti_bucket": "DTI Range"},
            color="approved",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Geographic Choropleth (state-level)
    st.subheader("Geographic Approval Rate Map")
    if "county_code" in df.columns:
        df_geo = df.copy()
        df_geo["approved"] = preds
        df_geo["state_fips"] = df_geo["county_code"].astype(str).str[:2]

        state_fips_to_abbr = {
            "06": "CA", "48": "TX", "12": "FL", "36": "NY", "17": "IL",
            "42": "PA", "39": "OH", "13": "GA", "37": "NC", "26": "MI",
            "34": "NJ", "04": "AZ", "19": "IA", "47": "TN", "55": "WI",
            "27": "MN", "53": "WA", "25": "MA", "08": "CO", "24": "MD",
        }
        df_geo["state"] = df_geo["state_fips"].map(state_fips_to_abbr)
        by_state = df_geo.groupby("state")["approved"].agg(["mean", "count"]).reset_index()
        by_state.columns = ["state", "approval_rate", "n"]

        fig = px.choropleth(
            by_state,
            locations="state",
            locationmode="USA-states",
            color="approval_rate",
            scope="usa",
            title="Approval Rate by State",
            color_continuous_scale="RdYlGn",
            range_color=[0.5, 1.0],
            labels={"approval_rate": "Approval Rate"},
            hover_data={"n": True},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Fairness Monitor Page ──────────────────────────────────────────────────────
def render_fairness_monitor(df, y, probas, preds):
    st.title("Fairness Monitor — ECOA/Fair Housing Act Compliance")
    st.markdown("Real-time monitoring of protected attribute parity and disparate impact ratios.")

    # ECOA Compliance summary
    with st.spinner("Computing compliance metrics..."):
        compliance = run_ecoa_compliance_test(preds, y.values, df)

    overall = compliance.get("overall_status", "UNKNOWN")
    status_color = "pass-badge" if overall == "PASS" else "fail-badge"
    st.markdown(
        f"**Overall ECOA Status:** <span class='{status_color}'>{overall}</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"Attributes tested: {', '.join(compliance.get('attributes_tested', []))}")

    # Compliance badges per attribute
    badge_cols = st.columns(len(compliance.get("attributes_tested", [])) or 1)
    for i, attr in enumerate(compliance.get("attributes_tested", [])):
        passes = compliance["compliance_summary"].get(attr, {}).get("passes_ecoa", True)
        badge = "PASS" if passes else "FAIL"
        color = "pass-badge" if passes else "fail-badge"
        badge_cols[i % len(badge_cols)].markdown(
            f"**{attr.replace('_', ' ').title()}**<br><span class='{color}'>{badge}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    if not _HAS_PLOTLY:
        st.dataframe(pd.DataFrame(compliance["compliance_summary"]).T)
        return

    # Demographic parity charts
    st.subheader("Demographic Parity — Approval Rates by Group")
    attr_tabs = st.tabs(["Race", "Sex", "Ethnicity", "Age Group"])
    attr_cols = ["applicant_race_1", "applicant_sex", "applicant_ethnicity_1", "applicant_age_group"]

    for tab, attr in zip(attr_tabs, attr_cols):
        with tab:
            if attr not in df.columns:
                st.info(f"Column '{attr}' not in dataset.")
                continue

            dp = compute_demographic_parity_difference(preds, attr, df)
            groups = dp.get("groups", {})
            if not groups:
                st.info("Insufficient data for this attribute.")
                continue

            group_names = list(groups.keys())
            approval_rates = [groups[g]["approval_rate"] for g in group_names]
            ns = [groups[g]["n"] for g in group_names]

            fig = px.bar(
                x=group_names,
                y=approval_rates,
                title=f"Approval Rate by {attr.replace('_', ' ').title()}",
                labels={"x": "Group", "y": "Approval Rate"},
                color=approval_rates,
                color_continuous_scale="RdYlGn",
                range_color=[0.5, 1.0],
                text=[f"{r:.1%}" for r in approval_rates],
            )
            fig.add_hline(y=0.8 * max(approval_rates) if approval_rates else 0.8,
                          line_dash="dash", line_color="orange",
                          annotation_text="80% Rule Threshold")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Disparate impact ratio table
            di = compute_disparate_impact_ratio(preds, attr, df)
            di_groups = di.get("groups", {})
            if di_groups:
                di_df = pd.DataFrame([
                    {
                        "Group": k,
                        "Approval Rate": f"{v['approval_rate']:.1%}",
                        "Disparate Impact Ratio": f"{v['disparate_impact_ratio']:.3f}",
                        "Passes 80% Rule": "PASS" if v["passes_80_rule"] else "FAIL",
                        "N Applications": v["n"],
                    }
                    for k, v in di_groups.items()
                ])
                st.dataframe(di_df, use_container_width=True, hide_index=True)


# ── Model Performance Page ─────────────────────────────────────────────────────
def render_model_performance(model, feature_cols, df, y, probas, preds):
    st.title("Model Performance")
    st.markdown("ROC, Precision-Recall, and Calibration curves for the underwriting model.")

    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
        roc_auc_score, confusion_matrix, classification_report,
    )
    from sklearn.calibration import calibration_curve

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC AUC", f"{roc_auc_score(y, probas):.4f}")
    k2.metric("Avg Precision", f"{average_precision_score(y, probas):.4f}")
    k3.metric("F1 Score", f"{((preds == y.values).sum() / len(y)):.4f}")
    k4.metric("Approval Rate", f"{preds.mean():.1%}")

    if not _HAS_PLOTLY:
        st.warning("Install plotly for interactive charts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prec, rec, _ = precision_recall_curve(y, probas)
        ap = average_precision_score(y, probas)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP={ap:.3f})", line=dict(color="#2ca02c", width=2)))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=400)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Calibration curve
        frac_pos, mean_pred = calibration_curve(y, probas, n_bins=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Model", line=dict(color="#ff7f0e")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="Calibration Curve", xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Confusion matrix
        cm = confusion_matrix(y, preds)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Denied", "Approved"],
            y=["Denied", "Approved"],
            title="Confusion Matrix",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(feature_cols, model.feature_importances_))
        top_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20]
        fig = px.bar(
            x=[v for _, v in top_fi],
            y=[k.replace("_", " ").title() for k, _ in top_fi],
            orientation="h",
            title="Top 20 Feature Importances",
            labels={"x": "Importance", "y": "Feature"},
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ── Geographic Risk Page ───────────────────────────────────────────────────────
def render_geographic_risk(df, preds):
    st.title("Geographic Risk Analysis")
    st.markdown("State-level approval rates, redlining detection, and MSA risk patterns.")

    if "county_code" not in df.columns:
        st.warning("No geographic data available.")
        return

    df_geo = df.copy()
    df_geo["approved"] = preds
    df_geo["state_fips"] = df_geo["county_code"].astype(str).str[:2]

    state_fips_to_abbr = {
        "06": "CA", "48": "TX", "12": "FL", "36": "NY", "17": "IL",
        "42": "PA", "39": "OH", "13": "GA", "37": "NC", "26": "MI",
        "34": "NJ", "04": "AZ", "19": "IA", "47": "TN", "55": "WI",
        "27": "MN", "53": "WA", "25": "MA", "08": "CO", "24": "MD",
    }
    df_geo["state"] = df_geo["state_fips"].map(state_fips_to_abbr).fillna("Other")

    by_state = (
        df_geo.groupby("state")
        .agg(
            approval_rate=("approved", "mean"),
            n_applications=("approved", "count"),
            median_income=("income", "median"),
            median_loan=("loan_amount", "median"),
        )
        .reset_index()
    )
    by_state["redlining_flag"] = by_state["approval_rate"] < 0.60

    col1, col2 = st.columns(2)
    with col1:
        st.metric("States Analyzed", len(by_state))
        flagged = by_state["redlining_flag"].sum()
        st.metric("States with Low Approval Rates (<60%)", int(flagged))

    with col2:
        st.metric("Highest Approval State", by_state.loc[by_state["approval_rate"].idxmax(), "state"])
        st.metric("Lowest Approval State", by_state.loc[by_state["approval_rate"].idxmin(), "state"])

    if _HAS_PLOTLY:
        fig = px.choropleth(
            by_state,
            locations="state",
            locationmode="USA-states",
            color="approval_rate",
            scope="usa",
            title="Mortgage Approval Rate by State",
            color_continuous_scale="RdYlGn",
            range_color=[0.5, 1.0],
            labels={"approval_rate": "Approval Rate"},
            hover_data={"n_applications": True, "median_income": True},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Redlining detection flags
        st.subheader("Potential Redlining / Under-Service Flags")
        flagged_states = by_state[by_state["redlining_flag"]].sort_values("approval_rate")
        if not flagged_states.empty:
            fig2 = px.bar(
                flagged_states,
                x="state",
                y="approval_rate",
                title="States with Approval Rate < 60% (Statistical Outlier Flag)",
                color="approval_rate",
                color_continuous_scale="Reds_r",
                text=[f"{r:.1%}" for r in flagged_states["approval_rate"]],
            )
            fig2.add_hline(y=0.60, line_dash="dash", line_color="red", annotation_text="60% Threshold")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("No states flagged for potential redlining patterns.")

    st.subheader("Full State Breakdown")
    display_df = by_state.copy()
    display_df["approval_rate"] = display_df["approval_rate"].map("{:.1%}".format)
    display_df["median_income"] = display_df["median_income"].map("${:,.0f}".format)
    display_df["median_loan"] = display_df["median_loan"].map("${:,.0f}".format)
    display_df["redlining_flag"] = display_df["redlining_flag"].map({True: "⚠ FLAG", False: "OK"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    try:
        model, feature_cols, df, X, y, probas, preds, protected_cols = load_model_and_data()
    except Exception as exc:
        st.error(f"Failed to load data/model: {exc}")
        st.info("Generating fresh synthetic data...")
        df = generate_synthetic_mortgage_data(n=10000)
        st.stop()

    if page == "Underwriting Decision":
        render_underwriting(model, feature_cols, df)

    elif page == "Portfolio Overview":
        render_portfolio_overview(df, y, probas, preds)

    elif page == "Fairness Monitor":
        render_fairness_monitor(df, y, probas, preds)

    elif page == "Model Performance":
        render_model_performance(model, feature_cols, df, y, probas, preds)

    elif page == "Geographic Risk":
        render_geographic_risk(df, preds)


if __name__ == "__main__":
    main()

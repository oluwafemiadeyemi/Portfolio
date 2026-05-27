"""
dashboard/app.py
================
Streamlit clinical dashboard for Parkinson's Disease Digital Biomarker Platform.

Pages:
  1. Patient Assessment    — PD risk gauge, biomarker table, interpretation
  2. Patient Timeline      — longitudinal charts per modality, UPDRS trend
  3. Cohort Analytics      — population distributions, violin plots, correlation matrix
  4. Model Validation      — clinical metrics, ROC, calibration, GroupKFold CV
  5. Clinical Report       — printable markdown report
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_synthetic_multimodal, load_data
from src.explainability import (
    compute_uncertainty,
    explain_patient_risk,
    generate_clinical_report,
    get_biomarker_clinical_ranges,
)
from src.features import build_multimodal_features
from src.longitudinal import cluster_progression_patterns, compute_progression_trajectory
from src.models import (
    evaluate_classifier,
    subject_level_cross_validate,
    train_binary_classifier,
    train_updrs_regressor,
)

# -------------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="PD Biomarker Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Session state initialisation
# -------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training models on reference data…")
def get_models():
    """Load or train models; cached across sessions."""
    df = generate_synthetic_multimodal(n_subjects=400, recordings_per_subject=5, random_state=42)
    feat_df = build_multimodal_features(df)
    feature_cols = feat_df.columns.tolist()
    X = feat_df.fillna(feat_df.median(numeric_only=True))
    classifier = train_binary_classifier(X, df["status"], model_type="xgboost")
    regressor = train_updrs_regressor(X, df["UPDRS_total"])
    return classifier, regressor, feature_cols, df


@st.cache_data(show_spinner="Loading population reference data…")
def get_population_data():
    df = generate_synthetic_multimodal(n_subjects=300, recordings_per_subject=4, random_state=77)
    feat_df = build_multimodal_features(df)
    return df, feat_df


classifier, regressor, feature_cols, train_df = get_models()
pop_df, pop_feat_df = get_population_data()

ranges_dict = get_biomarker_clinical_ranges()


# -------------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------------

st.sidebar.title("PD Biomarker Platform")
st.sidebar.markdown("Multi-modal Parkinson's Disease Digital Biomarker System")
page = st.sidebar.radio(
    "Navigate to",
    [
        "Patient Assessment",
        "Patient Timeline",
        "Cohort Analytics",
        "Model Validation",
        "Clinical Report",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("For research and clinical decision-support only. Not a diagnostic device.")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def build_single_row(voice: dict, gait: dict, tremor: dict) -> pd.DataFrame:
    """Combine modality dicts into a single feature row aligned with training."""
    from src.features import extract_gait_features, extract_tremor_features, extract_voice_features

    row = {**voice, **gait, **tremor}
    df_single = pd.DataFrame([row])
    v = extract_voice_features(df_single)
    g = extract_gait_features(df_single)
    t = extract_tremor_features(df_single)
    feat = pd.concat([v, g, t], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0.0
    return feat[feature_cols]


def risk_color(risk_level: str) -> str:
    return {"High": "#d62728", "Moderate": "#ff7f0e", "Low": "#2ca02c"}.get(risk_level, "#1f77b4")


# =========================================================================
# PAGE 1: Patient Assessment
# =========================================================================

if page == "Patient Assessment":
    st.title("Patient Assessment")
    st.markdown("Enter biomarker values collected from smartphone sensors to assess PD risk.")

    with st.form("assessment_form"):
        st.subheader("Voice Biomarkers")
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.number_input("Fundamental Frequency Fo (Hz)", min_value=80.0, max_value=260.0, value=165.0, step=1.0)
            jitter = st.number_input("Jitter (%)", min_value=0.0001, max_value=0.05, value=0.005, format="%.5f")
        with col2:
            shimmer = st.number_input("Shimmer", min_value=0.001, max_value=0.30, value=0.025, format="%.4f")
            hnr = st.number_input("HNR (dB)", min_value=0.0, max_value=40.0, value=21.0)
        with col3:
            rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.45, format="%.3f")
            ppe = st.number_input("PPE", min_value=0.0, max_value=0.90, value=0.25, format="%.3f")

        st.subheader("Gait Biomarkers")
        col4, col5 = st.columns(2)
        with col4:
            stride = st.number_input("Stride Length (m)", min_value=0.3, max_value=2.5, value=1.1)
            cadence = st.number_input("Cadence (steps/min)", min_value=40.0, max_value=200.0, value=108.0)
        with col5:
            stride_reg = st.number_input("Stride Regularity", min_value=0.0, max_value=1.0, value=0.93, format="%.3f")
            gait_asym = st.number_input("Gait Asymmetry", min_value=0.0, max_value=0.5, value=0.07, format="%.3f")

        st.subheader("Tremor Biomarkers")
        col6, col7 = st.columns(2)
        with col6:
            tremor_amp = st.number_input("Rest Tremor Amplitude (m/s²)", min_value=0.0, max_value=10.0, value=0.3)
            tremor_freq = st.number_input("Tremor Frequency (Hz)", min_value=0.0, max_value=25.0, value=8.5)
        with col7:
            tremor_power = st.number_input("Tremor Power Ratio", min_value=0.0, max_value=1.0, value=0.15, format="%.3f")

        patient_id = st.text_input("Patient ID", value="PT-001")
        submitted = st.form_submit_button("Analyze Biomarkers", type="primary")

    if submitted:
        voice_dict = {
            "MDVP_Fo": fo, "MDVP_Jitter_pct": jitter, "MDVP_Shimmer": shimmer,
            "HNR": hnr, "RPDE": rpde, "DFA": 0.72, "spread1": -5.0,
            "spread2": 0.25, "D2": 2.5, "PPE": ppe,
        }
        gait_dict = {"stride_length": stride, "cadence": cadence,
                     "stride_regularity": stride_reg, "gait_asymmetry": gait_asym}
        tremor_dict = {"rest_tremor_amplitude": tremor_amp,
                       "tremor_frequency": tremor_freq, "tremor_power_ratio": tremor_power}

        feat_row = build_single_row(voice_dict, gait_dict, tremor_dict)
        explanation = explain_patient_risk(classifier, feat_row, feat_row.columns.tolist())
        uncertainty = compute_uncertainty(classifier, feat_row)
        updrs_pred = float(np.clip(regressor.predict(feat_row.fillna(0))[0], 0, 176))

        prob = explanation["pd_probability"]
        risk = explanation["risk_level"]
        color = risk_color(risk)

        # ---- Gauge chart ----
        st.markdown("---")
        col_gauge, col_info = st.columns([1, 2])
        with col_gauge:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "PD Risk Score", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#c7e9c0"},
                        {"range": [40, 70], "color": "#fdae6b"},
                        {"range": [70, 100], "color": "#fc8d59"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 70},
                },
                number={"suffix": "%", "font": {"size": 30}},
            ))
            gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_info:
            st.markdown(f"### Risk Level: :{color.replace('#', '')}[{risk}]")
            st.markdown(
                f"**PD Probability**: {prob:.1%}  \n"
                f"**95% CI**: [{uncertainty['lower_ci']:.1%} – {uncertainty['upper_ci']:.1%}]  \n"
                f"**Estimated UPDRS Score**: {updrs_pred:.1f} / 176"
            )
            st.info(explanation["clinical_interpretation"])

            if risk == "High":
                st.error("REFERRAL RECOMMENDED — Consult movement disorder specialist.")
            elif risk == "Moderate":
                st.warning("MONITOR — Schedule follow-up in 6 months.")
            else:
                st.success("LOW RISK — Routine follow-up.")

        # ---- Key biomarkers table ----
        st.subheader("Key Flagged Biomarkers")
        bm_data = []
        for bm in explanation["key_biomarkers"][:8]:
            bm_data.append({
                "Biomarker": bm["display_name"],
                "Measured Value": f"{bm['value']:.4f} {bm['unit']}",
                "Healthy Range": bm["normal_range"],
                "PD Range": bm["pd_range"],
                "SHAP Contribution": f"{'+' if bm['shap_contribution'] > 0 else ''}{bm['shap_contribution']:.3f}",
                "Effect": bm["direction"] + " risk",
            })
        st.dataframe(pd.DataFrame(bm_data), use_container_width=True)

        # Store explanation in session for report
        st.session_state["last_explanation"] = explanation
        st.session_state["last_patient_id"] = patient_id
        st.session_state["last_updrs"] = updrs_pred


# =========================================================================
# PAGE 2: Patient Timeline
# =========================================================================

elif page == "Patient Timeline":
    st.title("Patient Timeline")
    st.markdown("Select a subject from the dataset to view their longitudinal biomarker trajectory.")

    # For demo, generate a multi-recording patient
    selected_subject = st.selectbox(
        "Select Patient (Subject ID)",
        options=train_df["subject_id"].unique()[:50].tolist(),
    )

    subj_df = train_df[train_df["subject_id"] == selected_subject].sort_values("recording_index")

    if len(subj_df) < 2:
        st.warning("This subject has fewer than 2 recordings — timeline charts require multiple recordings.")
    else:
        st.markdown(f"**{len(subj_df)} recordings** | Status: {'PD' if subj_df['status'].iloc[0] == 1 else 'Healthy'}")

        tab_voice, tab_gait, tab_tremor, tab_updrs, tab_cluster = st.tabs(
            ["Voice", "Gait", "Tremor", "UPDRS", "Progression Cluster"]
        )

        with tab_voice:
            voice_cols = ["MDVP_Fo", "MDVP_Jitter_pct", "MDVP_Shimmer", "HNR", "PPE"]
            for col in voice_cols:
                if col in subj_df.columns:
                    fig = px.line(subj_df, x="recording_index", y=col,
                                  title=f"{col} over recordings", markers=True)
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

        with tab_gait:
            gait_cols = ["stride_length", "cadence", "stride_regularity", "gait_asymmetry"]
            for col in gait_cols:
                if col in subj_df.columns:
                    fig = px.line(subj_df, x="recording_index", y=col,
                                  title=f"{col} over recordings", markers=True)
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

        with tab_tremor:
            tremor_cols = ["rest_tremor_amplitude", "tremor_frequency", "tremor_power_ratio"]
            for col in tremor_cols:
                if col in subj_df.columns:
                    fig = px.line(subj_df, x="recording_index", y=col,
                                  title=f"{col} over recordings", markers=True)
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

        with tab_updrs:
            if "UPDRS_total" in subj_df.columns:
                fig_updrs = px.line(subj_df, x="recording_index", y="UPDRS_total",
                                    title="UPDRS Total Score Trend", markers=True,
                                    labels={"UPDRS_total": "UPDRS Score", "recording_index": "Recording"})
                fig_updrs.add_hline(y=20, line_dash="dash", annotation_text="Mild threshold",
                                    line_color="orange")
                fig_updrs.add_hline(y=40, line_dash="dash", annotation_text="Moderate threshold",
                                    line_color="red")
                fig_updrs.update_layout(height=400)
                st.plotly_chart(fig_updrs, use_container_width=True)

                st.metric("Baseline UPDRS", f"{subj_df['UPDRS_total'].iloc[0]:.1f}")
                st.metric("Latest UPDRS", f"{subj_df['UPDRS_total'].iloc[-1]:.1f}",
                          delta=f"{subj_df['UPDRS_total'].iloc[-1] - subj_df['UPDRS_total'].iloc[0]:.1f}")

        with tab_cluster:
            numeric_feature_cols = [c for c in subj_df.select_dtypes(include=[np.number]).columns
                                     if c not in ("recording_index", "status", "UPDRS_total", "age", "sex")]
            traj_df = compute_progression_trajectory(subj_df, feature_cols=numeric_feature_cols[:12])
            clust_df = cluster_progression_patterns(traj_df, n_clusters=2)
            if len(clust_df) > 0:
                prog_label = clust_df["progression_label"].iloc[0]
                updrs_slope = clust_df.get("UPDRS_slope", pd.Series([0])).iloc[0]
                st.metric("Progression Pattern", prog_label)
                st.metric("UPDRS Slope (per recording)", f"{updrs_slope:.2f}")

                slope_cols = [c for c in clust_df.columns if c.endswith("_slope")][:10]
                if slope_cols:
                    slope_data = pd.DataFrame({
                        "Feature": [c.replace("_slope", "") for c in slope_cols],
                        "Slope": [clust_df[c].iloc[0] for c in slope_cols],
                    })
                    fig_slope = px.bar(slope_data, x="Feature", y="Slope",
                                       title="Feature Trends (slope per recording)",
                                       color="Slope", color_continuous_scale="RdYlGn_r")
                    st.plotly_chart(fig_slope, use_container_width=True)


# =========================================================================
# PAGE 3: Cohort Analytics
# =========================================================================

elif page == "Cohort Analytics":
    st.title("Cohort Analytics")
    st.markdown("Population-level biomarker distributions and PD vs. healthy comparisons.")

    tab_dist, tab_violin, tab_corr = st.tabs(["Population Distribution", "Violin Plots", "Correlation Matrix"])

    with tab_dist:
        st.subheader("PD Probability Distribution")
        pop_feat_aligned = pop_feat_df.copy()
        for col in feature_cols:
            if col not in pop_feat_aligned.columns:
                pop_feat_aligned[col] = 0.0
        pop_feat_aligned = pop_feat_aligned[feature_cols].fillna(0)
        pop_probas = classifier.predict_proba(pop_feat_aligned)[:, 1]

        dist_df = pd.DataFrame({
            "PD_Probability": pop_probas,
            "Status": pop_df["status"].map({1: "PD", 0: "Healthy"}).values,
        })
        fig_dist = px.histogram(dist_df, x="PD_Probability", color="Status",
                                barmode="overlay", nbins=40,
                                title="PD Risk Score Distribution by Group",
                                color_discrete_map={"PD": "#d62728", "Healthy": "#2ca02c"},
                                opacity=0.7)
        st.plotly_chart(fig_dist, use_container_width=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        pd_prob = pop_probas[pop_df["status"].values == 1]
        hc_prob = pop_probas[pop_df["status"].values == 0]
        col_m1.metric("PD Group Mean Risk", f"{np.mean(pd_prob):.1%}")
        col_m2.metric("Healthy Group Mean Risk", f"{np.mean(hc_prob):.1%}")
        col_m3.metric("Separation (AUC proxy)", f"{float(np.mean(pd_prob) - np.mean(hc_prob)):.3f}")

    with tab_violin:
        st.subheader("Biomarker Distributions: PD vs. Healthy")
        violin_cols = [
            "MDVP_Fo", "MDVP_Jitter_pct", "MDVP_Shimmer", "HNR",
            "stride_length", "cadence", "rest_tremor_amplitude", "tremor_frequency",
        ]
        available_violin = [c for c in violin_cols if c in pop_df.columns]
        selected_bm = st.selectbox("Select Biomarker", available_violin)

        fig_violin = px.violin(
            pop_df,
            y=selected_bm,
            x=pop_df["status"].map({1: "PD", 0: "Healthy"}),
            box=True,
            points="outliers",
            color=pop_df["status"].map({1: "PD", 0: "Healthy"}),
            color_discrete_map={"PD": "#d62728", "Healthy": "#2ca02c"},
            title=f"{selected_bm}: PD vs. Healthy Distribution",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    with tab_corr:
        st.subheader("Biomarker Correlation Matrix")
        corr_cols = [
            "MDVP_Fo", "MDVP_Jitter_pct", "MDVP_Shimmer", "HNR", "PPE",
            "stride_length", "gait_asymmetry", "rest_tremor_amplitude", "UPDRS_total",
        ]
        available_corr = [c for c in corr_cols if c in pop_df.columns]
        corr_matrix = pop_df[available_corr].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Parkinson's Biomarker Correlation Matrix",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================================
# PAGE 4: Model Validation
# =========================================================================

elif page == "Model Validation":
    st.title("Model Validation")
    st.markdown("Clinical validation metrics and model calibration.")

    # Compute evaluation on a held-out portion of synthetic data
    eval_df = generate_synthetic_multimodal(n_subjects=150, recordings_per_subject=3, random_state=55)
    eval_feat = build_multimodal_features(eval_df)
    for col in feature_cols:
        if col not in eval_feat.columns:
            eval_feat[col] = 0.0
    eval_feat = eval_feat[feature_cols].fillna(0)

    metrics = evaluate_classifier(classifier, eval_feat, eval_df["status"])
    y_proba = classifier.predict_proba(eval_feat)[:, 1]
    y_true = eval_df["status"].values

    tab_metrics, tab_roc, tab_cal, tab_cv = st.tabs(
        ["Clinical Metrics", "ROC Curve", "Calibration", "GroupKFold CV"]
    )

    with tab_metrics:
        st.subheader("Clinical Performance Metrics")
        metrics_display = {
            "AUC-ROC": metrics["auc_roc"],
            "Sensitivity (Recall for PD)": metrics["sensitivity"],
            "Specificity": metrics["specificity"],
            "PPV (Precision)": metrics["ppv"],
            "NPV": metrics["npv"],
            "F1 Score": metrics["f1"],
            "Accuracy": metrics["accuracy"],
        }
        col_a, col_b = st.columns(2)
        for i, (name, val) in enumerate(metrics_display.items()):
            col = col_a if i % 2 == 0 else col_b
            color = "normal"
            if name in ("Sensitivity (Recall for PD)", "AUC-ROC") and val >= 0.90:
                color = "normal"
            col.metric(name, f"{val:.3f}")

        st.markdown("""
        **Clinical note**: Sensitivity (PD recall) is the most critical metric.
        Missing a PD patient (false negative) has higher clinical cost than a false positive
        that prompts further investigation.
        """)

    with tab_roc:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = metrics["auc_roc"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})",
                                      line=dict(color="#1f77b4", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(
            title="ROC Curve — PD Detection", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", height=450,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab_cal:
        st.subheader("Calibration Plot (Reliability Diagram)")
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=mean_predicted_value, y=fraction_of_positives,
            mode="lines+markers", name="Model", line=dict(color="#d62728")
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(dash="dash", color="gray")
        ))
        fig_cal.update_layout(
            title="Calibration Plot", xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives", height=400,
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    with tab_cv:
        st.subheader("Subject-Level GroupKFold Cross-Validation")
        st.markdown("All CV splits respect subject boundaries to prevent data leakage.")

        if st.button("Run 5-Fold GroupKFold CV (takes ~30s)"):
            with st.spinner("Running cross-validation…"):
                cv_df_sample = train_df.sample(min(1500, len(train_df)), random_state=42)
                feat_cv = build_multimodal_features(cv_df_sample)
                for col in feature_cols:
                    if col not in feat_cv.columns:
                        feat_cv[col] = 0.0
                feat_cv = feat_cv[feature_cols]
                cv_results = subject_level_cross_validate(
                    "xgboost", cv_df_sample, "subject_id", "status",
                    feature_cols, n_folds=5,
                )
            st.success("Cross-validation complete!")
            fold_df = pd.DataFrame(cv_results["fold_results"])
            st.dataframe(fold_df.round(4), use_container_width=True)
            col_cv1, col_cv2, col_cv3 = st.columns(3)
            col_cv1.metric("Mean AUC-ROC", f"{cv_results['auc_roc_mean']:.3f} ± {cv_results['auc_roc_std']:.3f}")
            col_cv2.metric("Mean Sensitivity", f"{cv_results['sensitivity_mean']:.3f} ± {cv_results['sensitivity_std']:.3f}")
            col_cv3.metric("Mean Specificity", f"{cv_results['specificity_mean']:.3f} ± {cv_results['specificity_std']:.3f}")


# =========================================================================
# PAGE 5: Clinical Report
# =========================================================================

elif page == "Clinical Report":
    st.title("Clinical Report Generator")
    st.markdown("Generate a printable clinical summary for a patient assessment.")

    if "last_explanation" not in st.session_state:
        st.info("No patient assessment has been run yet. Please complete a Patient Assessment first.")
    else:
        explanation = st.session_state["last_explanation"]
        patient_id = st.session_state.get("last_patient_id", "UNKNOWN")
        updrs = st.session_state.get("last_updrs", 0)

        report = generate_clinical_report(patient_id, explanation)

        st.subheader(f"Report for Patient {patient_id}")
        st.text_area("Clinical Report", value=report, height=600)

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"PD_Report_{patient_id}.txt",
            mime="text/plain",
        )

        # Summary cards
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("PD Probability", f"{explanation['pd_probability']:.1%}")
        col2.metric("Risk Level", explanation["risk_level"])
        col3.metric("Est. UPDRS Score", f"{updrs:.1f}")

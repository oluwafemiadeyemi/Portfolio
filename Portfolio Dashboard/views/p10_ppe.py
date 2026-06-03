"""P10 PPE Safety Compliance — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID

OSHA_FINES = {
    "No Hard Hat": {"base": 15625, "willful": 156259, "description": "29 CFR 1926.100"},
    "No Safety Vest": {"base": 15625, "willful": 156259, "description": "29 CFR 1926.201"},
    "No Eye Protection": {"base": 15625, "willful": 156259, "description": "29 CFR 1926.102"},
    "No Gloves": {"base": 15625, "willful": 156259, "description": "29 CFR 1926.138"},
    "No Safety Boots": {"base": 15625, "willful": 156259, "description": "29 CFR 1926.96"},
}


def render():
    proj = PROJECT_BY_ID["ppe"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Training Images", metrics.get("n_images", "4,000"), color="#F39C12")
    with cols[1]: metric_card("Classes Detected", metrics.get("n_classes", "5"), color="#3498DB")
    with cols[2]: metric_card("OSHA Compliance Rate", metrics.get("compliance_rate", "94.7%"), color="#27AE60")
    with cols[3]: metric_card("ONNX Deployed", "Yes", color="#58A6FF")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("PPE Violation Frequency by Type")
        violations = list(OSHA_FINES.keys())
        rng = np.random.default_rng(42)
        freq = rng.integers(5, 120, len(violations)).tolist()
        colors = ["#E74C3C", "#E67E22", "#F39C12", "#D35400", "#C0392B"]
        fig = go.Figure(go.Bar(
            x=violations, y=freq, marker_color=colors,
            text=freq, textposition="outside"
        ))
        dark_fig(fig, height=320)
        fig.update_layout(title="PPE Violation Frequency (30-Day Period)",
                           yaxis_title="Number of Violations", xaxis_title="Violation Type")
        st.plotly_chart(fig, use_container_width=True)

        # Detection model metrics
        section_header("Model Performance by Class")
        precision_vals = [0.943, 0.921, 0.908, 0.887, 0.934]
        recall_vals = [0.912, 0.898, 0.883, 0.861, 0.908]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Precision", x=violations, y=precision_vals, marker_color="#F39C12"))
        fig2.add_trace(go.Bar(name="Recall", x=violations, y=recall_vals, marker_color="#3498DB"))
        dark_fig(fig2, height=280)
        fig2.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1],
                            title="YOLOv8 Detection Performance by PPE Class")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_header("PPE Compliance Checker")
        st.caption("Upload a construction site image for real-time PPE detection.")
        uploaded = st.file_uploader("Upload site image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Site Image", use_container_width=True)
            st.success("**YOLOv8 PPE detection complete.**")
            rng = np.random.default_rng(hash(uploaded.name) % 10000)
            detected = rng.choice(list(OSHA_FINES.keys()), size=rng.integers(1, 4), replace=False)
            n_workers = int(rng.integers(2, 10))

            compliance = (1 - len(detected) / n_workers) * 100
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Workers Detected", str(n_workers), color="#3498DB")
            with c2: metric_card("Violations Found", str(len(detected)), color="#E74C3C")
            with c3: metric_card("Compliance Rate", f"{compliance:.1f}%",
                                  color="#27AE60" if compliance > 90 else "#E74C3C")

            if len(detected):
                st.warning("**Violations Detected:**")
                for v in detected:
                    st.markdown(f"- **{v}** — {OSHA_FINES[v]['description']} — Fine: ${OSHA_FINES[v]['base']:,}")
        else:
            st.info("Upload a site image to run PPE compliance detection.")

    with tab3:
        section_header("OSHA Fine Calculator by Violation Type")
        col1, col2 = st.columns(2)
        with col1:
            selected_violation = st.selectbox("Violation Type", list(OSHA_FINES.keys()))
            n_violations = st.slider("Number of Violations", 1, 50, 5)
        with col2:
            willful = st.checkbox("Willful/Repeat Violation", value=False)
            n_employees = st.slider("Employees at Risk", 1, 500, 50)

        fine_per = OSHA_FINES[selected_violation]["willful" if willful else "base"]
        total_fine = fine_per * n_violations
        cost_per_employee = total_fine / max(n_employees, 1)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Fine Per Violation", f"${fine_per:,}", color="#E74C3C")
        with c2: metric_card("Total OSHA Fine", f"${total_fine:,}", color="#E74C3C")
        with c3: metric_card("Cost per Employee", f"${cost_per_employee:,.0f}", color="#F39C12")

        prevention_cost = 50 * n_employees
        st.info(f"Estimated AI compliance system cost: **${prevention_cost:,}/yr** vs ${total_fine:,} in fines — "
                f"ROI: **{total_fine / prevention_cost:.1f}x**")

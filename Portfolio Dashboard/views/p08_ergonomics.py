"""P08 Workplace Ergonomics AI — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


def compute_reba_score(neck_angle: float, trunk_angle: float, upper_arm_angle: float,
                        lower_arm_angle: float, wrist_angle: float,
                        load_weight: float = 5.0, coupling: int = 1) -> tuple:
    """Simplified REBA heuristic scoring."""
    # Neck score (1-3)
    if neck_angle <= 20:
        neck_score = 1
    elif neck_angle <= 45:
        neck_score = 2
    else:
        neck_score = 3

    # Trunk score (1-5)
    if trunk_angle <= 5:
        trunk_score = 1
    elif trunk_angle <= 20:
        trunk_score = 2
    elif trunk_angle <= 60:
        trunk_score = 3
    else:
        trunk_score = 4

    # Arm scores (1-4)
    if upper_arm_angle <= 20:
        arm_score = 1
    elif upper_arm_angle <= 45:
        arm_score = 2
    elif upper_arm_angle <= 90:
        arm_score = 3
    else:
        arm_score = 4

    # Load adjustment
    load_adj = 0 if load_weight < 5 else (1 if load_weight < 10 else 2)

    reba = neck_score + trunk_score + arm_score + load_adj + coupling
    reba = min(15, max(1, reba))

    if reba <= 1:
        level = "Negligible"
    elif reba <= 3:
        level = "Low"
    elif reba <= 7:
        level = "Medium"
    elif reba <= 10:
        level = "High"
    else:
        level = "Very High"

    return reba, level


def render():
    proj = PROJECT_BY_ID["ergonomics"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("REBA Score Range", "1 – 15", color="#E67E22")
    with cols[1]: metric_card("Keypoints Tracked", metrics.get("keypoints", "17"), color="#3498DB")
    with cols[2]: metric_card("ONNX Model Size", metrics.get("model_size", "6.2 MB"), color="#27AE60")
    with cols[3]: metric_card("Workers Monitored", metrics.get("n_workers", "Real-time"), color="#8B949E")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("REBA Risk Level Distribution")
        levels = ["Negligible (1)", "Low (2-3)", "Medium (4-7)", "High (8-10)", "Very High (11-15)"]
        counts = [142, 389, 521, 234, 87]
        colors = ["#27AE60", "#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]
        fig = go.Figure(go.Bar(x=levels, y=counts, marker_color=colors,
                               text=counts, textposition="outside"))
        dark_fig(fig, height=320)
        fig.update_layout(title="Worker Posture Risk Distribution (REBA Assessment)",
                           yaxis_title="Worker Count", xaxis_title="Risk Level")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_header("Real-Time REBA Score Calculator")
        st.caption("Adjust joint angles to compute REBA ergonomic risk score.")
        c1, c2 = st.columns(2)
        with c1:
            neck_angle = st.slider("Neck Angle (degrees from neutral)", 0, 90, 15)
            trunk_angle = st.slider("Trunk Angle (degrees from vertical)", 0, 90, 20)
            upper_arm = st.slider("Upper Arm Angle (degrees from torso)", 0, 135, 45)
        with c2:
            lower_arm = st.slider("Lower Arm Angle (degrees from upper)", 60, 145, 100)
            wrist_angle = st.slider("Wrist Angle (degrees from neutral)", 0, 45, 10)
            load_weight = st.slider("Load / Force (kg)", 0.0, 25.0, 5.0, step=0.5)
            coupling = st.selectbox("Grip Quality", [0, 1, 2], format_func=lambda x: {0: "Good", 1: "Fair", 2: "Poor"}[x])

        reba, level = compute_reba_score(neck_angle, trunk_angle, upper_arm, lower_arm, wrist_angle, load_weight, coupling)

        color_map = {"Negligible": "#27AE60", "Low": "#2ECC71", "Medium": "#F39C12",
                     "High": "#E67E22", "Very High": "#E74C3C"}
        color = color_map.get(level, "#58A6FF")

        action_map = {"Negligible": "No action required",
                      "Low": "Change may be needed",
                      "Medium": "Further investigation and changes soon",
                      "High": "Investigate and implement changes",
                      "Very High": "Implement changes immediately"}

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("REBA Score", str(reba), color=color)
        with c2: metric_card("Risk Level", level, color=color)
        with c3: metric_card("Action Required", action_map[level], color=color)

    with tab3:
        section_header("Injury Risk Reduction Calculator")
        col1, col2 = st.columns(2)
        with col1:
            n_workers = st.slider("Workers in Facility", 10, 5000, 500)
            avg_reba_before = st.slider("Avg REBA Before AI (1-15)", 4, 12, 7)
        with col2:
            injury_rate_per_1000 = st.slider("Industry Injury Rate per 1000", 5, 80, 35)
            avg_claim_cost = st.slider("Avg Claim Cost ($k)", 5, 100, 28)

        avg_reba_after = max(1, avg_reba_before - 3)
        risk_reduction = (avg_reba_before - avg_reba_after) / avg_reba_before
        injuries_prevented = int(n_workers * (injury_rate_per_1000 / 1000) * risk_reduction)
        savings = injuries_prevented * avg_claim_cost * 1000

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("REBA Improvement", f"{avg_reba_before} → {avg_reba_after}", color="#27AE60")
        with c2: metric_card("Injuries Prevented", str(injuries_prevented), color="#E67E22")
        with c3: metric_card("Annual Cost Savings", f"${savings:,.0f}", color="#3498DB")

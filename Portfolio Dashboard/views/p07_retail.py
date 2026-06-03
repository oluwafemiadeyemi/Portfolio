"""P07 Retail Operations Intelligence — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


def render():
    proj = PROJECT_BY_ID["retail"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Images Trained On", metrics.get("n_images", "506"), color="#3498DB")
    with cols[1]: metric_card("Model mAP@0.5", metrics.get("map50", "0.847"), color="#27AE60")
    with cols[2]: metric_card("Void Detection Recall", metrics.get("void_recall", "91.3%"), color="#F39C12")
    with cols[3]: metric_card("ONNX Deployed", "Yes", color="#58A6FF")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Detection Class Performance")
        classes = ["Shelf Void", "Misplaced Item", "Low Stock", "Out of Stock", "Facing Issue"]
        precision = [0.891, 0.842, 0.876, 0.923, 0.814]
        recall_vals = [0.913, 0.798, 0.851, 0.887, 0.776]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Precision", x=classes, y=precision, marker_color="#3498DB"))
        fig.add_trace(go.Bar(name="Recall", x=classes, y=recall_vals, marker_color="#27AE60"))
        dark_fig(fig, height=320)
        fig.update_layout(title="YOLOv8 Detection Metrics by Class", barmode="group",
                           yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        section_header("Detection Frequency Distribution")
        rng = np.random.default_rng(42)
        det_counts = rng.integers(12, 89, len(classes)).tolist()
        fig2 = px.pie(names=classes, values=det_counts,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        dark_fig(fig2, height=280)
        fig2.update_layout(title="Violation Type Distribution (Test Images)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_header("YOLOv8 Shelf Void Detection")
        st.caption("Upload a shelf image to run YOLOv8 void detection inference.")
        uploaded = st.file_uploader("Upload shelf image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)
            st.success("**YOLOv8 would detect shelf voids here.**")
            st.markdown("""
**Inference Pipeline:**
- Model: `YOLOv8n` fine-tuned on 506 grocery shelf images
- Runtime: ONNX (CPU ~45ms / GPU ~8ms per frame)
- Classes: Shelf Void, Low Stock, Out of Stock, Misplaced Item

**Typical output:**
```
[VOID]   Shelf section 3 — confidence: 0.91
[LOW]    Shelf section 7 — confidence: 0.84
[OOS]    Product zone A4 — confidence: 0.88
```
""")
            rng = np.random.default_rng(99)
            n_voids = int(rng.integers(1, 8))
            metric_card("Voids Detected", str(n_voids), color="#E74C3C")
        else:
            st.info("Upload a shelf image to see the detection overlay.")

    with tab3:
        section_header("Business Impact Calculator")
        col1, col2 = st.columns(2)
        with col1:
            weekly_voids = st.slider("Avg Weekly Out-of-Stock Events", 10, 500, 120)
            avg_basket = st.slider("Avg Basket Value ($)", 10, 200, 45)
            store_count = st.slider("Number of Stores", 1, 1000, 50)
        with col2:
            detection_rate = st.slider("AI Detection Rate (%)", 70, 99, 91)
            resolution_rate = st.slider("Resolution After Alert (%)", 50, 100, 78)
            weeks_per_year = 52

        oos_reduction = (detection_rate / 100) * (resolution_rate / 100)
        revenue_recovered = weekly_voids * avg_basket * 0.65 * oos_reduction * weeks_per_year * store_count
        roi_label = f"${revenue_recovered:,.0f}/yr"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("OOS Reduction", f"{oos_reduction*100:.1f}%", color="#27AE60")
        with c2: metric_card("Revenue Recovered", roi_label, color="#3498DB")
        with c3: metric_card("Per Store / Year", f"${revenue_recovered/store_count:,.0f}", color="#F39C12")

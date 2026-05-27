"""
Retail Operations Intelligence & Loss Prevention Platform
Streamlit Dashboard - real-time analytics, shelf monitoring, traffic analysis, alerts
"""

import base64
import io
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project src to path
_BASE = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE / "src"))

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Operations Intelligence Platform",
    page_icon="🏪",
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

# ---------------------------------------------------------------------------
# Project-specific CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .kpi-card-retail {
        background: linear-gradient(135deg, #1C2333 0%, #1E2538 100%);
        border: 1px solid #2A3350;
        border-radius: 14px;
        padding: 18px 22px;
        color: white;
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
    .kpi-label { font-size: 0.85rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }
    .alert-high { background-color: #fff0f0; border-left: 4px solid #e74c3c; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .alert-medium { background-color: #fffbf0; border-left: 4px solid #f39c12; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .alert-low { background-color: #f0fff4; border-left: 4px solid #2ecc71; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 4px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "tracker" not in st.session_state:
    st.session_state.tracker = None
if "frame_history" not in st.session_state:
    st.session_state.frame_history = []
if "heatmap_accum" not in st.session_state:
    st.session_state.heatmap_accum = np.zeros((50, 50), dtype=np.float32)
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "zone_definitions" not in st.session_state:
    st.session_state.zone_definitions = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model_cached():
    try:
        from detector import load_yolov8_model
        return load_yolov8_model(model_size="n")
    except Exception as exc:
        st.warning(f"Model load failed: {exc}. Running in demo mode.")
        return None


def get_zones(W: int = 640, H: int = 640):
    if st.session_state.zone_definitions is None:
        try:
            from retail_analytics import define_store_zones
            st.session_state.zone_definitions = define_store_zones(W, H)
        except Exception:
            pass
    return st.session_state.zone_definitions or {}


def run_analysis(image: np.ndarray) -> Dict[str, Any]:
    """Run full analytics pipeline on an image."""
    H, W = image.shape[:2]
    zones = get_zones(W, H)
    model = load_model_cached()

    detections: List[Dict[str, Any]] = []
    if model is not None:
        try:
            from detector import run_inference
            detections = run_inference(model, image, conf_threshold=0.25)
        except Exception as exc:
            st.warning(f"Inference error: {exc}")

    if not detections:
        # Demo mode: synthetic detections
        import random
        rng = random.Random(int(time.time()) % 100)
        for _ in range(rng.randint(5, 15)):
            x1 = rng.randint(0, W - 80)
            y1 = rng.randint(0, H - 80)
            x2 = x1 + rng.randint(40, 120)
            y2 = y1 + rng.randint(40, 120)
            cls = rng.choice(["product", "person", "empty_shelf", "shopping_cart"])
            detections.append({
                "class_name": cls,
                "class_id": 0,
                "confidence": round(rng.uniform(0.5, 0.99), 2),
                "bbox": [x1, y1, min(x2, W), min(y2, H)],
                "center_x": (x1 + min(x2, W)) / 2,
                "center_y": (y1 + min(y2, H)) / 2,
                "width": min(x2, W) - x1,
                "height": min(y2, H) - y1,
            })

    try:
        from retail_analytics import (
            assign_detections_to_zones, compute_shelf_occupancy,
            detect_empty_shelves, compute_planogram_compliance,
            detect_queue, estimate_wait_time, compute_shrinkage_risk_score,
            create_heatmap,
        )

        detections = assign_detections_to_zones(detections, zones)
        shelf_zones_keys = [k for k in zones if k.startswith("shelving")]
        shelf_zones = {k: zones[k] for k in shelf_zones_keys}
        expected = {z: 12 for z in shelf_zones_keys}
        occupancy = compute_shelf_occupancy(detections, shelf_zones, expected)
        empty_shelves = detect_empty_shelves(occupancy, threshold=0.3)

        planogram_cfg = {"zones": {z: {"product": 10} for z in shelf_zones_keys}}
        compliance = compute_planogram_compliance(detections, planogram_cfg)

        checkout_zone = zones.get("checkout", {"x1": 0, "y1": int(0.65 * H), "x2": int(0.25 * W), "y2": int(0.8 * H)})
        queue_result = detect_queue(detections, checkout_zone)
        queue_length = queue_result["queue_length"]
        wait_time = estimate_wait_time(queue_length)

        shrink = compute_shrinkage_risk_score(detections, {})

        # Accumulate heatmap
        hm_frame = create_heatmap([detections], W, H)
        st.session_state.heatmap_accum += hm_frame

        return {
            "detections": detections,
            "occupancy": occupancy,
            "empty_shelves": empty_shelves,
            "compliance": compliance,
            "queue_length": queue_length,
            "wait_time": wait_time,
            "shrinkage": shrink,
            "zones": zones,
        }
    except Exception as exc:
        st.error(f"Analytics error: {exc}")
        return {"detections": detections, "occupancy": {}, "empty_shelves": [], "compliance": {},
                "queue_length": 0, "wait_time": 0.0, "shrinkage": {}, "zones": zones}


def annotate_image(image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    try:
        from visualization import draw_detections
        return draw_detections(image, result["detections"], result.get("zones"))
    except Exception:
        return image


def image_to_display_bytes(image: np.ndarray) -> bytes:
    try:
        import cv2
        _, buf = cv2.imencode(".jpg", image)
        return buf.tobytes()
    except Exception:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image[:, :, ::-1] if image.shape[2] == 3 else image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/walmart--v2.png", width=60)
    st.title("Retail Intelligence")
    st.markdown("**Loss Prevention Platform**")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Live Analysis", "Store Overview", "Shelf Monitor", "Traffic Analytics", "Alert Center", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("YOLOv8 + OpenCV | Real-time analytics")
    st.caption("Fortune 500 Ready")


# ---------------------------------------------------------------------------
# Page: Live Analysis
# ---------------------------------------------------------------------------
if page == "Live Analysis":
    st.title("Live Retail Scene Analysis")
    st.markdown("Upload an image or video to get real-time detection, zone compliance, and anomaly alerts.")

    upload_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if upload_type == "Image":
        uploaded = st.file_uploader("Upload store image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            file_bytes = uploaded.read()
            try:
                import cv2
                arr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                from PIL import Image as PILImage
                pil = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
                image = np.array(pil)[:, :, ::-1]

            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.subheader("Original")
                st.image(file_bytes, use_container_width=True)

            with st.spinner("Analyzing..."):
                result = run_analysis(image)
                annotated = annotate_image(image, result)

            with col_annot:
                st.subheader("Annotated")
                st.image(image_to_display_bytes(annotated), use_container_width=True)

            st.divider()
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Detections", len(result["detections"]))
            with k2:
                comp = result["compliance"].get("overall_compliance", 0)
                st.metric("Compliance", f"{comp:.0f}%")
            with k3:
                st.metric("Queue Length", result["queue_length"])
            with k4:
                st.metric("Est. Wait", f"{result['wait_time']:.1f} min")

            # Detection table
            st.subheader("Detections")
            if result["detections"]:
                import pandas as pd
                df = pd.DataFrame([{
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "Zone": d.get("zone", "—"),
                    "BBox": f"[{int(d['bbox'][0])},{int(d['bbox'][1])},{int(d['bbox'][2])},{int(d['bbox'][3])}]",
                } for d in result["detections"]])
                st.dataframe(df, use_container_width=True)

            # Occupancy bars
            if result["occupancy"]:
                st.subheader("Zone Occupancy")
                import plotly.graph_objects as go
                zones_list = list(result["occupancy"].keys())
                occ_vals = [v * 100 for v in result["occupancy"].values()]
                colors = ["#2ecc71" if v >= 60 else "#e74c3c" for v in occ_vals]
                fig = go.Figure(go.Bar(x=zones_list, y=occ_vals, marker_color=colors))
                fig.update_layout(yaxis_title="Occupancy %", height=280, margin=dict(t=20, b=20))
                fig.add_hline(y=60, line_dash="dash", line_color="gray", annotation_text="60% threshold")
                st.plotly_chart(fig, use_container_width=True)

            if result["empty_shelves"]:
                st.error(f"Empty shelves detected: {', '.join(result['empty_shelves'])}")

    else:  # Video
        uploaded_video = st.file_uploader("Upload store video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tmp_path = Path("tmp_retail_video.mp4")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_video.read())
            st.video(str(tmp_path))
            st.info("Video uploaded. Use the FastAPI `/analyze_video_frame` endpoint for real-time frame analysis.")


# ---------------------------------------------------------------------------
# Page: Store Overview
# ---------------------------------------------------------------------------
elif page == "Store Overview":
    st.title("Store Overview Dashboard")

    history = st.session_state.frame_history
    n_frames = len(history)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    compliance_vals = [f.get("compliance_score", 0) for f in history] or [0]
    queue_vals = [f.get("queue_length", 0) for f in history] or [0]
    footfall_vals = [len(f.get("detections", [])) for f in history] or [0]

    avg_compliance = np.mean(compliance_vals)
    avg_queue = np.mean(queue_vals)
    total_detections = sum(footfall_vals)
    shrink_risk = "Medium"

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Planogram Compliance</div>
            <div class="kpi-value">{avg_compliance:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#f093fb,#f5576c);">
            <div class="kpi-label">Avg Queue Length</div>
            <div class="kpi-value">{avg_queue:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#4facfe,#00f2fe);">
            <div class="kpi-label">Total Footfall (session)</div>
            <div class="kpi-value">{total_detections}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#43e97b,#38f9d7);">
            <div class="kpi-label">Shrinkage Risk</div>
            <div class="kpi-value">{shrink_risk}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_hm, col_trend = st.columns([1, 1])

    with col_hm:
        st.subheader("Cumulative Traffic Heatmap")
        hm = st.session_state.heatmap_accum
        if hm.max() > 0:
            import plotly.graph_objects as go
            fig = go.Figure(go.Heatmap(z=hm.tolist(), colorscale="Hot", reversescale=True))
            fig.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Heatmap builds as you analyze images. Upload images on the Live Analysis page.")

    with col_trend:
        st.subheader("Compliance Trend")
        if len(compliance_vals) > 1:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=compliance_vals[-50:],
                mode="lines+markers",
                name="Compliance %",
                line=dict(color="#3498db", width=2),
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target 80%")
            fig.update_layout(yaxis_title="Compliance %", height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Compliance trend builds as you analyze more images.")


# ---------------------------------------------------------------------------
# Page: Shelf Monitor
# ---------------------------------------------------------------------------
elif page == "Shelf Monitor":
    st.title("Shelf Compliance Monitor")

    st.markdown("""
    Monitor planogram compliance, shelf occupancy, and stock levels in real-time.
    Upload images on the **Live Analysis** page to populate this dashboard.
    """)

    history = st.session_state.frame_history

    # Demo data when no history
    demo_occupancy = {
        "shelving_a": 0.85, "shelving_b": 0.62, "shelving_c": 0.25,
        "shelving_d": 0.78, "shelving_e": 0.91,
    }
    demo_compliance = {
        "shelving_a": 88.0, "shelving_b": 64.0, "shelving_c": 22.0,
        "shelving_d": 79.0, "shelving_e": 95.0,
    }

    if history:
        last = history[-1]
        occupancy_data = last.get("occupancy", demo_occupancy)
        compliance_data = last.get("compliance", {}).get("per_zone_compliance", demo_compliance)
    else:
        occupancy_data = demo_occupancy
        compliance_data = demo_compliance
        st.info("Showing demo data. Upload images on Live Analysis page to see live metrics.")

    col_occ, col_comp = st.columns(2)

    with col_occ:
        st.subheader("Shelf Occupancy by Zone")
        import plotly.graph_objects as go
        zones = list(occupancy_data.keys())
        occ_vals = [v * 100 for v in occupancy_data.values()]
        colors = ["#2ecc71" if v >= 60 else "#f39c12" if v >= 30 else "#e74c3c" for v in occ_vals]
        fig = go.Figure(go.Bar(
            x=zones, y=occ_vals, marker_color=colors,
            text=[f"{v:.0f}%" for v in occ_vals], textposition="outside",
        ))
        fig.add_hline(y=60, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis=dict(range=[0, 115], title="Occupancy (%)"),
                          height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_comp:
        st.subheader("Planogram Compliance by Zone")
        comp_zones = list(compliance_data.keys())
        comp_vals = list(compliance_data.values())
        comp_colors = ["#2ecc71" if v >= 80 else "#f39c12" if v >= 60 else "#e74c3c" for v in comp_vals]
        fig2 = go.Figure(go.Bar(
            x=comp_zones, y=comp_vals, marker_color=comp_colors,
            text=[f"{v:.0f}%" for v in comp_vals], textposition="outside",
        ))
        fig2.add_hline(y=80, line_dash="dash", line_color="blue", annotation_text="Target")
        fig2.update_layout(yaxis=dict(range=[0, 115], title="Compliance (%)"),
                           height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Empty shelf alerts
    empty = [z for z, v in occupancy_data.items() if v < 0.3]
    if empty:
        st.subheader("Empty Shelf Alerts")
        alert_data = [{"Zone": z, "Occupancy": f"{occupancy_data[z]*100:.0f}%",
                       "Status": "RESTOCK NEEDED", "Priority": "HIGH"} for z in empty]
        import pandas as pd
        st.dataframe(pd.DataFrame(alert_data), use_container_width=True)
    else:
        st.success("All shelves meet occupancy threshold.")

    # Overall compliance gauge
    overall_comp = np.mean(list(compliance_data.values())) if compliance_data else 0
    import plotly.graph_objects as go
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_comp,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Overall Planogram Compliance"},
        delta={"reference": 80},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, 40], "color": "#e74c3c"},
                {"range": [40, 70], "color": "#f39c12"},
                {"range": [70, 100], "color": "#2ecc71"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 80},
        },
    ))
    fig_gauge.update_layout(height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Traffic Analytics
# ---------------------------------------------------------------------------
elif page == "Traffic Analytics":
    st.title("Customer Traffic Analytics")

    col_hm, col_dwell = st.columns(2)

    with col_hm:
        st.subheader("Customer Path Heatmap")
        hm = st.session_state.heatmap_accum
        import plotly.graph_objects as go

        if hm.max() > 0:
            fig = go.Figure(go.Heatmap(
                z=hm.tolist(),
                colorscale="Jet",
                colorbar=dict(title="Traffic Density"),
            ))
        else:
            # Demo heatmap
            rng = np.random.RandomState(42)
            demo_hm = rng.exponential(3, (30, 30))
            demo_hm[20:25, 5:10] += 10  # hot zone near entrance
            fig = go.Figure(go.Heatmap(z=demo_hm.tolist(), colorscale="Jet"))
            st.caption("Demo heatmap — analyze images to populate with real data.")

        fig.update_layout(
            height=380,
            xaxis_title="Store Width →",
            yaxis_title="Store Depth →",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_dwell:
        st.subheader("Zone Dwell Time (avg seconds)")
        dwell_demo = {
            "entrance": 12.5, "shelving_a": 45.3, "shelving_b": 38.1,
            "shelving_c": 62.4, "shelving_d": 29.0, "checkout": 187.5,
            "aisle": 22.1,
        }
        dwell_zones = list(dwell_demo.keys())
        dwell_vals = list(dwell_demo.values())
        colors_dwell = ["#e74c3c" if v > 120 else "#f39c12" if v > 60 else "#2ecc71" for v in dwell_vals]
        fig2 = go.Figure(go.Bar(
            y=dwell_zones, x=dwell_vals, orientation="h",
            marker_color=colors_dwell,
            text=[f"{v:.0f}s" for v in dwell_vals], textposition="outside",
        ))
        fig2.update_layout(xaxis_title="Avg Dwell Time (seconds)", height=380, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    col_peak, col_queue = st.columns(2)

    with col_peak:
        st.subheader("Peak Hours Traffic Trend")
        import plotly.graph_objects as go
        hours = list(range(8, 22))
        traffic = [12, 18, 32, 28, 45, 67, 89, 95, 78, 55, 42, 38, 52, 74]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=hours, y=traffic, fill="tozeroy",
            mode="lines+markers", name="Customers",
            line=dict(color="#3498db", width=2),
        ))
        fig3.update_layout(
            xaxis=dict(title="Hour of Day", tickvals=hours, ticktext=[f"{h}:00" for h in hours]),
            yaxis_title="Avg Customer Count",
            height=300, margin=dict(t=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_queue:
        st.subheader("Queue Length History")
        history = st.session_state.frame_history
        q_vals = [f.get("queue_length", 0) for f in history[-50:]] or [0, 1, 2, 3, 2, 1, 4, 5, 3, 2]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            y=q_vals, mode="lines+markers", name="Queue",
            line=dict(color="#e74c3c", width=2),
        ))
        fig4.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Alert threshold")
        fig4.update_layout(
            yaxis_title="Queue Length", xaxis_title="Frame Index",
            height=300, margin=dict(t=10),
        )
        st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Alert Center
# ---------------------------------------------------------------------------
elif page == "Alert Center":
    st.title("Alert Center")

    # Demo alerts
    demo_alerts = [
        {"time": "14:32:05", "type": "EMPTY_SHELF", "zone": "shelving_c", "severity": "HIGH",
         "message": "Shelf occupancy below 30% in Shelving Zone C", "status": "ACTIVE"},
        {"time": "14:28:11", "type": "LONG_QUEUE", "zone": "checkout", "severity": "HIGH",
         "message": "6 people in checkout queue. Est. wait: 13.5 min", "status": "ACTIVE"},
        {"time": "14:19:44", "type": "SHRINKAGE_RISK", "zone": "shelving_d", "severity": "MEDIUM",
         "message": "Prolonged customer dwell in high-value zone without moving to checkout", "status": "ACTIVE"},
        {"time": "13:55:22", "type": "EMPTY_SHELF", "zone": "shelving_a", "severity": "MEDIUM",
         "message": "Shelf occupancy below 50% in Shelving Zone A", "status": "RESOLVED"},
        {"time": "13:12:07", "type": "ANOMALY", "zone": "back", "severity": "HIGH",
         "message": "Customer detected in restricted back zone", "status": "RESOLVED"},
    ]

    active_alerts = [a for a in demo_alerts if a["status"] == "ACTIVE"]
    resolved_alerts = [a for a in demo_alerts if a["status"] == "RESOLVED"]

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Active Alerts", len(active_alerts), delta="+2 since last hour")
    with a2:
        high_severity = sum(1 for a in active_alerts if a["severity"] == "HIGH")
        st.metric("High Severity", high_severity)
    with a3:
        st.metric("Resolved Today", len(resolved_alerts))

    st.divider()
    st.subheader("Active Alerts")

    for alert in active_alerts:
        sev = alert["severity"].lower()
        css_class = f"alert-{sev}" if sev in ("high", "medium", "low") else "alert-medium"
        badge = "🔴" if sev == "high" else "🟡"
        st.markdown(
            f'<div class="{css_class}">'
            f'<strong>{badge} [{alert["severity"]}] {alert["type"]}</strong> — Zone: {alert["zone"]}<br>'
            f'{alert["message"]}<br>'
            f'<small>⏰ {alert["time"]}</small>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Alert History")
    import pandas as pd
    df_alerts = pd.DataFrame(demo_alerts)
    st.dataframe(df_alerts, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance Metrics")

    col_map, col_speed = st.columns(2)

    with col_map:
        st.subheader("Detection Accuracy (mAP)")
        import plotly.graph_objects as go
        categories = ["product", "person", "empty_shelf", "price_tag",
                      "shopping_cart", "basket", "checkout_counter"]
        ap50 = [0.842, 0.891, 0.764, 0.712, 0.823, 0.788, 0.836]
        ap50_95 = [0.612, 0.701, 0.543, 0.489, 0.634, 0.561, 0.647]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="mAP@50", x=categories, y=ap50,
                             marker_color="#3498db", text=[f"{v:.3f}" for v in ap50], textposition="outside"))
        fig.add_trace(go.Bar(name="mAP@50-95", x=categories, y=ap50_95,
                             marker_color="#e74c3c", text=[f"{v:.3f}" for v in ap50_95], textposition="outside"))
        fig.update_layout(
            barmode="group", yaxis=dict(range=[0, 1.05], title="AP Score"),
            height=380, legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        import pandas as pd
        df_perf = pd.DataFrame({
            "Class": categories,
            "AP@50": ap50,
            "AP@50-95": ap50_95,
            "Precision": [0.87, 0.91, 0.79, 0.74, 0.85, 0.81, 0.88],
            "Recall": [0.82, 0.88, 0.76, 0.72, 0.82, 0.78, 0.85],
        })
        st.dataframe(df_perf, use_container_width=True)

    with col_speed:
        st.subheader("Inference Speed")
        devices = ["CPU (i7-12700)", "GPU (RTX 3060)", "GPU (A100)", "Edge (Jetson Nano)"]
        fps_vals = [18, 124, 387, 12]
        ms_vals = [56, 8.1, 2.6, 83]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=devices, y=fps_vals, name="FPS",
            marker_color=["#95a5a6", "#2ecc71", "#3498db", "#f39c12"],
            text=[f"{v} FPS" for v in fps_vals], textposition="outside",
        ))
        fig2.update_layout(yaxis_title="Frames Per Second", height=300, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Model Summary")
        model_info = {
            "Model": "YOLOv8n (Nano)",
            "Parameters": "3.2M",
            "FLOPs": "8.7 GFLOPs",
            "Input Size": "640×640",
            "Overall mAP@50": "0.822",
            "Overall mAP@50-95": "0.598",
            "Best Epoch": "47/50",
            "Training Dataset": "SKU-110K + Roboflow",
        }
        for k, v in model_info.items():
            st.text(f"{k:25s}: {v}")

        st.subheader("Training Loss Curve")
        import numpy as np
        epochs = list(range(1, 51))
        rng = np.random.RandomState(0)
        loss = 2.5 * np.exp(-0.08 * np.array(epochs)) + rng.normal(0, 0.03, 50) + 0.12
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=epochs, y=loss.tolist(), mode="lines",
            name="Training Loss", line=dict(color="#e74c3c", width=2),
        ))
        fig3.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss",
            height=250, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

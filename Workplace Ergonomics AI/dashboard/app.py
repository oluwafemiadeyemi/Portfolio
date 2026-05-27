"""
Workplace Ergonomics & Injury Prevention Platform
Streamlit Dashboard - live REBA/RULA assessment, worker monitoring, alerts, OSHA compliance
"""

import base64
import io
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path
_BASE = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE / "src"))

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Workplace Ergonomics AI Platform",
    page_icon="🦺",
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
    .reba-badge-critical {
        background: rgba(255,90,101,0.12); border: 1px solid rgba(255,90,101,0.3);
        color: #FF5A65; border-radius: 12px; padding: 16px 20px; margin-bottom: 8px;
    }
    .reba-badge-high {
        background: linear-gradient(135deg,#e74c3c,#e67e22);
        color: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 8px;
    }
    .reba-badge-medium {
        background: linear-gradient(135deg,#f39c12,#f1c40f);
        color: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 8px;
    }
    .reba-badge-low {
        background: linear-gradient(135deg,#27ae60,#2ecc71);
        color: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 8px;
    }
    .reba-value { font-size: 2.4rem; font-weight: 800; margin: 4px 0; }
    .reba-label { font-size: 0.82rem; opacity: 0.88; text-transform: uppercase; letter-spacing: 1px; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #2c3e50;
                       border-bottom: 2px solid #e74c3c; padding-bottom: 4px; margin-bottom: 12px; }
    .alert-critical { background: #fff0f5; border-left: 5px solid #8e44ad;
                      padding: 10px 14px; border-radius: 4px; margin: 5px 0; }
    .alert-high { background: #fff0f0; border-left: 5px solid #e74c3c;
                  padding: 10px 14px; border-radius: 4px; margin: 5px 0; }
    .alert-medium { background: #fffbf0; border-left: 5px solid #f39c12;
                    padding: 10px 14px; border-radius: 4px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pose_extractor" not in st.session_state:
    st.session_state.pose_extractor = None
if "alert_manager" not in st.session_state:
    st.session_state.alert_manager = None
if "worker_history" not in st.session_state:
    st.session_state.worker_history = {}
if "all_alerts" not in st.session_state:
    st.session_state.all_alerts = []
if "zone_scores" not in st.session_state:
    st.session_state.zone_scores = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pose_extractor():
    try:
        from pose_extractor import PoseExtractor
        return PoseExtractor(model_type="mediapipe", confidence=0.5)
    except Exception as exc:
        st.warning(f"MediaPipe unavailable: {exc}. Running demo mode.")
        return None


@st.cache_resource(show_spinner=False)
def load_alert_manager():
    try:
        from alert_system import AlertManager
        return AlertManager()
    except Exception:
        return None


def compute_reba(keypoints: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from reba_rula import compute_reba_score
        return compute_reba_score(keypoints)
    except Exception:
        return {"reba_score": 5, "risk_level": "medium",
                "action_required": "Demo mode", "body_part_scores": {}}


def compute_rula(keypoints: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from reba_rula import compute_rula_score
        return compute_rula_score(keypoints)
    except Exception:
        return {"rula_score": 3, "risk_level": "medium",
                "recommended_action": "Demo mode", "body_part_scores": {}}


def extract_keypoints_from_image(image: np.ndarray) -> Dict[str, Any]:
    extractor = load_pose_extractor()
    if extractor is not None:
        try:
            return extractor.extract_keypoints(image)
        except Exception:
            pass
    return {}


def draw_annotated(image: np.ndarray, keypoints: Dict, reba_result: Dict) -> np.ndarray:
    try:
        from visualization import draw_skeleton, draw_reba_overlay
        img = draw_skeleton(image.copy(), keypoints, reba_result.get("reba_score", 0))
        img = draw_reba_overlay(img, reba_result)
        return img
    except Exception:
        return image


def image_to_bytes(image: np.ndarray) -> bytes:
    try:
        import cv2
        _, buf = cv2.imencode(".jpg", image)
        return buf.tobytes()
    except Exception:
        from PIL import Image as PILImg
        pil = PILImg.fromarray(image[:, :, ::-1] if image.ndim == 3 else image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        return buf.getvalue()


def decode_upload(file_bytes: bytes) -> np.ndarray:
    try:
        import cv2
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    from PIL import Image as PILImg
    pil = PILImg.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(pil)[:, :, ::-1]


def reba_badge(reba_result: Dict) -> None:
    score = reba_result.get("reba_score", 0)
    risk = reba_result.get("risk_level", "unknown")
    action = reba_result.get("action_required", "")
    css = "critical" if score >= 11 else "high" if score >= 8 else "medium" if score >= 4 else "low"
    st.markdown(f"""
    <div class="reba-badge-{css}">
        <div class="reba-label">REBA Score</div>
        <div class="reba-value">{score} / 15</div>
        <div class="reba-label">Risk Level: {risk.upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"Action: {action}")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/spine-health.png", width=60)
    st.title("ErgoPro AI")
    st.markdown("**Workplace Injury Prevention**")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Live Assessment", "Worker Monitoring", "Zone Risk Map",
         "Shift Analytics", "Alert Center", "OSHA Compliance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Powered by MediaPipe + REBA/RULA")
    st.caption("OSHA MSD Prevention Platform")
    st.divider()
    selected_worker = st.text_input("Worker ID", value="worker_001")


# ---------------------------------------------------------------------------
# Page: Live Assessment
# ---------------------------------------------------------------------------
if page == "Live Assessment":
    st.title("Live Ergonomic Assessment")
    st.markdown(
        "Upload a worker image or video frame to receive instant REBA/RULA scoring, "
        "skeleton overlay, and injury risk breakdown."
    )

    upload_type = st.radio("Input type", ["Image", "Video Frame"], horizontal=True)
    uploaded = st.file_uploader(
        "Upload worker image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="live_upload",
    )

    if uploaded:
        file_bytes = uploaded.read()
        image = decode_upload(file_bytes)

        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.subheader("Original")
            st.image(file_bytes, use_container_width=True)

        with st.spinner("Analyzing pose..."):
            keypoints = extract_keypoints_from_image(image)
            reba_result = compute_reba(keypoints)
            rula_result = compute_rula(keypoints)
            annotated = draw_annotated(image, keypoints, reba_result)

        with col_annot:
            st.subheader("Pose Analysis")
            st.image(image_to_bytes(annotated), use_container_width=True)

        st.divider()

        # REBA score badge + RULA
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            reba_badge(reba_result)
        with c2:
            rula_score = rula_result.get("rula_score", 0)
            rula_risk = rula_result.get("risk_level", "unknown")
            st.metric("RULA Score", f"{rula_score} / 7", help="Rapid Upper Limb Assessment")
            st.caption(f"Risk: {rula_risk.upper()}")
            st.info(rula_result.get("recommended_action", ""))
        with c3:
            st.subheader("Body Part Risk Breakdown (REBA)")
            body_parts = reba_result.get("body_part_scores", {})
            if body_parts:
                import plotly.graph_objects as go
                parts = list(body_parts.keys())
                scores = [body_parts[p].get("score", 0) for p in parts]
                max_scores = [3 if p in ("neck", "wrist") else 6 if p == "upper_arm" else 5 if p == "trunk" else 2 for p in parts]
                ratios = [s / max(m, 1) for s, m in zip(scores, max_scores)]
                colors = ["#2ecc71" if r < 0.4 else "#f39c12" if r < 0.7 else "#e74c3c" for r in ratios]
                fig = go.Figure(go.Bar(
                    x=parts, y=scores,
                    marker_color=colors,
                    text=scores, textposition="outside",
                ))
                fig.update_layout(yaxis_title="Score", height=260, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        # Store in session history
        worker_id = selected_worker
        if worker_id not in st.session_state.worker_history:
            st.session_state.worker_history[worker_id] = []
        st.session_state.worker_history[worker_id].append({
            "reba_score": reba_result.get("reba_score", 0),
            "rula_score": rula_result.get("rula_score", 0),
            "timestamp": time.time(),
            "risk_level": reba_result.get("risk_level", ""),
        })

        # Alert check
        alert_mgr = load_alert_manager()
        if alert_mgr is not None:
            alert = alert_mgr.check_alert(
                worker_id=worker_id,
                reba_score=reba_result.get("reba_score", 0),
                timestamp=time.time(),
                zone="unknown",
            )
            if alert:
                st.error(f"ALERT TRIGGERED: {alert['type']} — {alert['message']}")
                st.session_state.all_alerts.append(alert)

        # Keypoints table
        if keypoints:
            with st.expander("Raw Keypoints"):
                import pandas as pd
                kp_rows = [{"landmark": k, "x": round(v["x"], 4),
                             "y": round(v["y"], 4), "visibility": round(v.get("visibility", 1.0), 3)}
                            for k, v in keypoints.items()]
                st.dataframe(pd.DataFrame(kp_rows), use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Page: Worker Monitoring
# ---------------------------------------------------------------------------
elif page == "Worker Monitoring":
    st.title("Worker Risk Monitoring")

    worker_ids = list(st.session_state.worker_history.keys()) or ["worker_001", "worker_002", "worker_003"]
    selected = st.selectbox("Select Worker", worker_ids)

    history = st.session_state.worker_history.get(selected, [])

    # Demo data if no history
    if not history:
        rng = np.random.RandomState(42)
        n = 80
        demo_scores = np.clip(rng.normal(5.5, 2.0, n), 1, 15).astype(int).tolist()
        demo_times = [time.time() - (n - i) * 60 for i in range(n)]
        history = [{"reba_score": s, "timestamp": t, "risk_level": "", "rula_score": 3}
                   for s, t in zip(demo_scores, demo_times)]
        st.info(f"Showing demo data for {selected}.")

    reba_scores = [h["reba_score"] for h in history]
    times_h = [(h["timestamp"] - history[0]["timestamp"]) / 3600 for h in history]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg REBA", f"{np.mean(reba_scores):.2f}")
    k2.metric("Peak REBA", max(reba_scores))
    k3.metric("Observations", len(reba_scores))
    pct_high = sum(1 for s in reba_scores if s >= 8) / max(len(reba_scores), 1) * 100
    k4.metric("% Time at High Risk", f"{pct_high:.1f}%")

    st.divider()

    col_timeline, col_donut = st.columns([2, 1])

    with col_timeline:
        st.subheader("REBA Score Timeline")
        import plotly.graph_objects as go
        fig = go.Figure()
        # Risk zone bands
        for y_low, y_high, color, label in [
            (0, 3, "rgba(46,204,113,0.1)", "Low"),
            (3, 7, "rgba(243,156,18,0.1)", "Medium"),
            (7, 10, "rgba(231,76,60,0.1)", "High"),
            (10, 15, "rgba(142,68,173,0.1)", "Very High"),
        ]:
            fig.add_hrect(y0=y_low, y1=y_high, fillcolor=color,
                          line_width=0, annotation_text=label,
                          annotation_position="right",
                          annotation_font_size=9)
        fig.add_trace(go.Scatter(
            x=times_h, y=reba_scores, mode="lines+markers",
            name="REBA", line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title="Hours into Shift",
            yaxis_title="REBA Score",
            yaxis=dict(range=[0, 16]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_donut:
        st.subheader("Cumulative Exposure")
        from reba_rula import get_reba_risk_level
        counts: Dict[str, int] = {"negligible": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0}
        for s in reba_scores:
            level = get_reba_risk_level(s)
            counts[level] = counts.get(level, 0) + 1
        import plotly.graph_objects as go
        fig_donut = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.55,
            marker_colors=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"],
        ))
        fig_donut.update_layout(height=300, margin=dict(t=10, b=10),
                                showlegend=True, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("Fatigue Detection")
    from risk_analytics import compute_posture_fatigue
    fake_timestamps = [i * 60.0 for i in range(len(reba_scores))]
    fatigue = compute_posture_fatigue(reba_scores, fake_timestamps)
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("Trend", fatigue["trend_label"].replace("_", " ").title())
    fcol2.metric("Slope", f"{fatigue['slope']:+.3f}")
    fcol3.metric("Start vs End REBA",
                 f"{fatigue.get('start_avg_reba', 0):.1f} → {fatigue.get('end_avg_reba', 0):.1f}")
    if fatigue["is_fatiguing"]:
        st.warning("Fatigue pattern detected. Recommend rest break.")


# ---------------------------------------------------------------------------
# Page: Zone Risk Map
# ---------------------------------------------------------------------------
elif page == "Zone Risk Map":
    st.title("Zone Risk Map")
    st.markdown("Warehouse/factory floor heatmap showing ergonomic risk scores by zone.")

    # Demo zone data
    zones_demo = {
        "Receiving Dock": 8.2,
        "Conveyor A": 6.5,
        "Conveyor B": 7.8,
        "Picking Zone 1": 5.1,
        "Picking Zone 2": 9.3,
        "Packing Station": 6.8,
        "Loading Bay": 8.7,
        "Break Room": 1.2,
        "Office": 2.1,
        "Quality Check": 5.5,
        "Returns": 4.8,
        "Freezer Zone": 7.1,
    }

    zone_data = zones_demo.copy()
    for zone, scores in st.session_state.zone_scores.items():
        if scores:
            zone_data[zone] = float(np.mean(scores))

    # Plotly heatmap grid (3x4)
    import plotly.graph_objects as go
    import plotly.express as px

    zone_names = list(zone_data.keys())
    zone_scores_vals = [zone_data[z] for z in zone_names]

    n_cols = 4
    n_rows = -(-len(zone_names) // n_cols)
    grid = np.full((n_rows, n_cols), np.nan)
    grid_labels = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for i, (name, score) in enumerate(zip(zone_names, zone_scores_vals)):
        r, c = divmod(i, n_cols)
        grid[r, c] = score
        grid_labels[r][c] = f"{name}<br>REBA: {score:.1f}"

    fig = go.Figure(go.Heatmap(
        z=grid.tolist(),
        text=grid_labels,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=[
            [0.0, "#2ecc71"], [0.2, "#27ae60"],
            [0.4, "#f39c12"], [0.65, "#e74c3c"],
            [1.0, "#8e44ad"],
        ],
        zmin=1, zmax=15,
        colorbar=dict(title="Avg REBA Score"),
    ))
    fig.update_layout(
        title="Warehouse Floor Risk Heatmap",
        height=450,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top risk zones
    st.subheader("Highest Risk Zones")
    import pandas as pd
    sorted_zones = sorted(zone_data.items(), key=lambda x: x[1], reverse=True)
    df_zones = pd.DataFrame([{"Zone": z, "Avg REBA": round(s, 2),
                               "Risk Level": "High" if s >= 8 else "Medium" if s >= 5 else "Low"}
                              for z, s in sorted_zones])
    st.dataframe(df_zones, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Shift Analytics
# ---------------------------------------------------------------------------
elif page == "Shift Analytics":
    st.title("Shift Analytics")

    # Demo multi-role comparison data
    roles = ["Picker", "Packer", "Loader", "Forklift Driver", "Quality Inspector"]
    role_avg_reba = [7.2, 5.8, 8.9, 4.1, 5.5]
    role_high_risk_pct = [42, 28, 67, 12, 25]

    col_box, col_trend = st.columns(2)

    with col_box:
        st.subheader("Risk by Job Role (Box Plot)")
        import plotly.graph_objects as go
        import numpy as np
        rng = np.random.RandomState(42)
        fig = go.Figure()
        colors_role = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
        for role, avg, color in zip(roles, role_avg_reba, colors_role):
            spread = rng.normal(avg, 1.5, 50)
            spread = np.clip(spread, 1, 15)
            fig.add_trace(go.Box(
                y=spread.tolist(), name=role,
                marker_color=color, line_color=color,
                boxpoints="outliers",
            ))
        fig.update_layout(yaxis=dict(range=[0, 16], title="REBA Score"), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_trend:
        st.subheader("Daily Average REBA Trend")
        days = [f"Day {i}" for i in range(1, 15)]
        rng2 = np.random.RandomState(7)
        for role, avg in [("Picker", 7.2), ("Loader", 8.9), ("Packer", 5.8)]:
            trend = avg - rng2.uniform(0, 0.1, 14).cumsum() * 0.2 + rng2.normal(0, 0.3, 14)
            trend = np.clip(trend, 1, 15)
            fig2 = go.Figure() if role == "Picker" else fig2
            fig2.add_trace(go.Scatter(
                x=days, y=trend.tolist(), mode="lines+markers",
                name=role, line=dict(width=2),
            ))
        fig2.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="High risk threshold")
        fig2.update_layout(yaxis=dict(range=[0, 15], title="Avg REBA"), height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top High-Risk Actions Identified")
    import pandas as pd
    actions_df = pd.DataFrame({
        "Action": ["Overhead stacking", "Floor-level picking", "Repetitive bending",
                   "Prolonged reaching", "Asymmetric lifting", "Sustained static posture"],
        "Avg REBA": [11.2, 9.8, 8.4, 7.9, 9.1, 6.8],
        "Frequency/Shift": [45, 120, 89, 67, 34, 210],
        "Job Role": ["Loader", "Picker", "Packer", "Picker", "Loader", "Quality Inspector"],
        "Recommended Fix": [
            "Install height-adjustable shelving",
            "Use gravity-feed conveyors",
            "Redesign bin height",
            "Reduce reach distance",
            "Add mechanical assist",
            "Implement rotation policy",
        ],
    })
    st.dataframe(actions_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Alert Center
# ---------------------------------------------------------------------------
elif page == "Alert Center":
    st.title("Alert Center")

    demo_alerts = [
        {"time": "09:14:32", "type": "IMMEDIATE_ACTION", "worker": "worker_003",
         "zone": "Loading Bay", "severity": "critical",
         "message": "REBA score 13 detected. Very high injury risk. Immediate intervention.", "status": "ACTIVE"},
        {"time": "09:08:55", "type": "HIGH_RISK_SUSTAINED", "worker": "worker_007",
         "zone": "Picking Zone 2", "severity": "high",
         "message": "Worker maintained REBA ≥ 8 for 45s. Ergonomic review needed.", "status": "ACTIVE"},
        {"time": "08:52:10", "type": "FATIGUE_DETECTED", "worker": "worker_001",
         "zone": "Conveyor A", "severity": "medium",
         "message": "Fatigue pattern: REBA increasing +1.2 points/10 min. Recommend rest break.", "status": "ACTIVE"},
        {"time": "08:31:44", "type": "ZONE_RISK", "worker": "worker_005",
         "zone": "Receiving Dock", "severity": "medium",
         "message": "Zone repeatedly shows REBA ≥ 7. Workstation redesign recommended.", "status": "RESOLVED"},
        {"time": "07:45:00", "type": "HIGH_RISK_SUSTAINED", "worker": "worker_002",
         "zone": "Packing Station", "severity": "high",
         "message": "Sustained high risk resolved after workstation adjustment.", "status": "RESOLVED"},
    ]

    session_alerts = st.session_state.all_alerts
    combined = demo_alerts + [
        {"time": time.strftime("%H:%M:%S", time.localtime(a.get("created_at", 0))),
         "type": a.get("type", ""), "worker": a.get("worker_id", ""),
         "zone": a.get("zone", ""), "severity": a.get("severity", ""),
         "message": a.get("message", ""), "status": "ACTIVE"}
        for a in session_alerts
    ]

    active = [a for a in combined if a["status"] == "ACTIVE"]
    resolved = [a for a in combined if a["status"] == "RESOLVED"]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Active Alerts", len(active))
    a2.metric("Critical", sum(1 for a in active if a["severity"] == "critical"))
    a3.metric("High", sum(1 for a in active if a["severity"] == "high"))
    a4.metric("Resolved Today", len(resolved))

    st.divider()
    st.subheader("Active Alerts")
    for alert in active:
        sev = alert["severity"]
        icon = "🚨" if sev == "critical" else "🔴" if sev == "high" else "🟡"
        css = f"alert-{sev}" if sev in ("critical", "high", "medium") else "alert-medium"
        st.markdown(
            f'<div class="{css}">'
            f'<strong>{icon} [{sev.upper()}] {alert["type"]}</strong>'
            f' — Worker: {alert["worker"]} | Zone: {alert["zone"]}<br>'
            f'{alert["message"]}<br>'
            f'<small>⏰ {alert["time"]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Alert History")
    import pandas as pd
    df_alerts = pd.DataFrame(combined)
    st.dataframe(df_alerts, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: OSHA Compliance
# ---------------------------------------------------------------------------
elif page == "OSHA Compliance":
    st.title("OSHA MSD Prevention Compliance")

    st.markdown("""
    **OSHA's Ergonomics Standard** targets Musculoskeletal Disorders (MSDs), the leading
    cause of workplace injury. This dashboard tracks compliance with OSHA General Duty Clause
    requirements and internal ergonomic targets.
    """)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Compliance Rate", "76.4%", delta="+3.2% vs last month")
    k2.metric("% Time at Low/Negligible Risk", "58.3%", delta="+5.1%", help="Target: ≥70%")
    k3.metric("MSD Incident Rate", "4.2 per 200K hrs", delta="-0.8 (improvement)")
    k4.metric("Days Away (Ergonomic)", "23 YTD", delta="-7 vs last year")

    st.divider()

    col_trend, col_osha = st.columns(2)

    with col_trend:
        st.subheader("Compliance Rate Trend")
        import plotly.graph_objects as go
        import numpy as np
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        compliance_trend = [65, 67, 68, 70, 71, 73, 72, 74, 75, 74, 76, 78]
        target = [70] * 12
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=compliance_trend, mode="lines+markers",
                                  name="Actual Compliance %", line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=months, y=target, mode="lines",
                                  name="OSHA Target (70%)", line=dict(color="red", dash="dash")))
        fig.update_layout(yaxis=dict(range=[50, 90], title="Compliance (%)"), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_osha:
        st.subheader("Risk Distribution (YTD)")
        import plotly.graph_objects as go
        fig2 = go.Figure(go.Pie(
            labels=["Negligible", "Low", "Medium", "High", "Very High"],
            values=[15.2, 43.1, 28.4, 10.7, 2.6],
            hole=0.5,
            marker_colors=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"],
        ))
        fig2.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("OSHA Incident Report Export")

    report_text = """
OSHA ERGONOMICS COMPLIANCE REPORT
Company: Acme Fulfillment Center
Period: YTD {year}
Generated: {date}

EXECUTIVE SUMMARY
-----------------
Total Workers Monitored: 247
Shifts Analyzed: 1,842
Total Pose Assessments: 184,320

REBA Score Distribution:
  Negligible (1):     15.2%
  Low (2-3):          43.1%
  Medium (4-7):       28.4%
  High (8-10):        10.7%
  Very High (11-15):   2.6%

Compliance Rate: 76.4% (Target: 70%) ✓
Workers at Chronic High Risk: 12 (4.9%)

TOP RISK FACTORS:
1. Overhead stacking in Loading Bay (Avg REBA 11.2)
2. Floor-level picking in Pick Zone 2 (Avg REBA 9.8)
3. Asymmetric lifting at Receiving Dock (Avg REBA 9.1)

CORRECTIVE ACTIONS IN PROGRESS:
1. Height-adjustable shelving installation (3 zones) — ETA: 30 days
2. Mechanical lift assist deployment — COMPLETE
3. Job rotation policy update — COMPLETE
4. Ergonomics training refresher (all staff) — In Progress

MSD INCIDENT STATISTICS:
  Incidents YTD: 18 (target <20) ✓
  Days Away Work Related: 23
  Restricted Duty Days: 41
  Incident Rate: 4.2 per 200,000 hrs worked

NEXT ASSESSMENT DATE: {next_date}
""".format(
        year=time.strftime("%Y"),
        date=time.strftime("%Y-%m-%d"),
        next_date=time.strftime("%Y-%m-%d",
                                time.localtime(time.time() + 30 * 86400)),
    )

    st.text_area("OSHA Report Preview", report_text, height=300)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download OSHA Report (TXT)",
            data=report_text,
            file_name=f"osha_report_{time.strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )
    with col_dl2:
        import pandas as pd
        csv_data = pd.DataFrame({
            "Metric": ["Compliance Rate", "% Low Risk", "MSD Incidents", "Days Away"],
            "Value": ["76.4%", "58.3%", "18", "23"],
            "Target": ["70%", "70%", "<20", "<30"],
            "Status": ["PASS", "FAIL", "PASS", "PASS"],
        })
        st.download_button(
            "Download CSV Metrics",
            data=csv_data.to_csv(index=False),
            file_name=f"ergonomics_metrics_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

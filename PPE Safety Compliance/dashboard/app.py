"""
Streamlit Dashboard — Workplace Safety & PPE Compliance Monitoring
Tabs: Live Monitor | Site Overview | Violation Analytics |
      Alert Management | OSHA Compliance Report
"""

import base64
import io
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="PPE Safety Compliance Platform",
    page_icon="⛑️",
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

# ─── Session State & Model Init ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PPE detection model...")
def load_detector():
    from src.ppe_detector import PPEDetector
    return PPEDetector(conf_threshold=0.4, iou_threshold=0.5)

@st.cache_resource(show_spinner=False)
def load_checker():
    from src.compliance_engine import PPEComplianceChecker
    return PPEComplianceChecker()

@st.cache_resource(show_spinner=False)
def load_alert_manager():
    from src.alert_system import AlertManager
    return AlertManager()

detector = load_detector()
checker = load_checker()
alert_manager = load_alert_manager()


def _generate_demo_violations() -> list[dict[str, Any]]:
    """Generate realistic demo violation log for dashboard."""
    rng = np.random.default_rng(42)
    zones = ["construction", "warehouse", "chemical", "general"]
    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ppe_types = [
        ["No_Hardhat"], ["No_Safety_Vest"], ["No_Hardhat", "No_Safety_Vest"],
        ["Safety_Glasses"], ["Gloves"],
    ]
    workers = [f"W{i:03d}" for i in range(1, 21)]

    violations = []
    now = datetime.now()
    for i in range(120):
        ts = now - timedelta(hours=rng.integers(0, 72), minutes=rng.integers(0, 60))
        zone = rng.choice(zones)
        missing = rng.choice(ppe_types).copy()
        sev = rng.choice(severities, p=[0.10, 0.25, 0.40, 0.25])
        violations.append({
            "violation_id": f"V{i:04d}",
            "worker_id": rng.choice(workers),
            "zone": zone,
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M:%S"),
            "shift": "Day" if 6 <= ts.hour < 18 else "Night",
            "missing_ppe": missing,
            "severity": sev,
            "status": rng.choice(["open", "resolved"], p=[0.4, 0.6]),
            "osha_code": "29 CFR 1926.100(a)",
            "worker_location": (int(rng.integers(50, 590)), int(rng.integers(50, 430))),
        })

    return violations

if "violation_log" not in st.session_state:
    st.session_state["violation_log"] = _generate_demo_violations()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PPE Compliance Monitor")
    st.markdown("**Amazon • Boeing • Ford • Honeywell**")
    st.divider()

    vlog = st.session_state.get("violation_log", [])
    total_v = len(vlog)
    open_v = sum(1 for v in vlog if v.get("status") == "open")
    critical_v = sum(1 for v in vlog if v.get("severity") == "CRITICAL" and v.get("status") == "open")

    from src.compliance_engine import compute_site_compliance_rate
    site_rate = compute_site_compliance_rate(vlog, time_window_hours=8)

    st.metric("Site Compliance", f"{site_rate:.1f}%",
              delta="Safe" if site_rate >= 90 else "At Risk")
    st.metric("Active Violations", f"{open_v}")
    st.metric("Critical Alerts", f"{critical_v}")
    st.metric("Total Violations (72h)", f"{total_v}")

    st.divider()
    st.markdown("**Model Status**")
    if detector.model:
        st.success("YOLOv8 Loaded")
    else:
        st.warning("Fallback Mode (HOG)")

    selected_zone = st.selectbox(
        "Active Zone", ["construction", "warehouse", "chemical", "office", "general"]
    )
    st.caption("OSHA Target: 95% compliance")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Monitor", "Site Overview", "Violation Analytics",
    "Alert Management", "OSHA Report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Monitor
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Live PPE Compliance Monitor")

    col_upload, col_result = st.columns([1, 2])

    with col_upload:
        st.subheader("Upload Image or Video")
        media_type = st.radio("Media Type", ["Image", "Video Frame (Base64)"], horizontal=True)

        if media_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload Image", type=["jpg", "jpeg", "png", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                with st.spinner("Detecting PPE..."):
                    t0 = time.time()
                    detections = detector.detect(image)
                    compliance = checker.check_worker_compliance(detections, zone=selected_zone)
                    elapsed = time.time() - t0

                from src.visualization import draw_ppe_detections
                annotated = draw_ppe_detections(image, detections, compliance)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                with col_result:
                    st.image(annotated_rgb, caption="Annotated Detection", use_column_width=True)

                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Workers", compliance.get("workers_detected", 0))
                    score = compliance.get("compliance_score", 100)
                    c2.metric("Compliance", f"{score:.0f}%",
                              delta="OK" if score >= 90 else "Non-Compliant")
                    c3.metric("Violations", compliance.get("violations_count", 0))
                    c4.metric("Inference", f"{elapsed*1000:.0f}ms")

                    # Detections table
                    if detections:
                        det_data = [d.to_dict() if hasattr(d, "to_dict") else d for d in detections]
                        st.dataframe(
                            pd.DataFrame(det_data)[["class_name", "confidence", "center_x", "center_y"]],
                            use_container_width=True,
                        )

                    # Auto-create alerts for violations
                    workers_detail = compliance.get("workers_detail", [])
                    for w in workers_detail:
                        if not w.get("is_compliant", True):
                            vrec = checker.generate_violation_record(
                                worker_id=w.get("worker_id", "unknown"),
                                missing_ppe=w.get("missing_ppe", []),
                                zone=selected_zone,
                            )
                            st.session_state["violation_log"].append(vrec)
                            alert = alert_manager.create_alert(vrec)
                            if w.get("missing_ppe"):
                                st.error(
                                    f"VIOLATION: Worker {w['worker_id']} missing "
                                    f"{', '.join(w['missing_ppe'])} "
                                    f"[{vrec['severity']}]"
                                )

        else:
            b64_input = st.text_area("Paste Base64 JPEG frame", height=100)
            if b64_input and st.button("Analyze Frame"):
                try:
                    img_bytes = base64.b64decode(b64_input)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    detections = detector.detect(image)
                    compliance = checker.check_worker_compliance(detections, zone=selected_zone)
                    from src.visualization import draw_ppe_detections
                    annotated = draw_ppe_detections(image, detections, compliance)
                    with col_result:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                except Exception as exc:
                    st.error(f"Frame analysis failed: {exc}")

    # Compliance gauge (always shown)
    with col_result:
        if not (media_type == "Image" and uploaded_file):
            from src.visualization import create_compliance_gauge
            vlog = st.session_state.get("violation_log", [])
            from src.compliance_engine import compute_site_compliance_rate
            gauge_rate = compute_site_compliance_rate(vlog, time_window_hours=1)
            gauge = create_compliance_gauge(gauge_rate, title="Current Site Compliance")
            if gauge:
                st.plotly_chart(gauge, use_container_width=True)

            # Active violation feed
            st.subheader("Active Violation Feed")
            open_violations = [v for v in vlog if v.get("status") == "open"][-10:]
            if open_violations:
                vdf = pd.DataFrame(open_violations[::-1])
                st.dataframe(
                    vdf[["timestamp", "worker_id", "zone", "missing_ppe", "severity", "status"]],
                    use_container_width=True,
                )
            else:
                st.success("No active violations")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Site Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Site Safety Overview")

    vlog = st.session_state.get("violation_log", [])
    from src.compliance_engine import (
        compute_site_compliance_rate, compute_compliance_by_zone,
        compute_compliance_trend,
    )
    from src.visualization import create_compliance_gauge, create_violation_timeline

    site_rate = compute_site_compliance_rate(vlog, time_window_hours=8)
    open_v = sum(1 for v in vlog if v.get("status") == "open")
    resolved_today = sum(1 for v in vlog if v.get("status") == "resolved"
                        and v.get("date") == datetime.now().strftime("%Y-%m-%d"))
    zone_stats = compute_compliance_by_zone(vlog)
    at_risk_zones = [z for z, s in zone_stats.items() if s.get("compliance_rate", 100) < 90]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Site Compliance %", f"{site_rate:.1f}%",
                delta="Safe" if site_rate >= 90 else "At Risk")
    col2.metric("Active Violations", open_v,
                delta="Critical" if any(v.get("severity") == "CRITICAL" for v in vlog if v.get("status") == "open") else "")
    col3.metric("Resolved Today", resolved_today)
    col4.metric("At-Risk Zones", len(at_risk_zones),
                delta=", ".join(at_risk_zones[:2]) if at_risk_zones else "None")

    col_a, col_b = st.columns(2)

    with col_a:
        gauge = create_compliance_gauge(site_rate, "Overall Site Compliance")
        if gauge:
            st.plotly_chart(gauge, use_container_width=True)

    with col_b:
        # Zone compliance heatmap
        if zone_stats:
            zone_df = pd.DataFrame([
                {"Zone": z, "Compliance (%)": s.get("compliance_rate", 100),
                 "Violations": s.get("violations", 0)}
                for z, s in zone_stats.items()
            ])
            fig_zone = px.bar(
                zone_df, x="Zone", y="Compliance (%)",
                color="Compliance (%)",
                color_continuous_scale=["red", "yellow", "green"],
                range_color=[0, 100],
                title="Compliance by Zone",
                text="Compliance (%)",
            )
            fig_zone.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="OSHA Target")
            fig_zone.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig_zone.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_zone, use_container_width=True)

    # Compliance trend
    st.subheader("Compliance Trend (Last 24 Hours)")
    trend = compute_compliance_trend(vlog, period="hourly")
    if trend:
        trend_df = pd.DataFrame(trend)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_df["period_label"], y=trend_df["compliance_rate"],
            mode="lines+markers", name="Compliance",
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
        ))
        fig_trend.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target")
        fig_trend.add_hline(y=80, line_dash="dot", line_color="orange")
        fig_trend.update_layout(
            yaxis=dict(range=[0, 105]),
            xaxis_title="Hour", yaxis_title="Compliance Rate (%)",
            height=300, template="plotly_white",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Violation timeline
    timeline_fig = create_violation_timeline(vlog)
    if timeline_fig:
        st.subheader("Violations Per Hour")
        st.plotly_chart(timeline_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Violation Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Violation Analytics")

    vlog = st.session_state.get("violation_log", [])
    from src.compliance_engine import identify_repeat_offenders
    from src.visualization import create_site_heatmap

    if vlog:
        vdf = pd.DataFrame(vlog)
        vdf["timestamp"] = pd.to_datetime(vdf["timestamp"], errors="coerce")
        vdf["hour"] = vdf["timestamp"].dt.hour

        col_a, col_b = st.columns(2)

        with col_a:
            # Violations by PPE type
            ppe_counts: dict[str, int] = {}
            for v in vlog:
                for ppe in v.get("missing_ppe", []):
                    ppe_counts[ppe] = ppe_counts.get(ppe, 0) + 1

            if ppe_counts:
                ppe_df = pd.DataFrame(list(ppe_counts.items()), columns=["PPE Item", "Violations"])
                ppe_df = ppe_df.sort_values("Violations", ascending=False)
                fig_ppe = px.bar(
                    ppe_df, x="PPE Item", y="Violations",
                    color="PPE Item", title="Violations by PPE Type",
                    color_discrete_sequence=px.colors.qualitative.Set1,
                )
                st.plotly_chart(fig_ppe, use_container_width=True)

        with col_b:
            # Time-of-day pattern
            hour_counts = vdf.groupby("hour").size().reset_index(name="violations")
            fig_hour = px.bar(
                hour_counts, x="hour", y="violations",
                title="Violations by Hour of Day",
                color="violations", color_continuous_scale="Reds",
            )
            fig_hour.update_xaxes(title="Hour (24h)")
            st.plotly_chart(fig_hour, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            # Zone comparison
            zone_counts = vdf.groupby("zone").size().reset_index(name="violations")
            fig_zone = px.bar(
                zone_counts, x="violations", y="zone", orientation="h",
                title="Violations by Zone",
                color="violations", color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig_zone, use_container_width=True)

        with col_d:
            # Severity breakdown pie
            sev_counts = vdf["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            sev_colors = {
                "CRITICAL": "#c0392b", "HIGH": "#e67e22",
                "MEDIUM": "#f1c40f", "LOW": "#27ae60",
            }
            fig_sev = px.pie(
                sev_counts, values="count", names="severity",
                title="Violations by Severity",
                color="severity",
                color_discrete_map=sev_colors,
            )
            st.plotly_chart(fig_sev, use_container_width=True)

        # Repeat offenders
        st.subheader("Repeat Offenders")
        offenders = identify_repeat_offenders(vlog, min_violations=2, shift_window_hours=72)
        if offenders:
            off_df = pd.DataFrame(offenders)
            st.dataframe(
                off_df[["worker_id", "violation_count", "ppe_types", "risk_level"]],
                use_container_width=True,
            )
        else:
            st.info("No repeat offenders detected in the monitoring window.")

        # Location heatmap
        st.subheader("Violation Location Heatmap")
        heatmap_fig = create_site_heatmap(vlog, floor_plan_dims=(640, 480), grid_size=30)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("No violations data available. Upload images in the Live Monitor tab.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Alert Management
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Alert Management Center")

    vlog = st.session_state.get("violation_log", [])

    # Auto-populate alert manager with demo violations
    if len(alert_manager) == 0 and vlog:
        for v in vlog[:30]:
            alert_manager.create_alert(v)

    active_alerts = alert_manager.get_active_alerts()
    alert_summary = alert_manager.get_shift_alert_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Open Alerts", len(active_alerts))
    col2.metric("Critical Alerts",
                len([a for a in active_alerts if a.get("severity") == "CRITICAL"]))
    col3.metric("Resolution Rate", f"{alert_summary.get('resolution_rate_pct', 0):.0f}%")
    col4.metric("Avg Resolution Time", f"{alert_summary.get('avg_resolution_time_min', 0):.0f} min")

    severity_filter = st.multiselect(
        "Filter by Severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"],
    )

    filtered_alerts = alert_manager.get_active_alerts(severity_filter=severity_filter)

    if filtered_alerts:
        alerts_df = pd.DataFrame(filtered_alerts)
        display_cols = [c for c in ["id", "timestamp", "worker_id", "zone", "severity",
                                     "status", "osha_code"] if c in alerts_df.columns]

        def _color_severity(val: str) -> str:
            colors = {
                "CRITICAL": "background-color: #ffcccc",
                "HIGH": "background-color: #ffe0cc",
                "MEDIUM": "background-color: #fffbcc",
                "LOW": "background-color: #ccffcc",
            }
            return colors.get(val, "")

        styled = alerts_df[display_cols].style.applymap(
            _color_severity, subset=["severity"] if "severity" in display_cols else []
        )
        st.dataframe(styled, use_container_width=True)

        # Resolve button
        col_res1, col_res2 = st.columns(2)
        resolve_id = col_res1.text_input("Alert ID to Resolve", placeholder="ALT-XXXXXXXX")
        resolved_by = col_res2.text_input("Resolved By", value="Safety Supervisor")

        if st.button("Resolve Alert", type="primary"):
            if resolve_id:
                result = alert_manager.resolve_alert(resolve_id, resolved_by=resolved_by)
                if result:
                    st.success(f"Alert {resolve_id} resolved by {resolved_by}")
                    st.rerun()
                else:
                    st.error(f"Alert {resolve_id} not found")
    else:
        st.success("No active alerts matching filter criteria")

    # Alert history
    st.subheader("Alert History")
    all_alerts = alert_manager.all_alerts
    if all_alerts:
        hist_df = pd.DataFrame(all_alerts)
        if "severity" in hist_df.columns:
            sev_hist = hist_df["severity"].value_counts()
            fig_hist = px.bar(
                x=sev_hist.index, y=sev_hist.values,
                color=sev_hist.index,
                color_discrete_map={
                    "CRITICAL": "#c0392b", "HIGH": "#e67e22",
                    "MEDIUM": "#f1c40f", "LOW": "#27ae60",
                },
                title="Alert History by Severity",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: OSHA Compliance Report
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("OSHA Compliance Report")

    from src.alert_system import generate_osha_incident_report
    from src.compliance_engine import compute_site_compliance_rate

    vlog = st.session_state.get("violation_log", [])

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    site_rate = compute_site_compliance_rate(vlog, time_window_hours=24)
    critical_count = sum(1 for v in vlog if v.get("severity") == "CRITICAL")
    high_count = sum(1 for v in vlog if v.get("severity") == "HIGH")

    # OSHA benchmark: construction industry incident rate ~3.1 per 100 workers
    incident_rate = round(len(vlog) / max(100, 1) * 100, 2)

    col_kpi1.metric("Compliance vs OSHA Target",
                    f"{site_rate:.1f}%",
                    delta=f"{site_rate - 95:.1f}% vs 95% target")
    col_kpi2.metric("Incident Rate (per 100)", f"{incident_rate:.1f}",
                    delta="vs Industry: 3.1")
    col_kpi3.metric("Critical Incidents", critical_count)
    col_kpi4.metric("OSHA Status",
                    "COMPLIANT" if site_rate >= 95 else "ACTION REQUIRED")

    # Compliance vs OSHA target bar
    fig_osha = go.Figure()
    fig_osha.add_trace(go.Bar(
        x=["Current Compliance", "OSHA Target", "Industry Best"],
        y=[site_rate, 95.0, 98.5],
        marker_color=["#3498db", "#e74c3c", "#2ecc71"],
        text=[f"{v:.1f}%" for v in [site_rate, 95.0, 98.5]],
        textposition="outside",
    ))
    fig_osha.update_layout(
        title="Compliance vs OSHA Target & Industry Benchmark",
        yaxis=dict(range=[0, 105]),
        height=350, template="plotly_white",
    )
    st.plotly_chart(fig_osha, use_container_width=True)

    # Generate OSHA report
    st.subheader("Generate OSHA 300 Log Report")
    col_date, col_site = st.columns(2)
    report_date = col_date.date_input("Report Date", value=datetime.now().date())
    site_name = col_site.text_input("Facility Name", value="Manufacturing Plant A")

    if st.button("Generate OSHA Report", type="primary"):
        site_info = {
            "site_name": site_name,
            "address": "Industrial District, State",
            "naics_code": "238990",
        }
        report_text = generate_osha_incident_report(
            vlog, site_info, report_date=str(report_date)
        )
        st.text_area("OSHA 300 Log Report", value=report_text, height=400)
        st.download_button(
            "Download OSHA Report (.txt)",
            data=report_text,
            file_name=f"OSHA_300_Log_{report_date}.txt",
            mime="text/plain",
        )

    # Trend vs industry benchmark
    st.subheader("Compliance Trend vs Industry Benchmark")
    months = pd.date_range("2024-01-01", periods=12, freq="ME")
    facility_rates = np.random.normal(site_rate, 3, 12).clip(70, 100)
    industry_rates = np.ones(12) * 87.5

    fig_bench = go.Figure()
    fig_bench.add_trace(go.Scatter(
        x=months, y=facility_rates, name=f"{site_name}",
        line=dict(color="#3498db", width=2), mode="lines+markers",
    ))
    fig_bench.add_trace(go.Scatter(
        x=months, y=industry_rates, name="Industry Average",
        line=dict(color="gray", width=1, dash="dash"),
    ))
    fig_bench.add_hline(y=95, line_dash="dot", line_color="red", annotation_text="OSHA Target 95%")
    fig_bench.update_layout(
        title="Monthly Compliance vs Industry Benchmark",
        yaxis=dict(range=[60, 105], title="Compliance Rate (%)"),
        height=350, template="plotly_white",
    )
    st.plotly_chart(fig_bench, use_container_width=True)

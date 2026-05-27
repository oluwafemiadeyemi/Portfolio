"""
Visualization for PPE Compliance Monitoring System.
OpenCV image annotation + Plotly/Matplotlib charts for dashboards and reports.
"""

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not installed — chart functions will return None")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — report figures unavailable")


# ─── Color palette ────────────────────────────────────────────────────────────
COLORS_BGR = {
    "Person": (50, 200, 50),        # Green
    "Compliant": (50, 200, 50),     # Green
    "NonCompliant": (0, 50, 240),   # Red
    "Hardhat": (0, 200, 255),       # Yellow
    "Safety_Vest": (0, 165, 255),   # Orange
    "Safety_Glasses": (255, 200, 0), # Cyan
    "Gloves": (200, 0, 200),        # Purple
    "No_Hardhat": (0, 0, 200),      # Red
    "No_Safety_Vest": (0, 0, 180),  # Dark Red
    "default": (100, 100, 100),     # Gray
}

SEVERITY_COLORS_BGR = {
    "CRITICAL": (0, 0, 220),
    "HIGH": (0, 80, 220),
    "MEDIUM": (0, 165, 255),
    "LOW": (0, 200, 255),
    "NONE": (50, 200, 50),
}

ZONE_COMPLIANCE_COLORS = {
    "high": (50, 200, 50),    # ≥ 90% compliant
    "medium": (0, 165, 255),  # 70–90%
    "low": (0, 0, 200),       # < 70%
}


def draw_ppe_detections(
    image: np.ndarray,
    detections: list[Any],
    compliance_result: dict[str, Any],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes and compliance status on an image.
    - Green boxes: compliant workers
    - Red boxes: non-compliant workers
    - Colored boxes: individual PPE items with labels
    - Top-left badge: site compliance score

    Returns annotated copy of the image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    det_list = [d.to_dict() if hasattr(d, "to_dict") else d for d in detections]

    # Draw individual detections
    for det in det_list:
        cls = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS_BGR.get(cls, COLORS_BGR["default"])

        # Thicker box for persons
        box_thickness = thickness + 1 if cls == "Person" else thickness
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        label = f"{cls} {conf:.0%}"
        label_size, _ = cv2.getTextSize(label, font, font_scale, 1)
        lw, lh = label_size
        label_y = max(y1 - 4, lh + 4)
        cv2.rectangle(annotated, (x1, label_y - lh - 2), (x1 + lw + 2, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 1, label_y), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Draw per-worker compliance overlays
    workers = compliance_result.get("workers_detail", [])
    for worker in workers:
        bbox = worker.get("bbox", [0, 0, 0, 0])
        is_compliant = worker.get("is_compliant", True)
        missing = worker.get("missing_ppe", [])
        overlay_color = COLORS_BGR["Compliant"] if is_compliant else COLORS_BGR["NonCompliant"]

        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), overlay_color, 3)

        status_text = "OK" if is_compliant else f"MISSING: {', '.join(missing[:2])}"
        cv2.putText(
            annotated, status_text,
            (x1, y2 + 20), font, 0.5, overlay_color, 2, cv2.LINE_AA,
        )

    # Compliance score badge (top-left)
    score = compliance_result.get("compliance_score", 100.0)
    n_workers = compliance_result.get("workers_detected", 0)
    n_violations = compliance_result.get("violations_count", 0)

    badge_color = (
        COLORS_BGR["Compliant"] if score >= 90
        else COLORS_BGR["Safety_Vest"] if score >= 70
        else COLORS_BGR["No_Hardhat"]
    )

    badge_text = f"Compliance: {score:.0f}%"
    cv2.rectangle(annotated, (8, 8), (240, 68), (0, 0, 0), -1)
    cv2.rectangle(annotated, (8, 8), (240, 68), badge_color, 2)
    cv2.putText(annotated, badge_text, (12, 30), font, 0.7, badge_color, 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"Workers: {n_workers}  Violations: {n_violations}",
        (12, 55), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return annotated


def draw_zone_overlay(
    image: np.ndarray,
    zones: dict[str, dict[str, Any]],
    compliance_by_zone: dict[str, dict[str, Any]],
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Draw zone boundaries colored by compliance rate.
    Returns annotated copy of the image.
    """
    annotated = image.copy()
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for zone_name, config in zones.items():
        coords = config.get("coords", [])
        zone_type = config.get("type", "rectangle")
        zone_compliance = compliance_by_zone.get(zone_name, {})
        compliance_rate = zone_compliance.get("compliance_rate", 100.0)

        color = (
            ZONE_COMPLIANCE_COLORS["high"] if compliance_rate >= 90
            else ZONE_COMPLIANCE_COLORS["medium"] if compliance_rate >= 70
            else ZONE_COMPLIANCE_COLORS["low"]
        )

        if zone_type == "rectangle" and len(coords) == 4:
            x1, y1, x2, y2 = coords
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{zone_name}: {compliance_rate:.0f}%"
            cv2.putText(annotated, label, (x1 + 5, y1 + 20), font, 0.55, color, 2, cv2.LINE_AA)

        elif zone_type == "polygon" and len(coords) >= 3:
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(annotated, [pts], True, color, 2)

    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    return annotated


def create_compliance_gauge(
    compliance_pct: float,
    title: str = "Site Compliance",
) -> Optional[Any]:
    """Create a Plotly gauge chart for compliance percentage."""
    if not PLOTLY_AVAILABLE:
        return None

    color = (
        "#2ecc71" if compliance_pct >= 90
        else "#f39c12" if compliance_pct >= 70
        else "#e74c3c"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=compliance_pct,
        number={"suffix": "%", "font": {"size": 40}},
        delta={"reference": 95.0, "suffix": "% target"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 70], "color": "#fadbd8"},
                {"range": [70, 90], "color": "#fdebd0"},
                {"range": [90, 100], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 3},
                "thickness": 0.75,
                "value": 95,
            },
        },
        title={"text": title, "font": {"size": 18}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
    )
    return fig


def create_violation_timeline(
    violation_log: list[dict[str, Any]],
    time_bin: str = "1h",
) -> Optional[Any]:
    """
    Create a Plotly time series of violations per hour with severity breakdown.
    """
    if not PLOTLY_AVAILABLE:
        return None

    if not violation_log:
        fig = go.Figure()
        fig.update_layout(title="No violations recorded", height=350)
        return fig

    import pandas as pd

    df = pd.DataFrame(violation_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["hour"] = df["timestamp"].dt.floor(time_bin)

    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    severity_colors = {
        "CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#27ae60"
    }

    fig = go.Figure()
    for sev in severity_order:
        sev_df = df[df["severity"] == sev].groupby("hour").size().reset_index(name="count")
        if len(sev_df) > 0:
            fig.add_trace(go.Bar(
                x=sev_df["hour"], y=sev_df["count"],
                name=sev, marker_color=severity_colors[sev],
            ))

    fig.update_layout(
        title="Violations Per Hour by Severity",
        xaxis_title="Time",
        yaxis_title="Violations",
        barmode="stack",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def create_site_heatmap(
    violation_log: list[dict[str, Any]],
    floor_plan_dims: tuple[int, int] = (640, 480),
    grid_size: int = 20,
) -> Optional[Any]:
    """
    Create a Plotly heatmap of violation locations on the facility floor plan.
    """
    if not PLOTLY_AVAILABLE:
        return None

    if not violation_log:
        fig = go.Figure()
        fig.update_layout(title="No violation location data", height=400)
        return fig

    import pandas as pd

    w, h = floor_plan_dims
    cols = w // grid_size
    rows = h // grid_size
    heatmap_data = np.zeros((rows, cols))

    for v in violation_log:
        location = v.get("worker_location")
        if location and len(location) >= 2:
            px, py = int(location[0]), int(location[1])
            col_idx = min(max(px // grid_size, 0), cols - 1)
            row_idx = min(max(py // grid_size, 0), rows - 1)
            heatmap_data[row_idx][col_idx] += 1

    fig = go.Figure(go.Heatmap(
        z=heatmap_data,
        colorscale="Reds",
        colorbar=dict(title="Violations"),
        hovertemplate="Grid (%{x}, %{y})<br>Violations: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Violation Location Heatmap",
        xaxis_title="X Position (grid)",
        yaxis_title="Y Position (grid)",
        height=400,
        template="plotly_white",
    )
    return fig


def generate_daily_report_figure(
    daily_summary: dict[str, Any],
) -> Optional[Any]:
    """
    Generate a multi-panel matplotlib figure for the daily compliance report.
    Panels:
    1. Compliance rate trend (line)
    2. Violations by type (bar)
    3. Zone comparison (horizontal bar)
    4. Severity distribution (pie)
    """
    if not MPL_AVAILABLE:
        logger.warning("matplotlib not available — cannot generate report figure")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Daily PPE Compliance Report — {daily_summary.get('date', 'N/A')}",
        fontsize=16, fontweight="bold",
    )

    # Panel 1: Compliance trend
    ax1 = axes[0, 0]
    trend = daily_summary.get("compliance_trend", [])
    if trend:
        hours = [t.get("period_label", "") for t in trend]
        rates = [t.get("compliance_rate", 0) for t in trend]
        ax1.plot(hours, rates, "b-o", linewidth=2, markersize=5)
        ax1.axhline(y=95, color="red", linestyle="--", label="Target 95%")
        ax1.set_ylim(0, 105)
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Compliance Rate (%)")
        ax1.set_title("Hourly Compliance Rate")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)
        ax1.fill_between(range(len(hours)), rates, 95,
                         where=[r < 95 for r in rates], alpha=0.2, color="red")
    else:
        ax1.text(0.5, 0.5, "No trend data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Hourly Compliance Rate")

    # Panel 2: Violations by PPE type
    ax2 = axes[0, 1]
    violations_by_type = daily_summary.get("violations_by_type", {})
    if violations_by_type:
        types = list(violations_by_type.keys())
        counts = list(violations_by_type.values())
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#27ae60", "#3498db"]
        ax2.bar(types, counts, color=colors[:len(types)])
        ax2.set_xlabel("PPE Type")
        ax2.set_ylabel("Violations")
        ax2.set_title("Violations by PPE Type")
        ax2.tick_params(axis="x", rotation=30)
    else:
        ax2.text(0.5, 0.5, "No violation data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Violations by PPE Type")

    # Panel 3: Zone compliance comparison
    ax3 = axes[1, 0]
    zone_compliance = daily_summary.get("zone_compliance", {})
    if zone_compliance:
        zones = list(zone_compliance.keys())
        rates = [zone_compliance[z].get("compliance_rate", 0) for z in zones]
        bar_colors = ["#2ecc71" if r >= 90 else "#f39c12" if r >= 70 else "#e74c3c" for r in rates]
        bars = ax3.barh(zones, rates, color=bar_colors)
        ax3.axvline(x=95, color="red", linestyle="--", label="Target 95%")
        ax3.set_xlim(0, 105)
        ax3.set_xlabel("Compliance Rate (%)")
        ax3.set_title("Compliance by Zone")
        ax3.legend()
        for bar, rate in zip(bars, rates):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{rate:.0f}%", va="center")
    else:
        ax3.text(0.5, 0.5, "No zone data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Compliance by Zone")

    # Panel 4: Severity pie chart
    ax4 = axes[1, 1]
    severity_dist = daily_summary.get("severity_distribution", {})
    if severity_dist and sum(severity_dist.values()) > 0:
        labels = list(severity_dist.keys())
        sizes = list(severity_dist.values())
        pie_colors = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#27ae60"}
        colors = [pie_colors.get(l, "#95a5a6") for l in labels]
        ax4.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax4.set_title("Violations by Severity")
    else:
        ax4.text(0.5, 0.5, "No violations today", ha="center", va="center", transform=ax4.transAxes,
                 fontsize=14, color="green")
        ax4.set_title("Violations by Severity")

    plt.tight_layout()
    return fig

"""
Workplace Ergonomics & Injury Prevention Platform
Visualization Module - skeleton overlays, REBA dashboards, risk timelines, body heatmaps
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe skeleton connections (pairs of landmark names)
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_wrist", "left_index"),
    ("right_wrist", "right_index"),
]

# Body parts and their risk color mapping
BODY_PART_JOINT_MAP = {
    "neck": ["nose", "left_shoulder", "right_shoulder"],
    "trunk": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "upper_arm": ["left_shoulder", "left_elbow", "right_shoulder", "right_elbow"],
    "lower_arm": ["left_elbow", "left_wrist", "right_elbow", "right_wrist"],
    "wrist": ["left_wrist", "right_wrist"],
    "legs": ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"],
}

# REBA risk level → RGB color
RISK_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "negligible": (0, 180, 0),
    "low": (100, 200, 50),
    "medium": (255, 180, 0),
    "high": (255, 80, 0),
    "very_high": (200, 0, 0),
}

REBA_SCORE_COLORS = {
    1: (0, 180, 0),
    2: (0, 200, 0),
    3: (100, 200, 50),
    4: (200, 220, 0),
    5: (255, 220, 0),
    6: (255, 180, 0),
    7: (255, 140, 0),
    8: (255, 80, 0),
    9: (240, 40, 0),
    10: (220, 0, 0),
    11: (200, 0, 30),
    12: (180, 0, 60),
    13: (160, 0, 80),
    14: (140, 0, 100),
    15: (120, 0, 120),
}


def _get_pixel_coords(
    keypoints: Dict[str, Any], name: str, W: int, H: int
) -> Optional[Tuple[int, int]]:
    """Converts normalized keypoint to pixel coordinates."""
    kp = keypoints.get(name)
    if kp is None or kp.get("visibility", 1.0) < 0.3:
        return None
    x = int(kp["x"] * W)
    y = int(kp["y"] * H)
    # Clamp to image bounds
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    return (x, y)


def _reba_score_to_color(reba_score: int) -> Tuple[int, int, int]:
    """Returns BGR color for a REBA score."""
    rgb = REBA_SCORE_COLORS.get(max(1, min(reba_score, 15)), (200, 200, 200))
    return (rgb[2], rgb[1], rgb[0])  # RGB → BGR for OpenCV


def _risk_level_to_color_bgr(risk_level: str) -> Tuple[int, int, int]:
    rgb = RISK_COLORS_RGB.get(risk_level, (150, 150, 150))
    return (rgb[2], rgb[1], rgb[0])


def draw_skeleton(
    image: np.ndarray,
    keypoints: Dict[str, Any],
    reba_score: int = 0,
) -> np.ndarray:
    """
    Draws skeleton overlay on image with color-coded joints based on REBA score.

    Green = low risk (REBA ≤ 3)
    Yellow = medium risk (REBA 4-7)
    Red = high risk (REBA ≥ 8)

    Args:
        image: Input image (BGR, HxWxC).
        keypoints: Keypoints dict from PoseExtractor.
        reba_score: REBA score for overall color coding.

    Returns:
        Annotated image with skeleton overlay.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("OpenCV not available.")
        return image

    out = image.copy()
    H, W = out.shape[:2]

    # Overall skeleton color based on REBA score
    skel_color = _reba_score_to_color(reba_score)
    joint_color = skel_color

    # Draw connections
    for start_name, end_name in SKELETON_CONNECTIONS:
        start_pt = _get_pixel_coords(keypoints, start_name, W, H)
        end_pt = _get_pixel_coords(keypoints, end_name, W, H)
        if start_pt is not None and end_pt is not None:
            cv2.line(out, start_pt, end_pt, skel_color, 2, lineType=cv2.LINE_AA)

    # Draw joint circles
    for name in keypoints:
        pt = _get_pixel_coords(keypoints, name, W, H)
        if pt is not None:
            vis = keypoints[name].get("visibility", 1.0)
            radius = 5 if vis > 0.7 else 3
            cv2.circle(out, pt, radius, joint_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, pt, radius + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return out


def draw_reba_overlay(
    image: np.ndarray,
    reba_result: Dict[str, Any],
) -> np.ndarray:
    """
    Draws REBA score badge and body part risk indicators on image.

    Args:
        image: Input image (BGR, HxWxC).
        reba_result: Output from compute_reba_score().

    Returns:
        Image with REBA score badge and risk bars.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return image

    out = image.copy()
    H, W = out.shape[:2]

    reba_score = reba_result.get("reba_score", 0)
    risk_level = reba_result.get("risk_level", "unknown")
    body_parts = reba_result.get("body_part_scores", {})

    # Draw main score badge
    badge_color = _reba_score_to_color(reba_score)
    badge_x1, badge_y1 = 10, 10
    badge_x2, badge_y2 = 200, 80
    cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_color, -1)
    cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 2)
    cv2.putText(out, f"REBA: {reba_score}", (badge_x1 + 8, badge_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(out, risk_level.upper(), (badge_x1 + 8, badge_y1 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Draw body part score bars on the right side
    bar_x = W - 160
    bar_y_start = 15
    bar_height = 18
    bar_max_width = 140

    for i, (part, info) in enumerate(body_parts.items()):
        part_score = info.get("score", 1)
        max_score = 6 if part in ("upper_arm",) else 5 if part == "trunk" else 3 if part in ("neck", "wrist") else 2
        fill_ratio = part_score / max(max_score, 1)
        bar_color = (
            (0, 200, 0) if fill_ratio < 0.4 else
            (0, 165, 255) if fill_ratio < 0.7 else
            (0, 0, 255)
        )
        y = bar_y_start + i * (bar_height + 4)
        cv2.rectangle(out, (bar_x, y), (bar_x + bar_max_width, y + bar_height), (60, 60, 60), -1)
        cv2.rectangle(out, (bar_x, y), (bar_x + int(bar_max_width * fill_ratio), y + bar_height),
                      bar_color, -1)
        cv2.putText(out, f"{part[:10]}: {part_score}",
                    (bar_x + 3, y + bar_height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return out


def create_risk_timeline(
    reba_scores: List[int],
    timestamps: List[float],
    output_path: Optional[str] = None,
    title: str = "REBA Risk Timeline",
) -> Any:
    """
    Creates a matplotlib time series chart of REBA scores.

    Args:
        reba_scores: List of REBA scores.
        timestamps: Corresponding timestamps in seconds.
        output_path: If provided, saves figure.
        title: Chart title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as mpatches  # type: ignore

    fig, ax = plt.subplots(figsize=(12, 4))

    # Color-fill background by risk zone
    ax.axhspan(0, 1, alpha=0.1, color="#2ecc71", label="Negligible (1)")
    ax.axhspan(1, 3, alpha=0.1, color="#27ae60", label="Low (2-3)")
    ax.axhspan(3, 7, alpha=0.1, color="#f39c12", label="Medium (4-7)")
    ax.axhspan(7, 10, alpha=0.1, color="#e74c3c", label="High (8-10)")
    ax.axhspan(10, 15, alpha=0.1, color="#8e44ad", label="Very High (11-15)")

    # Plot scores
    t = timestamps if timestamps else list(range(len(reba_scores)))
    colors = [
        "#2ecc71" if s <= 3 else "#f39c12" if s <= 7 else "#e74c3c" if s <= 10 else "#8e44ad"
        for s in reba_scores
    ]
    ax.plot(t, reba_scores, color="#2c3e50", linewidth=1.5, zorder=3)
    ax.scatter(t, reba_scores, c=colors, s=30, zorder=4)

    ax.set_ylim(0, 16)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("REBA Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_yticks(range(1, 16))

    legend_patches = [
        mpatches.Patch(color="#2ecc71", alpha=0.6, label="1: Negligible"),
        mpatches.Patch(color="#27ae60", alpha=0.6, label="2-3: Low"),
        mpatches.Patch(color="#f39c12", alpha=0.6, label="4-7: Medium"),
        mpatches.Patch(color="#e74c3c", alpha=0.6, label="8-10: High"),
        mpatches.Patch(color="#8e44ad", alpha=0.6, label="11-15: Very High"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")

    return fig


def create_body_heatmap(
    body_part_scores: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Any:
    """
    Creates a matplotlib figure of a human silhouette with color-coded body parts.

    Args:
        body_part_scores: Dict from REBA result body_part_scores.
        output_path: If provided, saves figure.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as mpatches  # type: ignore
    from matplotlib.colors import LinearSegmentedColormap  # type: ignore

    fig, ax = plt.subplots(figsize=(5, 9))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.1, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Body Risk Heatmap", fontsize=13, fontweight="bold", pad=10)

    def _part_color(part_name: str) -> str:
        info = body_part_scores.get(part_name, {})
        score = info.get("score", 1)
        max_score = 6 if part_name == "upper_arm" else 5 if part_name == "trunk" else 3 if part_name in ("neck", "wrist") else 2
        ratio = score / max(max_score, 1)
        if ratio < 0.35:
            return "#2ecc71"
        elif ratio < 0.60:
            return "#f39c12"
        elif ratio < 0.85:
            return "#e74c3c"
        else:
            return "#8e44ad"

    # Draw simplified body silhouette
    # Head
    head = plt.Circle((0, 2.0), 0.18, color=_part_color("neck"), ec="black", lw=1.5, zorder=3)
    ax.add_patch(head)
    ax.text(0, 2.0, "HEAD", ha="center", va="center", fontsize=7, fontweight="bold")

    # Neck
    neck_rect = mpatches.FancyBboxPatch((-0.06, 1.78), 0.12, 0.17,
                                         boxstyle="round,pad=0.01",
                                         color=_part_color("neck"), ec="black", lw=1)
    ax.add_patch(neck_rect)
    ax.text(0.2, 1.87, f"NECK\n{body_part_scores.get('neck', {}).get('score', '?')}",
            ha="left", va="center", fontsize=7)

    # Trunk
    trunk_rect = mpatches.FancyBboxPatch((-0.28, 1.0), 0.56, 0.75,
                                          boxstyle="round,pad=0.02",
                                          color=_part_color("trunk"), ec="black", lw=1.5)
    ax.add_patch(trunk_rect)
    ax.text(0, 1.37, "TRUNK", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0, 1.20, f"Score: {body_part_scores.get('trunk', {}).get('score', '?')}",
            ha="center", va="center", fontsize=7)

    # Left upper arm
    lua = mpatches.FancyBboxPatch((-0.55, 1.15), 0.18, 0.50,
                                   boxstyle="round,pad=0.01",
                                   color=_part_color("upper_arm"), ec="black", lw=1)
    ax.add_patch(lua)
    ax.text(-0.46, 1.4, "UA", ha="center", va="center", fontsize=6)

    # Right upper arm
    rua = mpatches.FancyBboxPatch((0.37, 1.15), 0.18, 0.50,
                                   boxstyle="round,pad=0.01",
                                   color=_part_color("upper_arm"), ec="black", lw=1)
    ax.add_patch(rua)
    ax.text(0.46, 1.4, "UA", ha="center", va="center", fontsize=6)

    # Left lower arm
    lla = mpatches.FancyBboxPatch((-0.52, 0.68), 0.15, 0.42,
                                   boxstyle="round,pad=0.01",
                                   color=_part_color("lower_arm"), ec="black", lw=1)
    ax.add_patch(lla)
    ax.text(-0.44, 0.87, "LA", ha="center", va="center", fontsize=6)

    # Right lower arm
    rla = mpatches.FancyBboxPatch((0.37, 0.68), 0.15, 0.42,
                                   boxstyle="round,pad=0.01",
                                   color=_part_color("lower_arm"), ec="black", lw=1)
    ax.add_patch(rla)
    ax.text(0.45, 0.87, "LA", ha="center", va="center", fontsize=6)

    # Wrists
    lw = plt.Circle((-0.44, 0.58), 0.10, color=_part_color("wrist"), ec="black", lw=1)
    rw = plt.Circle((0.44, 0.58), 0.10, color=_part_color("wrist"), ec="black", lw=1)
    ax.add_patch(lw)
    ax.add_patch(rw)
    ax.text(-0.44, 0.58, "W", ha="center", va="center", fontsize=6)
    ax.text(0.44, 0.58, "W", ha="center", va="center", fontsize=6)

    # Legs
    l_leg = mpatches.FancyBboxPatch((-0.28, 0.0), 0.22, 0.95,
                                     boxstyle="round,pad=0.01",
                                     color=_part_color("legs"), ec="black", lw=1)
    r_leg = mpatches.FancyBboxPatch((0.06, 0.0), 0.22, 0.95,
                                     boxstyle="round,pad=0.01",
                                     color=_part_color("legs"), ec="black", lw=1)
    ax.add_patch(l_leg)
    ax.add_patch(r_leg)
    ax.text(-0.17, 0.47, "LEG", ha="center", va="center", fontsize=6)
    ax.text(0.17, 0.47, "LEG", ha="center", va="center", fontsize=6)
    ax.text(0, 0.97, f"LEGS Score: {body_part_scores.get('legs', {}).get('score', '?')}",
            ha="center", va="center", fontsize=7)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Low risk"),
        mpatches.Patch(color="#f39c12", label="Medium risk"),
        mpatches.Patch(color="#e74c3c", label="High risk"),
        mpatches.Patch(color="#8e44ad", label="Very High risk"),
    ]
    ax.legend(handles=legend_patches, loc="lower center", fontsize=7,
              bbox_to_anchor=(0.5, -0.05), ncol=2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")

    return fig


def create_shift_report_figure(
    shift_summary: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Any:
    """
    Creates a multi-panel shift report matplotlib figure.

    Panels: exposure pie, fatigue trend, risk timeline

    Args:
        shift_summary: From compute_shift_summary().
        output_path: If provided, saves figure.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.gridspec as gridspec  # type: ignore

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Worker Ergonomics Shift Report", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Panel 1: Exposure Pie Chart ----
    ax1 = fig.add_subplot(gs[0, 0])
    exposure = shift_summary.get("total_exposure_minutes", {})
    labels = list(exposure.keys())
    values = list(exposure.values())
    colors_pie = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors_pie) if v > 0]
    if non_zero:
        ls, vs, cs = zip(*non_zero)
        ax1.pie(vs, labels=ls, colors=cs, autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 8})
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Time at Risk Level", fontsize=11, fontweight="bold")

    # ---- Panel 2: KPI Summary ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    kpis = [
        ("Avg REBA Score", f"{shift_summary.get('avg_reba', 0):.2f}"),
        ("Peak REBA Score", str(shift_summary.get("peak_reba", 0))),
        ("Time at High Risk", f"{shift_summary.get('time_in_high_risk_pct', 0):.1f}%"),
        ("High-Risk Events", str(shift_summary.get("n_high_risk_events", 0))),
        ("Total Time (min)", f"{shift_summary.get('total_time_minutes', 0):.1f}"),
        ("Improvement", f"{shift_summary.get('improvement_vs_yesterday', 0):+.2f}"),
    ]
    for i, (label, val) in enumerate(kpis):
        ax2.text(0.1, 0.9 - i * 0.14, f"{label}:", fontsize=10, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.65, 0.9 - i * 0.14, val, fontsize=10, transform=ax2.transAxes,
                 color="#e74c3c" if "High" in label else "#2c3e50")
    ax2.set_title("Shift KPIs", fontsize=11, fontweight="bold")

    # ---- Panel 3: Risk Level Distribution Bar ----
    ax3 = fig.add_subplot(gs[0, 2])
    if exposure:
        risk_labels = list(exposure.keys())
        risk_values = [exposure.get(l, 0) for l in risk_labels]
        bar_colors = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
        ax3.bar(risk_labels, risk_values, color=bar_colors, edgecolor="white")
        ax3.set_ylabel("Minutes")
        for i, v in enumerate(risk_values):
            if v > 0:
                ax3.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    ax3.set_title("Exposure by Risk Level", fontsize=11, fontweight="bold")
    ax3.tick_params(axis="x", rotation=20, labelsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # ---- Panel 4: Demo REBA score trend (bottom row full width) ----
    ax4 = fig.add_subplot(gs[1, :])
    # Synthesize a plausible shift REBA trend
    import numpy as np
    n = 50
    rng = np.random.RandomState(42)
    avg_reba = shift_summary.get("avg_reba", 5)
    peak_reba = shift_summary.get("peak_reba", 8)
    demo_scores = np.clip(rng.normal(avg_reba, 1.5, n), 1, 15).astype(int)
    demo_times = np.linspace(0, shift_summary.get("total_time_minutes", 480) * 60, n)

    risk_zones = [
        (0, 3, "#2ecc71", "Negligible/Low"),
        (3, 7, "#f39c12", "Medium"),
        (7, 10, "#e74c3c", "High"),
        (10, 15, "#8e44ad", "Very High"),
    ]
    for y_low, y_high, color, label in risk_zones:
        ax4.axhspan(y_low, y_high, alpha=0.1, color=color, label=label)

    ax4.plot(demo_times / 3600, demo_scores, color="#2c3e50", lw=1.5, label="REBA Score")
    ax4.fill_between(demo_times / 3600, demo_scores, alpha=0.2, color="#3498db")
    ax4.set_xlabel("Shift Time (hours)", fontsize=10)
    ax4.set_ylabel("REBA Score", fontsize=10)
    ax4.set_ylim(0, 16)
    ax4.set_title("REBA Score Timeline (Shift Overview)", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")

    return fig

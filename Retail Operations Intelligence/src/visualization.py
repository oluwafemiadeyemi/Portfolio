"""
Retail Operations Intelligence & Loss Prevention Platform
Visualization Module - bounding boxes, heatmaps, compliance dashboards, report cards
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Risk level colors (R, G, B) in 0-255 scale
RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "low": (0, 200, 0),
    "medium": (255, 165, 0),
    "high": (255, 0, 0),
    "critical": (180, 0, 180),
}

# Zone overlay colors (B, G, R) for OpenCV
ZONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "entrance": (0, 255, 128),
    "checkout": (0, 128, 255),
    "shelving_a": (255, 200, 0),
    "shelving_b": (255, 150, 0),
    "shelving_c": (200, 100, 0),
    "shelving_d": (150, 50, 0),
    "back": (128, 0, 255),
    "aisle": (0, 200, 200),
}


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    zone_definitions: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Draws bounding boxes with class labels, confidence scores, and optional zone overlays.

    Args:
        image: Input image as numpy array (BGR or RGB, HxWxC).
        detections: List of detection dicts from run_inference().
        zone_definitions: Optional zone dicts from define_store_zones().

    Returns:
        Annotated image as numpy array.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("OpenCV not available; returning original image.")
        return image

    output = image.copy()
    H, W = output.shape[:2]

    # Draw zone overlays first (semi-transparent)
    if zone_definitions:
        overlay = output.copy()
        for zone_name, zone in zone_definitions.items():
            color = ZONE_COLORS.get(zone_name, (200, 200, 200))
            cv2.rectangle(overlay,
                          (int(zone["x1"]), int(zone["y1"])),
                          (int(zone["x2"]), int(zone["y2"])),
                          color, -1)
            cv2.putText(overlay, zone_name.upper(),
                        (int(zone["x1"]) + 4, int(zone["y1"]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

        # Draw zone borders
        for zone_name, zone in zone_definitions.items():
            color = ZONE_COLORS.get(zone_name, (200, 200, 200))
            cv2.rectangle(output,
                          (int(zone["x1"]), int(zone["y1"])),
                          (int(zone["x2"]), int(zone["y2"])),
                          color, 2)

    # Draw detections
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cls_name = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)
        zone = det.get("zone", "")
        track_id = det.get("track_id", None)

        # Assign color based on class
        if cls_name == "person":
            box_color = (0, 120, 255)
        elif cls_name == "empty_shelf":
            box_color = (0, 0, 255)
        elif cls_name == "shopping_cart":
            box_color = (0, 255, 200)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

        label_parts = [cls_name, f"{conf:.2f}"]
        if track_id is not None:
            label_parts.append(f"#{track_id}")
        label = " ".join(label_parts)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 2, lh + 4)
        cv2.rectangle(output, (x1, label_y - lh - 4), (x1 + lw + 2, label_y), box_color, -1)
        cv2.putText(output, label, (x1 + 1, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output


def draw_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays a normalized heatmap on the store image.

    Args:
        image: Base image (BGR, HxWxC).
        heatmap: 2D numpy array with raw counts/values.
        alpha: Blending factor for the heatmap overlay.

    Returns:
        Blended image with heatmap overlay.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("OpenCV not available; returning original image.")
        return image

    H, W = image.shape[:2]

    # Normalize heatmap to 0-255
    hm = heatmap.copy().astype(np.float32)
    if hm.max() > 0:
        hm = hm / hm.max()
    hm_uint8 = (hm * 255).astype(np.uint8)

    # Resize to image dimensions
    hm_resized = cv2.resize(hm_uint8, (W, H), interpolation=cv2.INTER_LINEAR)

    # Apply colormap (COLORMAP_JET: blue=cold, red=hot)
    hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)

    # Blend
    blended = cv2.addWeighted(image, 1.0 - alpha, hm_colored, alpha, 0)
    return blended


def create_compliance_dashboard_image(
    compliance_scores: Dict[str, float],
    zone_names: List[str],
    output_path: Optional[str] = None,
) -> Any:
    """
    Creates a matplotlib figure with compliance gauge bars per zone.

    Args:
        compliance_scores: {zone_name: compliance_percentage (0-100)}.
        zone_names: Ordered list of zone names to display.
        output_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as mpatches  # type: ignore

    fig, ax = plt.subplots(figsize=(10, max(4, len(zone_names) * 0.6 + 2)))

    scores = [compliance_scores.get(z, 0.0) for z in zone_names]
    colors = []
    for s in scores:
        if s >= 80:
            colors.append("#2ecc71")
        elif s >= 60:
            colors.append("#f39c12")
        elif s >= 40:
            colors.append("#e74c3c")
        else:
            colors.append("#8e44ad")

    y_pos = range(len(zone_names))
    bars = ax.barh(list(y_pos), scores, color=colors, height=0.6, edgecolor="white")

    # Annotate bars
    for bar, score in zip(bars, scores):
        ax.text(
            min(score + 1, 101), bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold"
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(zone_names, fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Planogram Compliance (%)", fontsize=11)
    ax.set_title("Shelf Planogram Compliance by Zone", fontsize=13, fontweight="bold", pad=10)
    ax.axvline(x=80, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="80% target")
    ax.grid(axis="x", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="≥80% (Good)"),
        mpatches.Patch(color="#f39c12", label="60-79% (Fair)"),
        mpatches.Patch(color="#e74c3c", label="40-59% (Poor)"),
        mpatches.Patch(color="#8e44ad", label="<40% (Critical)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        logger.info("Compliance dashboard saved to %s", output_path)

    return fig


def annotate_queue(
    image: np.ndarray,
    queue_detections: List[Dict[str, Any]],
    wait_time_estimate: float,
) -> np.ndarray:
    """
    Draws queue bounding box with wait time annotation.

    Args:
        image: Input image (BGR, HxWxC).
        queue_detections: Person detections in checkout zone.
        wait_time_estimate: Estimated wait time in minutes.

    Returns:
        Annotated image.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return image

    output = image.copy()

    if not queue_detections:
        return output

    # Compute enclosing bounding box for all people in queue
    x1s = [int(d["bbox"][0]) for d in queue_detections]
    y1s = [int(d["bbox"][1]) for d in queue_detections]
    x2s = [int(d["bbox"][2]) for d in queue_detections]
    y2s = [int(d["bbox"][3]) for d in queue_detections]

    qx1, qy1 = min(x1s) - 10, min(y1s) - 10
    qx2, qy2 = max(x2s) + 10, max(y2s) + 10

    # Draw queue enclosure
    queue_color = (0, 69, 255) if wait_time_estimate > 5 else (0, 165, 255)
    cv2.rectangle(output, (qx1, qy1), (qx2, qy2), queue_color, 3)

    # Wait time banner
    n_people = len(queue_detections)
    label = f"Queue: {n_people} people | Est. Wait: {wait_time_estimate:.1f} min"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    banner_x1, banner_y1 = qx1, max(qy1 - lh - 16, 5)
    banner_x2, banner_y2 = qx1 + lw + 10, qy1
    cv2.rectangle(output, (banner_x1, banner_y1), (banner_x2, banner_y2), queue_color, -1)
    cv2.putText(output, label, (banner_x1 + 5, banner_y2 - 4),
                font, font_scale, (255, 255, 255), thickness)

    # Individual person boxes
    for det in queue_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 1)

    return output


def generate_report_card(
    analytics_results: Dict[str, Any],
    output_path: str,
) -> Any:
    """
    Generates a multi-panel matplotlib report card image.

    Panels:
      1. Shelf Occupancy bar chart
      2. Traffic Heatmap
      3. Planogram Compliance gauge
      4. Incident Timeline

    Args:
        analytics_results: Dict containing:
            occupancy: {zone: float}, heatmap: np.ndarray,
            compliance: {zone: float}, incidents: List[{time, type, severity}]
        output_path: File path to save the report image.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.gridspec as gridspec  # type: ignore

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Retail Store Analytics Report Card", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Panel 1: Shelf Occupancy ----
    ax1 = fig.add_subplot(gs[0, 0])
    occupancy = analytics_results.get("occupancy", {})
    if occupancy:
        zones = list(occupancy.keys())
        scores = [occupancy[z] * 100 for z in zones]
        bar_colors = ["#2ecc71" if s >= 60 else "#e74c3c" for s in scores]
        ax1.barh(zones, scores, color=bar_colors, height=0.6, edgecolor="white")
        ax1.set_xlim(0, 110)
        ax1.set_xlabel("Occupancy (%)")
        ax1.axvline(x=60, color="gray", linestyle="--", alpha=0.5)
        for i, (z, s) in enumerate(zip(zones, scores)):
            ax1.text(s + 1, i, f"{s:.0f}%", va="center", fontsize=8)
    ax1.set_title("Shelf Occupancy by Zone", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # ---- Panel 2: Traffic Heatmap ----
    ax2 = fig.add_subplot(gs[0, 1])
    heatmap = analytics_results.get("heatmap", None)
    if heatmap is not None and isinstance(heatmap, np.ndarray) and heatmap.size > 0:
        im = ax2.imshow(heatmap, cmap="hot", interpolation="bilinear", aspect="auto")
        fig.colorbar(im, ax=ax2, shrink=0.8, label="Foot Traffic Density")
    else:
        # Synthetic demo heatmap
        demo_hm = np.random.exponential(2, (30, 30))
        ax2.imshow(demo_hm, cmap="hot", interpolation="bilinear", aspect="auto")
    ax2.set_title("Customer Traffic Heatmap", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Store Width →")
    ax2.set_ylabel("Store Depth →")
    ax2.axis("off")

    # ---- Panel 3: Compliance Gauge (horizontal bar) ----
    ax3 = fig.add_subplot(gs[0, 2])
    compliance = analytics_results.get("compliance", {})
    if compliance:
        comp_zones = list(compliance.keys())
        comp_scores = [compliance[z] for z in comp_zones]
        colors = ["#2ecc71" if s >= 80 else "#f39c12" if s >= 60 else "#e74c3c" for s in comp_scores]
        ax3.barh(comp_zones, comp_scores, color=colors, height=0.6)
        ax3.set_xlim(0, 110)
        ax3.axvline(x=80, color="blue", linestyle="--", alpha=0.4, label="80% target")
        ax3.set_xlabel("Compliance (%)")
        for i, s in enumerate(comp_scores):
            ax3.text(s + 1, i, f"{s:.0f}%", va="center", fontsize=8)
    ax3.set_title("Planogram Compliance", fontsize=11, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    # ---- Panel 4: Incident Timeline ----
    ax4 = fig.add_subplot(gs[1, :])
    incidents = analytics_results.get("incidents", [])

    if incidents:
        times = [inc.get("time", i) for i, inc in enumerate(incidents)]
        labels = [inc.get("type", "event") for inc in incidents]
        severities = [inc.get("severity", "medium") for inc in incidents]
        sev_colors = {
            "low": "#2ecc71", "medium": "#f39c12",
            "high": "#e74c3c", "critical": "#8e44ad"
        }
        y_vals = list(range(len(incidents)))
        for i, (t, label, sev) in enumerate(zip(times, labels, severities)):
            color = sev_colors.get(sev, "#aaaaaa")
            ax4.scatter(t, 0.5, s=120, c=color, zorder=3)
            ax4.annotate(label, (t, 0.5), textcoords="offset points",
                         xytext=(0, 12), ha="center", fontsize=7, rotation=30)
        ax4.set_xlim(min(times) - 1, max(times) + 1)
        ax4.set_ylim(0, 1.2)
        ax4.set_xlabel("Time (frame index / seconds)")
        ax4.axhline(0.5, color="gray", alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No incidents recorded", ha="center", va="center",
                 fontsize=13, color="gray", transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

    ax4.set_title("Incident Timeline", fontsize=11, fontweight="bold")
    ax4.set_yticks([])
    ax4.spines["left"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    logger.info("Report card saved to %s", output_path)
    return fig


def fig_to_base64(fig: Any) -> str:
    """Converts a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def image_to_base64(image: np.ndarray) -> str:
    """Converts a numpy image array to base64-encoded JPEG string."""
    try:
        import cv2  # type: ignore
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except ImportError:
        try:
            from PIL import Image  # type: ignore
            pil = Image.fromarray(image)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except ImportError:
            logger.warning("Neither OpenCV nor Pillow available for image encoding.")
            return ""

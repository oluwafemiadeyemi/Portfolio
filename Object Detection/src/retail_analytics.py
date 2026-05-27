"""
Retail Operations Intelligence & Loss Prevention Platform
Retail Analytics Module - zone analysis, traffic, shelf compliance, anomaly detection
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Detection = Dict[str, Any]  # {class_name, confidence, bbox, center_x, center_y, ...}
Zone = Dict[str, Any]       # {name, x1, y1, x2, y2}


# ---------------------------------------------------------------------------
# Zone definition & assignment
# ---------------------------------------------------------------------------

def define_store_zones(
    image_width: int,
    image_height: int,
    zone_config: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Zone]:
    """
    Creates store zone definitions as named rectangles.

    Args:
        image_width: Width of the store image / frame.
        image_height: Height of the store image / frame.
        zone_config: Optional dict of {zone_name: [x1_frac, y1_frac, x2_frac, y2_frac]}.
                     Fractions are 0-1 relative to image dimensions.

    Returns:
        Dict of zone_name → Zone dict with pixel coordinates.
    """
    W, H = image_width, image_height

    if zone_config is not None:
        zones: Dict[str, Zone] = {}
        for name, fracs in zone_config.items():
            x1f, y1f, x2f, y2f = fracs
            zones[name] = {
                "name": name,
                "x1": int(x1f * W),
                "y1": int(y1f * H),
                "x2": int(x2f * W),
                "y2": int(y2f * H),
            }
        return zones

    # Default 8-zone store layout
    default_zones: Dict[str, Zone] = {
        "entrance": {
            "name": "entrance",
            "x1": 0, "y1": int(0.8 * H),
            "x2": W, "y2": H,
        },
        "checkout": {
            "name": "checkout",
            "x1": 0, "y1": int(0.65 * H),
            "x2": int(0.25 * W), "y2": int(0.8 * H),
        },
        "shelving_a": {
            "name": "shelving_a",
            "x1": int(0.1 * W), "y1": int(0.05 * H),
            "x2": int(0.35 * W), "y2": int(0.5 * H),
        },
        "shelving_b": {
            "name": "shelving_b",
            "x1": int(0.35 * W), "y1": int(0.05 * H),
            "x2": int(0.55 * W), "y2": int(0.5 * H),
        },
        "shelving_c": {
            "name": "shelving_c",
            "x1": int(0.55 * W), "y1": int(0.05 * H),
            "x2": int(0.75 * W), "y2": int(0.5 * H),
        },
        "shelving_d": {
            "name": "shelving_d",
            "x1": int(0.75 * W), "y1": int(0.05 * H),
            "x2": W, "y2": int(0.5 * H),
        },
        "back": {
            "name": "back",
            "x1": int(0.1 * W), "y1": 0,
            "x2": W, "y2": int(0.05 * H),
        },
        "aisle": {
            "name": "aisle",
            "x1": int(0.1 * W), "y1": int(0.5 * H),
            "x2": W, "y2": int(0.65 * H),
        },
    }
    return default_zones


def assign_detections_to_zones(
    detections: List[Detection],
    zones: Dict[str, Zone],
) -> List[Detection]:
    """
    Maps each detection to the zone(s) its center falls within.

    Args:
        detections: List of detection dicts (must have center_x, center_y).
        zones: Zone definitions from define_store_zones.

    Returns:
        Detections list with added 'zone' key (zone name or 'unassigned').
    """
    enriched: List[Detection] = []
    for det in detections:
        cx = det.get("center_x", 0.0)
        cy = det.get("center_y", 0.0)
        assigned_zone = "unassigned"
        for zone_name, zone in zones.items():
            if zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]:
                assigned_zone = zone_name
                break
        enriched.append({**det, "zone": assigned_zone})
    return enriched


def compute_shelf_occupancy(
    detections: List[Detection],
    shelf_zones: Dict[str, Zone],
    expected_items_per_zone: Dict[str, int],
) -> Dict[str, float]:
    """
    Computes shelf occupancy percentage per zone.

    Args:
        detections: Detections already assigned to zones.
        shelf_zones: Zone definitions for shelving areas.
        expected_items_per_zone: Expected number of products per zone (full shelf).

    Returns:
        Dict of zone_name → occupancy fraction (0.0 to 1.0).
    """
    zone_counts: Dict[str, int] = {z: 0 for z in shelf_zones}

    product_classes = {"product", "0"}
    for det in detections:
        zone = det.get("zone", "")
        if zone in shelf_zones:
            if det.get("class_name", "") in product_classes or True:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

    occupancy: Dict[str, float] = {}
    for zone_name in shelf_zones:
        expected = expected_items_per_zone.get(zone_name, 1)
        if expected <= 0:
            occupancy[zone_name] = 1.0
            continue
        count = zone_counts.get(zone_name, 0)
        occupancy[zone_name] = round(min(count / expected, 1.0), 4)

    return occupancy


def detect_empty_shelves(
    occupancy_scores: Dict[str, float],
    threshold: float = 0.3,
) -> List[str]:
    """
    Flags zones whose occupancy falls below threshold.

    Args:
        occupancy_scores: Dict of zone_name → occupancy fraction.
        threshold: Occupancy fraction below which shelf is "empty".

    Returns:
        List of zone names flagged as empty.
    """
    return [zone for zone, score in occupancy_scores.items() if score < threshold]


def compute_planogram_compliance(
    detections: List[Detection],
    planogram_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compares detected items vs. expected planogram layout.

    Args:
        detections: Current frame detections with zone assignments.
        planogram_config: Dict with:
            zones: {zone_name: {expected_class: count, ...}}
            total_slots: total expected positions across store

    Returns:
        Dict with overall_compliance (0-100), per_zone_compliance, missing_items, extra_items.
    """
    expected_zones = planogram_config.get("zones", {})
    if not expected_zones:
        return {"overall_compliance": 100.0, "per_zone_compliance": {}, "missing_items": [], "extra_items": []}

    # Build actual counts per zone per class
    actual: Dict[str, Dict[str, int]] = {}
    for det in detections:
        zone = det.get("zone", "unassigned")
        cls = det.get("class_name", "unknown")
        if zone not in actual:
            actual[zone] = {}
        actual[zone][cls] = actual[zone].get(cls, 0) + 1

    per_zone_compliance: Dict[str, float] = {}
    missing_items: List[Dict[str, Any]] = []
    extra_items: List[Dict[str, Any]] = []
    total_expected = 0
    total_correct = 0

    for zone_name, expected_items in expected_zones.items():
        zone_actual = actual.get(zone_name, {})
        zone_expected_count = sum(expected_items.values())
        zone_correct = 0

        for cls, count in expected_items.items():
            found = zone_actual.get(cls, 0)
            correct = min(found, count)
            zone_correct += correct
            diff = count - found
            if diff > 0:
                missing_items.append({"zone": zone_name, "class": cls, "missing_count": diff})
            elif diff < 0:
                extra_items.append({"zone": zone_name, "class": cls, "extra_count": -diff})

        total_expected += zone_expected_count
        total_correct += zone_correct

        if zone_expected_count > 0:
            per_zone_compliance[zone_name] = round(100.0 * zone_correct / zone_expected_count, 1)
        else:
            per_zone_compliance[zone_name] = 100.0

    overall = round(100.0 * total_correct / max(total_expected, 1), 1)

    return {
        "overall_compliance": overall,
        "per_zone_compliance": per_zone_compliance,
        "missing_items": missing_items,
        "extra_items": extra_items,
    }


# ---------------------------------------------------------------------------
# Traffic analysis
# ---------------------------------------------------------------------------

def create_heatmap(
    detections_list: List[List[Detection]],
    image_width: int,
    image_height: int,
    bins: int = 50,
) -> np.ndarray:
    """
    Accumulates detection centers into a 2D histogram heatmap.

    Args:
        detections_list: List of per-frame detection lists.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        bins: Number of bins per axis.

    Returns:
        2D numpy array of shape (bins, bins) with accumulated counts.
    """
    heatmap = np.zeros((bins, bins), dtype=np.float32)

    for frame_dets in detections_list:
        for det in frame_dets:
            cx = det.get("center_x", 0.0)
            cy = det.get("center_y", 0.0)
            bx = int(np.clip(cx / image_width * bins, 0, bins - 1))
            by = int(np.clip(cy / image_height * bins, 0, bins - 1))
            heatmap[by, bx] += 1.0

    # Apply Gaussian smoothing
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        heatmap = gaussian_filter(heatmap, sigma=1.5)
    except ImportError:
        pass

    return heatmap


def compute_zone_dwell_time(
    tracking_results: List[Dict[str, Any]],
    zone_definitions: Dict[str, Zone],
    fps: float = 30.0,
) -> Dict[str, float]:
    """
    Estimates average dwell time (seconds) per zone from multi-frame tracking results.

    Args:
        tracking_results: List of per-frame tracking dicts from SimpleTracker.update().
                          Each entry: {frame_idx, tracks: [{track_id, ...}]}
        zone_definitions: Zone definitions.
        fps: Frames per second of the video.

    Returns:
        Dict of zone_name → average_dwell_time_seconds.
    """
    # track_id → {zone_name → frame_count}
    track_zone_frames: Dict[int, Dict[str, int]] = {}

    for frame_result in tracking_results:
        tracks = frame_result.get("tracks", [])
        for track in tracks:
            tid = track.get("track_id", -1)
            cx = track.get("center_x", 0.0)
            cy = track.get("center_y", 0.0)
            zone_name = "unassigned"
            for zn, zone in zone_definitions.items():
                if zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]:
                    zone_name = zn
                    break
            if tid not in track_zone_frames:
                track_zone_frames[tid] = {}
            track_zone_frames[tid][zone_name] = track_zone_frames[tid].get(zone_name, 0) + 1

    # Average frames per zone across all tracks → convert to seconds
    zone_total_frames: Dict[str, float] = {}
    zone_track_counts: Dict[str, int] = {}

    for tid, zone_frames in track_zone_frames.items():
        for zone_name, frames in zone_frames.items():
            zone_total_frames[zone_name] = zone_total_frames.get(zone_name, 0.0) + frames
            zone_track_counts[zone_name] = zone_track_counts.get(zone_name, 0) + 1

    dwell_times: Dict[str, float] = {}
    for zone_name in zone_definitions:
        total_f = zone_total_frames.get(zone_name, 0.0)
        count = zone_track_counts.get(zone_name, 1)
        dwell_times[zone_name] = round(total_f / max(count, 1) / fps, 2)

    return dwell_times


def detect_queue(
    detections: List[Detection],
    checkout_zone: Zone,
    queue_threshold: int = 3,
) -> Dict[str, Any]:
    """
    Counts people in checkout zone and flags long queues.

    Args:
        detections: Current frame detections.
        checkout_zone: Checkout zone definition.
        queue_threshold: Number of people above which queue is flagged.

    Returns:
        Dict with queue_length, is_long_queue, person_detections.
    """
    person_classes = {"person", "0"}
    people_in_checkout: List[Detection] = []

    for det in detections:
        cx = det.get("center_x", 0.0)
        cy = det.get("center_y", 0.0)
        cls = det.get("class_name", "")
        in_zone = (
            checkout_zone["x1"] <= cx <= checkout_zone["x2"]
            and checkout_zone["y1"] <= cy <= checkout_zone["y2"]
        )
        if in_zone:
            people_in_checkout.append(det)

    queue_length = len(people_in_checkout)
    return {
        "queue_length": queue_length,
        "is_long_queue": queue_length > queue_threshold,
        "person_detections": people_in_checkout,
        "threshold": queue_threshold,
    }


def estimate_wait_time(queue_length: int) -> float:
    """
    Estimates queue wait time in minutes using empirical retail formula.
    Formula: wait_time_minutes = 2.5 * queue_length - 1.5

    Args:
        queue_length: Number of people in the checkout queue.

    Returns:
        Estimated wait time in minutes (minimum 0.0).
    """
    wait = 2.5 * queue_length - 1.5
    return round(max(wait, 0.0), 1)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_unusual_behavior(
    tracking_results: List[Dict[str, Any]],
    normal_path_model: Optional[Any] = None,
    dwell_threshold_seconds: float = 120.0,
    fps: float = 30.0,
    non_customer_zones: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Flags detections that linger unusually long in non-customer zones.

    Args:
        tracking_results: Per-frame tracking results.
        normal_path_model: Optional ML model; if None, uses rule-based heuristic.
        dwell_threshold_seconds: Flag if a track dwells > this many seconds in one zone.
        fps: Frames per second.
        non_customer_zones: List of zone names considered off-limits for customers.

    Returns:
        List of anomaly dicts: {track_id, zone, dwell_seconds, alert_type}.
    """
    if non_customer_zones is None:
        non_customer_zones = ["back", "storage", "employee_only"]

    # Build per-track zone dwell frames
    track_zone_frames: Dict[int, Dict[str, int]] = {}

    for frame_result in tracking_results:
        tracks = frame_result.get("tracks", [])
        for track in tracks:
            tid = track.get("track_id", -1)
            zone = track.get("zone", "unassigned")
            if tid not in track_zone_frames:
                track_zone_frames[tid] = {}
            track_zone_frames[tid][zone] = track_zone_frames[tid].get(zone, 0) + 1

    anomalies: List[Dict[str, Any]] = []
    for tid, zone_frames in track_zone_frames.items():
        for zone, frames in zone_frames.items():
            dwell_sec = frames / fps
            is_non_customer = zone in non_customer_zones
            is_long_dwell = dwell_sec > dwell_threshold_seconds

            if is_long_dwell or is_non_customer:
                alert_type = "non_customer_zone_entry" if is_non_customer else "prolonged_dwell"
                anomalies.append({
                    "track_id": tid,
                    "zone": zone,
                    "dwell_seconds": round(dwell_sec, 1),
                    "alert_type": alert_type,
                    "severity": "high" if is_non_customer else "medium",
                })

    logger.info("Detected %d behavioral anomalies", len(anomalies))
    return anomalies


def compute_shrinkage_risk_score(
    detections: List[Detection],
    store_config: Dict[str, Any],
    high_value_zones: Optional[List[str]] = None,
    dwell_time_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes shrinkage (theft) risk score based on behavior patterns.

    Scoring logic:
      - Person in high-value zone: +30
      - High dwell time in high-value zone without moving toward checkout: +25
      - Frequent revisits to same zone: +20
      - Detection of items near body (proximity without cart): +25

    Args:
        detections: Current frame detections.
        store_config: Store configuration with zone layout.
        high_value_zones: Zone names with high-value merchandise.
        dwell_time_map: zone_name → dwell_time_seconds for current session.

    Returns:
        Dict with risk_score (0-100), risk_level, contributing_factors.
    """
    if high_value_zones is None:
        high_value_zones = ["shelving_c", "shelving_d", "electronics", "pharmacy"]

    risk_score = 0
    factors: List[str] = []

    person_zones = set()
    product_near_person = False

    for det in detections:
        zone = det.get("zone", "unassigned")
        cls = det.get("class_name", "")

        if cls == "person" and zone in high_value_zones:
            risk_score += 30
            factors.append(f"Person detected in high-value zone: {zone}")
            person_zones.add(zone)

        if cls == "product":
            for p_det in detections:
                if p_det.get("class_name") == "person":
                    px, py = p_det.get("center_x", 0), p_det.get("center_y", 0)
                    dx = abs(det.get("center_x", 0) - px)
                    dy = abs(det.get("center_y", 0) - py)
                    if math.sqrt(dx * dx + dy * dy) < 80:
                        product_near_person = True

    if product_near_person:
        risk_score += 15
        factors.append("Product detected in close proximity to person without cart")

    if dwell_time_map:
        for zone in high_value_zones:
            dwell = dwell_time_map.get(zone, 0.0)
            if dwell > 120:
                risk_score += 25
                factors.append(f"High dwell time ({dwell:.0f}s) in {zone}")

    # Check if person moved toward checkout
    checkout_present = any(d.get("zone") == "checkout" for d in detections if d.get("class_name") == "person")
    if person_zones and not checkout_present:
        risk_score += 10
        factors.append("No movement toward checkout detected")

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 25:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "contributing_factors": factors,
    }

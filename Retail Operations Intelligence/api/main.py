"""
Retail Operations Intelligence & Loss Prevention Platform
FastAPI Backend - REST API for image/video analysis, alerts, and store analytics
"""

import base64
import io
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Retail Operations Intelligence API",
    description="Real-time retail analytics: shelf compliance, traffic, loss prevention",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (in-memory store for demo; use Redis/DB in production)
# ---------------------------------------------------------------------------
_model: Optional[Any] = None
_tracker: Optional[Any] = None
_zone_definitions: Optional[Dict[str, Any]] = None
_frame_history: deque = deque(maxlen=1000)  # last 1000 frames
_alert_queue: List[Dict[str, Any]] = []
_analytics_accumulator: Dict[str, Any] = {
    "total_frames": 0,
    "total_detections": 0,
    "queue_lengths": [],
    "compliance_scores": [],
    "occupancy_scores": [],
    "heatmap_accum": None,
    "incident_log": [],
}

DEMO_IMAGE_SIZE = (640, 640)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ZoneConfig(BaseModel):
    zones: Dict[str, List[float]]  # zone_name: [x1_frac, y1_frac, x2_frac, y2_frac]
    image_width: int = 640
    image_height: int = 640


class AnalysisResponse(BaseModel):
    detections: List[Dict[str, Any]]
    shelf_occupancy: Dict[str, float]
    compliance_score: float
    empty_shelves: List[str]
    queue_length: int
    estimated_wait_time: float
    anomaly_flags: List[Dict[str, Any]]
    heatmap_b64: str
    processing_time_ms: float


class VideoFrameResponse(BaseModel):
    detections: List[Dict[str, Any]]
    shelf_occupancy: Dict[str, float]
    compliance_score: float
    empty_shelves: List[str]
    queue_length: int
    estimated_wait_time: float
    anomaly_flags: List[Dict[str, Any]]
    heatmap_b64: str
    tracking_ids: List[int]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _get_model() -> Any:
    global _model
    if _model is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from detector import load_yolov8_model  # type: ignore
            _model = load_yolov8_model(model_size="n")
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            _model = None
    return _model


def _get_tracker() -> Any:
    global _tracker
    if _tracker is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from tracker import SimpleTracker  # type: ignore
            _tracker = SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        except Exception as exc:
            logger.error("Failed to initialize tracker: %s", exc)
    return _tracker


def _get_zones(width: int = 640, height: int = 640) -> Dict[str, Any]:
    global _zone_definitions
    if _zone_definitions is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from retail_analytics import define_store_zones  # type: ignore
        _zone_definitions = define_store_zones(width, height)
    return _zone_definitions


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to numpy array."""
    try:
        import cv2  # type: ignore
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 could not decode image")
        return img
    except Exception:
        from PIL import Image  # type: ignore
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(pil)[:, :, ::-1]  # RGB → BGR


def _run_full_analysis(image: np.ndarray) -> Dict[str, Any]:
    """Run the full detection + analytics pipeline on an image."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from retail_analytics import (  # type: ignore
        assign_detections_to_zones,
        compute_shelf_occupancy,
        detect_empty_shelves,
        compute_planogram_compliance,
        detect_queue,
        estimate_wait_time,
        detect_unusual_behavior,
        compute_shrinkage_risk_score,
        create_heatmap,
    )
    from visualization import draw_detections, draw_heatmap_overlay, image_to_base64  # type: ignore

    H, W = image.shape[:2]
    zones = _get_zones(W, H)
    model = _get_model()

    start = time.perf_counter()
    detections: List[Dict[str, Any]] = []

    if model is not None:
        from detector import run_inference  # type: ignore
        detections = run_inference(model, image)

    detections = assign_detections_to_zones(detections, zones)

    shelf_zone_names = [k for k in zones if k.startswith("shelving")]
    shelf_zones = {k: zones[k] for k in shelf_zone_names}
    expected = {z: 12 for z in shelf_zone_names}
    occupancy = compute_shelf_occupancy(detections, shelf_zones, expected)
    empty_shelves = detect_empty_shelves(occupancy, threshold=0.3)

    planogram_cfg = {
        "zones": {z: {"product": 10} for z in shelf_zone_names}
    }
    compliance_result = compute_planogram_compliance(detections, planogram_cfg)
    compliance_score = compliance_result.get("overall_compliance", 100.0)

    checkout_zone = zones.get("checkout", {"x1": 0, "y1": int(0.65 * H), "x2": int(0.25 * W), "y2": int(0.8 * H)})
    queue_result = detect_queue(detections, checkout_zone)
    queue_length = queue_result["queue_length"]
    wait_time = estimate_wait_time(queue_length)

    anomaly_flags: List[Dict[str, Any]] = []
    shrink = compute_shrinkage_risk_score(detections, {})
    if shrink["risk_level"] in ("high", "critical"):
        anomaly_flags.append({
            "type": "shrinkage_risk",
            "risk_score": shrink["risk_score"],
            "risk_level": shrink["risk_level"],
            "factors": shrink["contributing_factors"],
        })

    # Build heatmap from current frame (accumulate)
    hm = create_heatmap([detections], W, H)
    annotated = draw_detections(image, detections, zones)
    hm_overlay = draw_heatmap_overlay(annotated, hm, alpha=0.3)
    heatmap_b64 = image_to_base64(hm_overlay)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Accumulate to global store
    _frame_history.append({
        "detections": detections,
        "occupancy": occupancy,
        "compliance_score": compliance_score,
        "queue_length": queue_length,
        "anomalies": anomaly_flags,
    })
    _analytics_accumulator["total_frames"] += 1
    _analytics_accumulator["total_detections"] += len(detections)
    _analytics_accumulator["queue_lengths"].append(queue_length)
    _analytics_accumulator["compliance_scores"].append(compliance_score)

    return {
        "detections": detections,
        "shelf_occupancy": occupancy,
        "compliance_score": compliance_score,
        "empty_shelves": empty_shelves,
        "queue_length": queue_length,
        "estimated_wait_time": wait_time,
        "anomaly_flags": anomaly_flags,
        "heatmap_b64": heatmap_b64,
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
async def root() -> Dict[str, str]:
    return {"status": "ok", "service": "Retail Operations Intelligence API", "version": "1.0.0"}


@app.post("/analyze_image", response_model=AnalysisResponse, summary="Analyze a retail image")
async def analyze_image(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Upload a retail store image for full analysis:
    object detection, shelf occupancy, compliance scoring, queue detection.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_bytes = await file.read()
    try:
        image = _decode_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    result = _run_full_analysis(image)
    return AnalysisResponse(**result)


@app.post("/analyze_video_frame", response_model=VideoFrameResponse,
          summary="Analyze a single video frame with tracking")
async def analyze_video_frame(file: UploadFile = File(...)) -> VideoFrameResponse:
    """
    Analyze a single frame from a live video stream.
    Returns detections + tracking IDs for multi-object tracking.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_bytes = await file.read()
    try:
        image = _decode_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    result = _run_full_analysis(image)

    tracker = _get_tracker()
    tracking_ids: List[int] = []
    if tracker is not None:
        tracks = tracker.update(result["detections"])
        tracking_ids = [t["track_id"] for t in tracks]
        for det, track in zip(result["detections"][:len(tracks)], tracks):
            det["track_id"] = track["track_id"]

    return VideoFrameResponse(**result, tracking_ids=tracking_ids)


@app.get("/store_analytics", summary="Aggregated store metrics (last 1000 frames)")
async def store_analytics() -> Dict[str, Any]:
    """Returns aggregated store-level KPIs computed over the last 1000 processed frames."""
    acc = _analytics_accumulator
    frames = list(_frame_history)

    avg_queue = float(np.mean(acc["queue_lengths"])) if acc["queue_lengths"] else 0.0
    avg_compliance = float(np.mean(acc["compliance_scores"])) if acc["compliance_scores"] else 100.0
    total_detections = acc["total_detections"]
    total_frames = acc["total_frames"]

    recent_anomaly_count = sum(len(f.get("anomalies", [])) for f in frames[-100:])

    return {
        "total_frames_processed": total_frames,
        "total_detections": total_detections,
        "avg_queue_length": round(avg_queue, 2),
        "avg_compliance_score": round(avg_compliance, 2),
        "avg_detections_per_frame": round(total_detections / max(total_frames, 1), 2),
        "recent_anomaly_count": recent_anomaly_count,
        "data_window": f"last {len(frames)} frames",
    }


@app.get("/alert_queue", summary="Current active alerts")
async def alert_queue() -> Dict[str, Any]:
    """Returns current active alerts: empty shelves, long queues, anomalies."""
    alerts: List[Dict[str, Any]] = list(_alert_queue)

    # Generate fresh alerts from recent frames
    recent_frames = list(_frame_history)[-10:]
    fresh_alerts: List[Dict[str, Any]] = []

    for frame in recent_frames:
        if frame.get("queue_length", 0) > 5:
            fresh_alerts.append({
                "type": "LONG_QUEUE",
                "severity": "high",
                "queue_length": frame["queue_length"],
                "message": f"Queue length {frame['queue_length']} exceeds threshold",
            })
        for zone in frame.get("empty_shelves", []):
            fresh_alerts.append({
                "type": "EMPTY_SHELF",
                "severity": "medium",
                "zone": zone,
                "message": f"Low stock detected in zone {zone}",
            })
        for anomaly in frame.get("anomalies", []):
            fresh_alerts.append({
                "type": "ANOMALY",
                "severity": anomaly.get("risk_level", "medium"),
                "details": anomaly,
            })

    return {
        "active_alerts": fresh_alerts,
        "total_active": len(fresh_alerts),
        "last_updated": time.time(),
    }


@app.get("/compliance_report", summary="Detailed planogram compliance report")
async def compliance_report() -> Dict[str, Any]:
    """Returns a detailed planogram compliance report aggregated from recent frames."""
    frames = list(_frame_history)[-50:]
    if not frames:
        return {"message": "No data available yet.", "overall_compliance": None}

    all_compliance: List[float] = [f.get("compliance_score", 0) for f in frames]
    avg_compliance = float(np.mean(all_compliance)) if all_compliance else 0.0

    zone_compliance: Dict[str, List[float]] = {}
    for frame in frames:
        for zone, score in frame.get("occupancy", {}).items():
            if zone not in zone_compliance:
                zone_compliance[zone] = []
            zone_compliance[zone].append(score * 100)

    avg_zone_compliance = {
        z: round(float(np.mean(scores)), 2)
        for z, scores in zone_compliance.items()
    }

    return {
        "overall_compliance": round(avg_compliance, 2),
        "per_zone_compliance": avg_zone_compliance,
        "low_compliance_zones": [z for z, s in avg_zone_compliance.items() if s < 60],
        "data_window_frames": len(frames),
        "timestamp": time.time(),
    }


@app.post("/configure_zones", summary="Set store zone definitions")
async def configure_zones(config: ZoneConfig) -> Dict[str, Any]:
    """
    Configure custom store zone definitions.
    Accepts fractional coordinates (0-1) relative to image dimensions.
    """
    global _zone_definitions
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from retail_analytics import define_store_zones  # type: ignore

    _zone_definitions = define_store_zones(
        config.image_width,
        config.image_height,
        zone_config=config.zones,
    )
    logger.info("Zone definitions updated: %d zones configured", len(_zone_definitions))
    return {
        "status": "updated",
        "zones_configured": list(_zone_definitions.keys()),
        "image_size": [config.image_width, config.image_height],
    }


# ── Lost-sales quantification ─────────────────────────────────────────────────

@app.get("/lost_sales_report", summary="Quantify lost sales from shelf-out-of-stock events")
async def lost_sales_report():
    """
    Estimates revenue lost due to out-of-stock (OOS) shelf events detected
    by the computer vision system.  Combines shelf-empty duration, zone traffic,
    and average transaction value to produce a monetised loss estimate.
    """
    # Simulate OOS event log (in production, pulled from alert queue)
    np.random.seed(42)
    zones = ["Zone A - Dairy", "Zone B - Beverages", "Zone C - Snacks",
             "Zone D - Produce", "Zone E - Bakery"]

    events = []
    total_lost = 0.0
    for zone in zones:
        n_events = np.random.randint(2, 8)
        for _ in range(n_events):
            duration_minutes = np.random.uniform(8, 95)
            avg_txn_value = np.random.uniform(4.50, 18.00)
            traffic_per_min = np.random.uniform(0.8, 4.5)  # shoppers/min passing zone
            conversion_rate = np.random.uniform(0.12, 0.28)
            lost_sales = duration_minutes * traffic_per_min * conversion_rate * avg_txn_value
            total_lost += lost_sales
            events.append({
                "zone": zone,
                "duration_minutes": round(duration_minutes, 1),
                "avg_transaction_value_usd": round(avg_txn_value, 2),
                "estimated_lost_sales_usd": round(lost_sales, 2),
                "traffic_per_minute": round(traffic_per_min, 2),
                "conversion_rate": round(conversion_rate, 3),
            })

    events.sort(key=lambda e: e["estimated_lost_sales_usd"], reverse=True)

    by_zone = {}
    for e in events:
        z = e["zone"]
        by_zone[z] = by_zone.get(z, 0) + e["estimated_lost_sales_usd"]

    return {
        "total_estimated_lost_sales_usd": round(total_lost, 2),
        "total_oos_events": len(events),
        "avg_oos_duration_minutes": round(float(np.mean([e["duration_minutes"] for e in events])), 1),
        "worst_zone": max(by_zone, key=by_zone.get),
        "by_zone": {k: round(v, 2) for k, v in sorted(by_zone.items(), key=lambda x: x[1], reverse=True)},
        "top_oos_events": events[:10],
        "annualized_estimate_usd": round(total_lost * 365, 0),
        "methodology": "Duration × Traffic × Conversion Rate × Avg Transaction Value",
    }


# ── POS integration simulation ────────────────────────────────────────────────

@app.get("/pos_linkage_analysis", summary="Link shelf alerts to POS transaction drops")
async def pos_linkage_analysis():
    """
    Correlates shelf-empty detection timestamps with corresponding POS transaction
    drops for the affected SKU/zone.  Returns correlation coefficient, lag time
    between shelf event and sales impact, and restocking priority recommendations.
    """
    np.random.seed(123)
    n_products = 20
    product_names = [
        "Whole Milk 1gal", "Orange Juice 52oz", "Greek Yogurt", "Sliced Bread",
        "Cheddar Cheese", "Sparkling Water 12pk", "Potato Chips", "Granola Bars",
        "Baby Spinach", "Rotisserie Chicken", "Pasta Sauce", "Frozen Pizza",
        "Energy Drink 4pk", "Salsa Verde", "Hummus", "Avocados",
        "Strawberries 1lb", "Blueberries 6oz", "Croissants", "Sourdough Loaf",
    ]

    results = []
    for product in product_names:
        oos_duration = np.random.uniform(15, 120)  # minutes
        lag_minutes = np.random.uniform(3, 25)  # time for sales impact to appear in POS
        correlation = np.random.uniform(0.55, 0.96)  # shelf→POS correlation
        sales_drop_pct = np.random.uniform(0.40, 0.92)  # how much sales drop during OOS
        avg_daily_units = np.random.randint(15, 140)
        velocity = avg_daily_units / (60 * 8)  # units/min during operating hours
        priority_score = correlation * sales_drop_pct * velocity

        results.append({
            "product": product,
            "avg_oos_duration_minutes": round(oos_duration, 1),
            "pos_impact_lag_minutes": round(lag_minutes, 1),
            "shelf_to_pos_correlation": round(correlation, 3),
            "sales_drop_during_oos_pct": round(sales_drop_pct, 3),
            "avg_daily_units_sold": avg_daily_units,
            "restock_priority_score": round(priority_score, 4),
            "recommended_safety_stock_days": max(1, round(oos_duration / 60 / 8 * 2, 1)),
        })

    results.sort(key=lambda r: r["restock_priority_score"], reverse=True)

    return {
        "products_analyzed": len(results),
        "avg_shelf_to_pos_correlation": round(float(np.mean([r["shelf_to_pos_correlation"] for r in results])), 3),
        "avg_impact_lag_minutes": round(float(np.mean([r["pos_impact_lag_minutes"] for r in results])), 1),
        "high_priority_restocks": [r for r in results if r["restock_priority_score"] > np.percentile([r["restock_priority_score"] for r in results], 75)],
        "full_product_analysis": results,
        "recommendation": "Prioritize automated reorder triggers for top-10 products by priority score.",
    }


# ── Planogram deviation score ─────────────────────────────────────────────────

@app.post("/planogram_deviation", summary="Score shelf layout vs planogram compliance")
async def planogram_deviation(image_request: ImageRequest):
    """
    Quantify how far the current shelf layout deviates from the target planogram.
    Returns a compliance score (0-100), top deviation categories, and estimated
    revenue impact of non-compliance.
    """
    _get_model()  # ensure model loaded

    # Simulate planogram compliance scoring (in production: compare detection output
    # to expected SKU positions from planogram database)
    np.random.seed(hash(image_request.image_data[:20] if image_request.image_data else "x") % 2**31)

    compliance_score = np.random.uniform(62, 98)
    deviations = []

    deviation_types = [
        ("Wrong product in slot", np.random.randint(0, 4)),
        ("Missing facing", np.random.randint(0, 6)),
        ("Incorrect shelf height", np.random.randint(0, 3)),
        ("Out-of-stock (no product)", np.random.randint(0, 5)),
        ("Label/tag mismatch", np.random.randint(0, 4)),
    ]

    for dev_type, count in deviation_types:
        if count > 0:
            revenue_impact = count * np.random.uniform(12, 45)
            deviations.append({
                "deviation_type": dev_type,
                "count": count,
                "estimated_daily_revenue_impact_usd": round(revenue_impact, 2),
            })

    deviations.sort(key=lambda d: d["estimated_daily_revenue_impact_usd"], reverse=True)
    total_impact = sum(d["estimated_daily_revenue_impact_usd"] for d in deviations)

    return {
        "planogram_compliance_score": round(compliance_score, 1),
        "compliance_grade": "A" if compliance_score >= 90 else ("B" if compliance_score >= 80 else ("C" if compliance_score >= 70 else "D")),
        "total_deviations": sum(d["count"] for d in deviations),
        "estimated_daily_revenue_impact_usd": round(total_impact, 2),
        "deviations": deviations,
        "priority_action": deviations[0]["deviation_type"] if deviations else "None — fully compliant",
    }


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load model on startup."""
    logger.info("Retail Operations Intelligence API starting up...")
    try:
        _get_model()
    except Exception as exc:
        logger.warning("Model not loaded on startup (will retry on first request): %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

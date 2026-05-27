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

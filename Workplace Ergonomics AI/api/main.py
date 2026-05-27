"""
Workplace Ergonomics & Injury Prevention Platform
FastAPI Backend - REST API for pose analysis, REBA/RULA scoring, shift management
"""

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Workplace Ergonomics & Injury Prevention API",
    description="Real-time REBA/RULA ergonomic risk scoring from pose estimation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pose_extractor: Optional[Any] = None
_alert_manager: Optional[Any] = None
_worker_sessions: Dict[str, List[Dict[str, Any]]] = {}  # worker_id → list of observations
_zone_risk_map: Dict[str, List[float]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ImageAnalysisResponse(BaseModel):
    keypoints: Dict[str, Any]
    reba_score: int
    reba_breakdown: Dict[str, Any]
    rula_score: int
    rula_breakdown: Dict[str, Any]
    risk_level: str
    alert_triggered: Optional[Dict[str, Any]]
    annotated_image_b64: str
    processing_time_ms: float


class SessionData(BaseModel):
    worker_id: str
    reba_scores: List[int]
    timestamps: List[float]
    zone: str = "unknown"


class ShiftReportResponse(BaseModel):
    shift_date: str
    worker_summaries: Dict[str, Any]
    overall_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_pose_extractor() -> Any:
    global _pose_extractor
    if _pose_extractor is None:
        try:
            from pose_extractor import PoseExtractor  # type: ignore
            _pose_extractor = PoseExtractor(model_type="mediapipe", confidence=0.5)
            logger.info("PoseExtractor initialized.")
        except Exception as exc:
            logger.error("PoseExtractor init failed: %s", exc)
    return _pose_extractor


def _get_alert_manager() -> Any:
    global _alert_manager
    if _alert_manager is None:
        try:
            from alert_system import AlertManager  # type: ignore
            _alert_manager = AlertManager()
        except Exception as exc:
            logger.error("AlertManager init failed: %s", exc)
    return _alert_manager


def _decode_image(file_bytes: bytes) -> np.ndarray:
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
        return np.array(pil)[:, :, ::-1]


def _image_to_b64(image: np.ndarray) -> str:
    try:
        import cv2  # type: ignore
        _, buf = cv2.imencode(".jpg", image)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return ""


def _run_full_analysis(
    image: np.ndarray,
    worker_id: str = "unknown",
    zone: str = "unknown",
) -> Dict[str, Any]:
    """Run complete ergonomics analysis pipeline on a single image."""
    start = time.perf_counter()

    pose_ext = _get_pose_extractor()
    alert_mgr = _get_alert_manager()

    # Extract keypoints
    keypoints: Dict[str, Any] = {}
    if pose_ext is not None:
        try:
            keypoints = pose_ext.extract_keypoints(image)
        except Exception as exc:
            logger.warning("Keypoint extraction failed: %s", exc)

    # Compute REBA & RULA scores
    reba_result: Dict[str, Any] = {}
    rula_result: Dict[str, Any] = {}

    try:
        from reba_rula import compute_reba_score, compute_rula_score  # type: ignore
        reba_result = compute_reba_score(keypoints)
        rula_result = compute_rula_score(keypoints)
    except Exception as exc:
        logger.warning("REBA/RULA scoring failed: %s", exc)
        reba_result = {"reba_score": 1, "risk_level": "negligible", "action_required": "N/A", "body_part_scores": {}}
        rula_result = {"rula_score": 1, "risk_level": "acceptable", "recommended_action": "N/A", "body_part_scores": {}}

    reba_score = reba_result.get("reba_score", 1)
    rula_score = rula_result.get("rula_score", 1)
    risk_level = reba_result.get("risk_level", "negligible")

    # Check alerts
    alert_triggered: Optional[Dict[str, Any]] = None
    if alert_mgr is not None:
        try:
            alert_triggered = alert_mgr.check_alert(
                worker_id=worker_id,
                reba_score=reba_score,
                timestamp=time.time(),
                zone=zone,
            )
        except Exception as exc:
            logger.warning("Alert check failed: %s", exc)

    # Annotate image
    annotated = image.copy()
    try:
        from visualization import draw_skeleton, draw_reba_overlay  # type: ignore
        annotated = draw_skeleton(annotated, keypoints, reba_score)
        annotated = draw_reba_overlay(annotated, reba_result)
    except Exception as exc:
        logger.debug("Visualization failed: %s", exc)

    annotated_b64 = _image_to_b64(annotated)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Store session data
    if worker_id not in _worker_sessions:
        _worker_sessions[worker_id] = []
    _worker_sessions[worker_id].append({
        "reba_score": reba_score,
        "rula_score": rula_score,
        "timestamp": time.time(),
        "zone": zone,
        "risk_level": risk_level,
    })

    # Update zone risk map
    if zone != "unknown":
        _zone_risk_map.setdefault(zone, []).append(float(reba_score))
        if len(_zone_risk_map[zone]) > 1000:
            _zone_risk_map[zone] = _zone_risk_map[zone][-500:]

    return {
        "keypoints": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in keypoints.items()},
        "reba_score": reba_score,
        "reba_breakdown": reba_result,
        "rula_score": rula_score,
        "rula_breakdown": rula_result,
        "risk_level": risk_level,
        "alert_triggered": alert_triggered,
        "annotated_image_b64": annotated_b64,
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
async def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "Workplace Ergonomics & Injury Prevention API",
        "version": "1.0.0",
    }


@app.post("/analyze_image", response_model=ImageAnalysisResponse,
          summary="Analyze a single image for ergonomic risk")
async def analyze_image(
    file: UploadFile = File(...),
    worker_id: str = "worker_001",
    zone: str = "unknown",
) -> ImageAnalysisResponse:
    """
    Upload a worker image to receive REBA/RULA scores, keypoints, and risk assessment.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_bytes = await file.read()
    try:
        image = _decode_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    result = _run_full_analysis(image, worker_id=worker_id, zone=zone)
    return ImageAnalysisResponse(**result)


@app.post("/analyze_video_frame", summary="Analyze a live video stream frame")
async def analyze_video_frame(
    file: UploadFile = File(...),
    worker_id: str = "worker_001",
    zone: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze a single frame from a live video stream.
    Returns REBA/RULA + worker_id tracking context.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_bytes = await file.read()
    try:
        image = _decode_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    result = _run_full_analysis(image, worker_id=worker_id, zone=zone)
    result["worker_id"] = worker_id
    result["frame_timestamp"] = time.time()
    return result


@app.post("/submit_session", summary="Submit a full session for shift report")
async def submit_session(session: SessionData) -> Dict[str, Any]:
    """
    Submit a complete session of REBA scores for a worker.
    Returns a full shift summary with cumulative exposure and recommendations.
    """
    try:
        from risk_analytics import (  # type: ignore
            compute_cumulative_exposure,
            compute_posture_fatigue,
            identify_high_risk_moments,
            compute_shift_summary,
            generate_ergonomics_report,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics module error: {exc}")

    reba_scores = session.reba_scores
    timestamps = session.timestamps

    if len(reba_scores) != len(timestamps):
        raise HTTPException(status_code=422, detail="reba_scores and timestamps must be same length.")

    exposure = compute_cumulative_exposure(reba_scores, timestamps)
    fatigue = compute_posture_fatigue(reba_scores, timestamps)
    high_risk = identify_high_risk_moments(reba_scores, timestamps, threshold=7)

    session_data = {
        "worker_id": session.worker_id,
        "reba_scores": reba_scores,
        "cumulative_exposure": exposure,
        "fatigue_info": fatigue,
        "high_risk_moments": high_risk,
        "avg_reba": float(sum(reba_scores) / max(len(reba_scores), 1)),
        "peak_reba": max(reba_scores) if reba_scores else 0,
    }

    shift_summary = compute_shift_summary([session_data])
    report = generate_ergonomics_report(session.worker_id, shift_summary, high_risk)

    return {
        "worker_id": session.worker_id,
        "shift_summary": shift_summary,
        "report": report,
        "cumulative_exposure": exposure,
        "fatigue_info": fatigue,
        "high_risk_moments": high_risk[:20],
    }


@app.get("/worker_summary/{worker_id}", summary="Cumulative exposure and risk trend for a worker")
async def worker_summary(worker_id: str) -> Dict[str, Any]:
    """Returns cumulative REBA scores, risk level distribution, and trend for a worker."""
    sessions = _worker_sessions.get(worker_id, [])
    if not sessions:
        return {"worker_id": worker_id, "message": "No data yet for this worker."}

    reba_scores = [s["reba_score"] for s in sessions]
    import numpy as np
    avg_reba = float(np.mean(reba_scores))
    peak_reba = max(reba_scores)

    from reba_rula import get_reba_risk_level  # type: ignore
    risk_counts: Dict[str, int] = {}
    for score in reba_scores:
        level = get_reba_risk_level(score)
        risk_counts[level] = risk_counts.get(level, 0) + 1

    return {
        "worker_id": worker_id,
        "n_observations": len(sessions),
        "avg_reba": round(avg_reba, 2),
        "peak_reba": peak_reba,
        "risk_distribution": risk_counts,
        "recent_scores": reba_scores[-20:],
        "zones_active": list({s["zone"] for s in sessions}),
    }


@app.get("/active_alerts", summary="Current active ergonomic alerts")
async def active_alerts() -> Dict[str, Any]:
    """Returns all currently active ergonomic alerts."""
    alert_mgr = _get_alert_manager()
    if alert_mgr is None:
        return {"alerts": [], "total": 0}

    alerts = alert_mgr.get_active_alerts()
    return {
        "alerts": alerts,
        "total": len(alerts),
        "critical": sum(1 for a in alerts if a["severity"] == "critical"),
        "high": sum(1 for a in alerts if a["severity"] == "high"),
        "medium": sum(1 for a in alerts if a["severity"] == "medium"),
    }


@app.get("/zone_risk_map", summary="Risk scores aggregated by physical zone")
async def zone_risk_map() -> Dict[str, Any]:
    """Returns average REBA scores and risk levels by physical zone."""
    import numpy as np
    from reba_rula import get_reba_risk_level  # type: ignore

    zone_summary: Dict[str, Any] = {}
    for zone, scores in _zone_risk_map.items():
        if not scores:
            continue
        avg = float(np.mean(scores))
        zone_summary[zone] = {
            "avg_reba": round(avg, 2),
            "peak_reba": max(scores),
            "risk_level": get_reba_risk_level(int(round(avg))),
            "n_observations": len(scores),
        }

    return {"zones": zone_summary, "total_zones": len(zone_summary)}


@app.get("/shift_report/{shift_date}", summary="Full shift ergonomics report")
async def shift_report(shift_date: str) -> Dict[str, Any]:
    """Returns the full ergonomics report for all workers for a given shift date."""
    try:
        from risk_analytics import compute_shift_summary, generate_ergonomics_report  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics error: {exc}")

    import numpy as np
    all_summaries: Dict[str, Any] = {}
    all_reba: List[float] = []

    for worker_id, sessions in _worker_sessions.items():
        reba_scores = [s["reba_score"] for s in sessions]
        all_reba.extend(reba_scores)
        session_like = [{
            "reba_scores": reba_scores,
            "cumulative_exposure": {},
            "avg_reba": float(np.mean(reba_scores)) if reba_scores else 0.0,
            "peak_reba": max(reba_scores) if reba_scores else 0,
            "high_risk_moments": [s for s in sessions if s["reba_score"] > 7],
        }]
        all_summaries[worker_id] = compute_shift_summary(session_like)

    overall_stats = {
        "total_workers": len(_worker_sessions),
        "avg_reba_fleet": round(float(np.mean(all_reba)), 2) if all_reba else 0.0,
        "peak_reba_fleet": max(all_reba) if all_reba else 0,
        "shift_date": shift_date,
    }

    return {
        "shift_date": shift_date,
        "worker_summaries": all_summaries,
        "overall_stats": overall_stats,
    }


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Workplace Ergonomics API starting up...")
    try:
        _get_pose_extractor()
        _get_alert_manager()
    except Exception as exc:
        logger.warning("Startup init failed (will retry on first request): %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)

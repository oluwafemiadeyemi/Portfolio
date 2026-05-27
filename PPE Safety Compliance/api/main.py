"""
FastAPI service for PPE Compliance Monitoring System.
Endpoints: analyze_image, analyze_video_frame, site_compliance,
           active_violations, zone_compliance, shift_report,
           configure_site, osha_report
"""

import base64
import io
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe_detector import PPEDetector
from src.compliance_engine import (
    PPEComplianceChecker,
    compute_site_compliance_rate,
    compute_compliance_by_zone,
    compute_compliance_trend,
    identify_repeat_offenders,
    define_zones,
    monitor_restricted_areas,
)
from src.alert_system import AlertManager, generate_osha_incident_report
from src.visualization import draw_ppe_detections, draw_zone_overlay

app = FastAPI(
    title="PPE Compliance Monitoring API",
    description="OSHA-grade AI-powered PPE detection and compliance monitoring.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
_detector = PPEDetector(conf_threshold=0.4, iou_threshold=0.5, device="cpu")
_checker = PPEComplianceChecker()
_alert_manager = AlertManager()
_violation_log: list[dict[str, Any]] = []
_site_zones: dict[str, dict] = {
    "construction": {
        "type": "rectangle",
        "coords": [0, 0, 640, 480],
        "ppe_required": ["Hardhat", "Safety_Vest"],
        "restricted": False,
    }
}
_site_info: dict[str, str] = {
    "site_name": "Default Facility",
    "address": "N/A",
    "establishment": "N/A",
    "naics_code": "238990",
}


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class SiteConfig(BaseModel):
    zones: dict[str, dict[str, Any]] = Field(default={})
    site_name: str = Field(default="Facility")
    address: str = Field(default="N/A")
    default_zone: str = Field(default="construction")


class VideoFrameRequest(BaseModel):
    frame_b64: str = Field(..., description="Base64-encoded JPEG frame")
    zone: str = Field(default="construction")
    frame_idx: int = Field(default=0)


# ─── Helper Functions ─────────────────────────────────────────────────────────
def _decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image data")
    return image


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode numpy image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _process_frame(
    image: np.ndarray,
    zone: str = "construction",
    frame_idx: int = 0,
    save_violation: bool = True,
) -> dict[str, Any]:
    """Core frame processing: detect → check compliance → create alerts."""
    t0 = time.perf_counter()

    detections = _detector.detect(image)
    compliance = _checker.check_worker_compliance(detections, zone=zone)

    # Generate violation records for non-compliant workers
    alerts_triggered: list[dict] = []
    if save_violation:
        for worker in compliance.get("workers_detail", []):
            if not worker.get("is_compliant", True) and worker.get("missing_ppe"):
                violation = _checker.generate_violation_record(
                    worker_id=worker.get("worker_id", "unknown"),
                    missing_ppe=worker.get("missing_ppe", []),
                    zone=zone,
                    timestamp=datetime.now(),
                )
                _violation_log.append(violation)
                alert = _alert_manager.create_alert(violation)
                _alert_manager.send_alert(alert)
                alerts_triggered.append({
                    "alert_id": alert["id"],
                    "severity": alert["severity"],
                    "worker_id": alert["worker_id"],
                })

    # Draw annotations
    annotated = draw_ppe_detections(image, detections, compliance)
    annotated_b64 = _encode_image_b64(annotated)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "frame_idx": frame_idx,
        "workers_detected": compliance.get("workers_detected", 0),
        "compliance_rate": compliance.get("compliance_score", 100.0),
        "compliance_status": compliance.get("compliance_status", "UNKNOWN"),
        "violations": [
            {
                "worker_id": w.get("worker_id"),
                "missing_ppe": w.get("missing_ppe", []),
                "is_compliant": w.get("is_compliant", True),
            }
            for w in compliance.get("workers_detail", [])
            if not w.get("is_compliant", True)
        ],
        "all_detections": [
            d.to_dict() if hasattr(d, "to_dict") else d for d in detections
        ],
        "alerts_triggered": alerts_triggered,
        "annotated_image_b64": annotated_b64,
        "processing_time_ms": elapsed_ms,
        "zone": zone,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/analyze_image")
async def analyze_image(
    file: UploadFile = File(...),
    zone: str = "construction",
) -> dict[str, Any]:
    """
    Upload an image file and receive PPE compliance analysis.
    Returns: workers_detected, compliance_rate, violations, alerts, annotated_image_b64.
    """
    try:
        image_bytes = await file.read()
        image = _decode_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    result = _process_frame(image, zone=zone, save_violation=True)
    return result


@app.post("/analyze_video_frame")
async def analyze_video_frame(request: VideoFrameRequest) -> dict[str, Any]:
    """
    Analyze a single video frame sent as base64-encoded JPEG.
    Suitable for real-time streaming analysis.
    """
    try:
        image_bytes = base64.b64decode(request.frame_b64)
        image = _decode_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {exc}")

    result = _process_frame(
        image, zone=request.zone,
        frame_idx=request.frame_idx,
        save_violation=(request.frame_idx % 5 == 0),  # Save every 5th frame to avoid spam
    )
    return result


@app.get("/site_compliance")
async def site_compliance() -> dict[str, Any]:
    """Return overall site compliance rate for the current shift (last 8 hours)."""
    compliance_rate = compute_site_compliance_rate(_violation_log, time_window_hours=8)
    alert_summary = _alert_manager.get_shift_alert_summary()

    return {
        "site_compliance_rate_pct": compliance_rate,
        "shift_summary": alert_summary,
        "total_violations_today": len(_violation_log),
        "critical_open_alerts": alert_summary.get("critical_open", 0),
        "status": "SAFE" if compliance_rate >= 95 else "CAUTION" if compliance_rate >= 80 else "UNSAFE",
    }


@app.get("/active_violations")
async def active_violations(
    severity: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Return current open violations, optionally filtered by severity."""
    alerts = _alert_manager.get_active_alerts(severity_filter=severity)[:limit]
    return {
        "active_violations": alerts,
        "total_open": len(_alert_manager.get_active_alerts()),
        "by_severity": {
            sev: len(_alert_manager.get_active_alerts(severity_filter=sev))
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
    }


@app.get("/zone_compliance")
async def zone_compliance() -> dict[str, Any]:
    """Return compliance rates broken down by zone."""
    zone_stats = compute_compliance_by_zone(_violation_log)
    restricted_violations = [v for v in _violation_log if v.get("severity") == "CRITICAL"]
    return {
        "zone_compliance": zone_stats,
        "total_zones_monitored": len(_site_zones),
        "zones_at_risk": [z for z, s in zone_stats.items() if s.get("compliance_rate", 100) < 90],
        "critical_violations": len(restricted_violations),
    }


@app.get("/shift_report")
async def shift_report() -> dict[str, Any]:
    """Return full shift compliance report."""
    compliance_by_zone = compute_compliance_by_zone(_violation_log)
    compliance_trend = compute_compliance_trend(_violation_log, period="hourly")
    repeat_offenders = identify_repeat_offenders(_violation_log)
    alert_summary = _alert_manager.get_shift_alert_summary()
    site_rate = compute_site_compliance_rate(_violation_log)

    # Violations by type
    violations_by_type: dict[str, int] = {}
    for v in _violation_log:
        for missing in v.get("missing_ppe", []):
            violations_by_type[missing] = violations_by_type.get(missing, 0) + 1

    severity_distribution: dict[str, int] = {}
    for v in _violation_log:
        sev = v.get("severity", "LOW")
        severity_distribution[sev] = severity_distribution.get(sev, 0) + 1

    return {
        "report_generated_at": datetime.now().isoformat(),
        "site_compliance_rate_pct": site_rate,
        "total_violations": len(_violation_log),
        "compliance_by_zone": compliance_by_zone,
        "compliance_trend_hourly": compliance_trend,
        "violations_by_type": violations_by_type,
        "severity_distribution": severity_distribution,
        "alert_summary": alert_summary,
        "repeat_offenders": repeat_offenders,
        "osha_status": "COMPLIANT" if site_rate >= 95 else "REQUIRES_ATTENTION",
    }


@app.post("/configure_site")
async def configure_site(config: SiteConfig) -> dict[str, Any]:
    """Configure site zones and PPE requirements."""
    global _site_zones, _site_info

    if config.zones:
        _site_zones = define_zones(config.zones)

    _site_info.update({
        "site_name": config.site_name,
        "address": config.address,
    })

    return {
        "status": "configured",
        "zones_configured": list(_site_zones.keys()),
        "site_name": config.site_name,
    }


@app.get("/osha_report/{report_date}")
async def osha_report(report_date: str) -> dict[str, Any]:
    """Generate OSHA 300 Log-compatible report for a given date."""
    day_violations = [
        v for v in _violation_log
        if v.get("date", "") == report_date
    ]

    report_text = generate_osha_incident_report(
        day_violations, _site_info, report_date=report_date
    )

    return {
        "report_date": report_date,
        "total_incidents": len(day_violations),
        "report_text": report_text,
        "site": _site_info.get("site_name"),
    }


@app.post("/resolve_alert/{alert_id}")
async def resolve_alert(
    alert_id: str,
    resolved_by: str = "supervisor",
) -> dict[str, Any]:
    """Resolve an open alert."""
    alert = _alert_manager.resolve_alert(alert_id, resolved_by=resolved_by)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"status": "resolved", "alert": alert}


@app.get("/model_info")
async def model_info() -> dict[str, Any]:
    """Return detector model metadata."""
    return _detector.get_model_info()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "backend": "ultralytics" if _detector.model else "fallback"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)

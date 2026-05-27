"""
PPE Compliance Engine — OSHA-Grade Workplace Safety Analysis.
Checks worker PPE compliance, assigns violations, and monitors zones.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PPEComplianceChecker:
    """
    Analyzes PPE detections to determine worker compliance status.

    Implements:
    - Per-worker compliance checking
    - Spatial PPE-to-worker assignment
    - Violation severity classification
    - Structured violation records
    """

    REQUIRED_PPE_BY_ZONE: dict[str, list[str]] = {
        "construction": ["Hardhat", "Safety_Vest"],
        "chemical": ["Hardhat", "Safety_Vest", "Safety_Glasses", "Gloves"],
        "office": [],
        "warehouse": ["Safety_Vest"],
        "electrical": ["Hardhat", "Safety_Gloves", "Safety_Glasses"],
        "welding": ["Hardhat", "Safety_Vest", "Safety_Glasses"],
        "general": ["Hardhat", "Safety_Vest"],
    }

    PPE_TO_MISSING_MAP: dict[str, str] = {
        "Hardhat": "No_Hardhat",
        "Safety_Vest": "No_Safety_Vest",
    }

    SEVERITY_WEIGHTS: dict[str, dict[str, int]] = {
        "construction": {"No_Hardhat": 3, "No_Safety_Vest": 2},
        "chemical": {"No_Hardhat": 3, "No_Safety_Vest": 2, "No_Safety_Glasses": 2, "Missing_Gloves": 1},
        "warehouse": {"No_Safety_Vest": 2},
        "office": {},
    }

    def __init__(self) -> None:
        self._zone_config: dict[str, dict] = {}

    def check_worker_compliance(
        self,
        detections: list[Any],  # list of Detection objects or dicts
        zone: str = "construction",
    ) -> dict[str, Any]:
        """
        Analyze a frame's detections to determine compliance.

        Args:
            detections: list of Detection objects with .class_name and .bbox
            zone: work zone type (determines required PPE)

        Returns:
            {
              workers_detected, ppe_detected, workers_detail,
              missing_ppe, compliance_status, compliance_score,
              violations_count, zone
            }
        """
        required_ppe = self.REQUIRED_PPE_BY_ZONE.get(zone, self.REQUIRED_PPE_BY_ZONE["general"])

        # Normalize detections to dicts
        det_list = [d.to_dict() if hasattr(d, "to_dict") else d for d in detections]

        persons = [d for d in det_list if d["class_name"] == "Person"]
        ppe_items = [d for d in det_list if d["class_name"] not in ("Person",)]
        missing_items = [d for d in det_list if d["class_name"].startswith("No_")]
        present_ppe = [d for d in det_list if d["class_name"] not in ("Person",)
                       and not d["class_name"].startswith("No_")]

        n_workers = len(persons)

        # Match PPE to workers
        workers_detail = self.match_ppe_to_workers(det_list)

        # Determine missing PPE per worker
        all_missing: list[str] = []
        for worker in workers_detail:
            worker_present = set(worker.get("ppe_present", []))
            worker_missing = [
                ppe for ppe in required_ppe if ppe not in worker_present
            ]
            # Also check explicit "No_X" detections near this worker
            for explicit_missing in worker.get("explicit_violations", []):
                if explicit_missing not in worker_missing:
                    worker_missing.append(explicit_missing)
            worker["missing_ppe"] = worker_missing
            worker["is_compliant"] = len(worker_missing) == 0
            all_missing.extend(worker_missing)

        n_violations = sum(1 for w in workers_detail if not w.get("is_compliant", True))
        compliance_score = (
            round((1.0 - n_violations / max(n_workers, 1)) * 100, 1)
            if n_workers > 0
            else 100.0
        )
        compliance_status = "COMPLIANT" if n_violations == 0 else "NON_COMPLIANT"

        return {
            "workers_detected": n_workers,
            "ppe_detected": [d["class_name"] for d in present_ppe],
            "missing_ppe_detected": [d["class_name"] for d in missing_items],
            "workers_detail": workers_detail,
            "missing_ppe": list(set(all_missing)),
            "compliance_status": compliance_status,
            "compliance_score": compliance_score,
            "violations_count": n_violations,
            "zone": zone,
            "required_ppe": required_ppe,
        }

    def match_ppe_to_workers(
        self,
        detections: list[dict[str, Any]],
        proximity_threshold: float = 200.0,
    ) -> list[dict[str, Any]]:
        """
        Spatially assign PPE items to nearest worker using center-point distance.
        Returns list of worker dicts with ppe_present and explicit_violations.
        """
        persons = [d for d in detections if d["class_name"] == "Person"]
        ppe_items = [d for d in detections if d["class_name"] not in ("Person",)]

        workers_detail: list[dict[str, Any]] = []

        for i, person in enumerate(persons):
            px, py = person["center_x"], person["center_y"]

            # Person bounding box dimensions for relative proximity
            bbox = person["bbox"]
            person_height = max(1, bbox[3] - bbox[1])

            ppe_present: list[str] = []
            explicit_violations: list[str] = []

            for ppe in ppe_items:
                ex, ey = ppe["center_x"], ppe["center_y"]
                dist = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
                # Scale threshold by person size
                threshold = proximity_threshold * (person_height / 200.0)

                if dist <= threshold:
                    if ppe["class_name"].startswith("No_"):
                        explicit_violations.append(ppe["class_name"])
                    else:
                        ppe_present.append(ppe["class_name"])

            workers_detail.append({
                "worker_id": f"worker_{i}",
                "bbox": person["bbox"],
                "center_x": px,
                "center_y": py,
                "confidence": person.get("confidence", 1.0),
                "ppe_present": ppe_present,
                "explicit_violations": explicit_violations,
            })

        return workers_detail

    def classify_violation_severity(
        self,
        missing_ppe_list: list[str],
        zone: str = "construction",
    ) -> str:
        """
        Classify violation severity: CRITICAL / HIGH / MEDIUM / LOW.

        Rules:
        - CRITICAL: No hardhat in high-risk zone, or multiple missing items in chemical zone
        - HIGH: No safety vest on construction/warehouse, or hardhat missing in general
        - MEDIUM: Missing glasses in chemical zone, or single item missing in moderate-risk zone
        - LOW: Minor missing items in low-risk zones
        """
        if not missing_ppe_list:
            return "NONE"

        missing_set = set(missing_ppe_list)

        # CRITICAL conditions
        if zone in ("construction", "electrical", "welding"):
            if "No_Hardhat" in missing_set or "Hardhat" in missing_set:
                return "CRITICAL"

        if zone == "chemical" and len(missing_ppe_list) >= 3:
            return "CRITICAL"

        if zone == "chemical":
            if "No_Hardhat" in missing_set or "Hardhat" in missing_set:
                return "CRITICAL"

        # HIGH conditions
        if zone in ("construction", "warehouse", "chemical"):
            if "No_Safety_Vest" in missing_set:
                return "HIGH"

        if len(missing_ppe_list) >= 2:
            return "HIGH"

        # MEDIUM conditions
        if zone == "chemical" and ("Safety_Glasses" in missing_set or "No_Safety_Glasses" in missing_set):
            return "MEDIUM"

        if missing_ppe_list:
            return "MEDIUM"

        return "LOW"

    def generate_violation_record(
        self,
        worker_id: str,
        missing_ppe: list[str],
        zone: str,
        timestamp: Optional[datetime] = None,
        frame_b64: Optional[str] = None,
        location: Optional[tuple[int, int]] = None,
    ) -> dict[str, Any]:
        """
        Generate a structured violation record compatible with OSHA logging.

        Returns dict with all violation metadata.
        """
        if timestamp is None:
            timestamp = datetime.now()

        severity = self.classify_violation_severity(missing_ppe, zone)

        record = {
            "violation_id": str(uuid.uuid4())[:8],
            "worker_id": worker_id,
            "zone": zone,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "shift": "Day" if 6 <= timestamp.hour < 18 else "Night",
            "missing_ppe": missing_ppe,
            "severity": severity,
            "status": "open",
            "osha_code": _get_osha_code(missing_ppe, zone),
            "description": _build_violation_description(worker_id, missing_ppe, zone, severity),
            "frame_image_b64": frame_b64,
            "worker_location": location,
        }

        logger.debug(f"Violation record: {record['violation_id']} — {severity} — {worker_id}")
        return record


def _get_osha_code(missing_ppe: list[str], zone: str) -> str:
    """Map missing PPE to approximate OSHA code."""
    code_map = {
        "No_Hardhat": "29 CFR 1926.100(a)",
        "Hardhat": "29 CFR 1926.100(a)",
        "No_Safety_Vest": "29 CFR 1926.201",
        "Safety_Vest": "29 CFR 1926.201",
        "No_Safety_Glasses": "29 CFR 1910.133(a)(1)",
        "Safety_Glasses": "29 CFR 1910.133(a)(1)",
        "Gloves": "29 CFR 1910.138(a)",
        "Missing_Gloves": "29 CFR 1910.138(a)",
    }
    codes = [code_map.get(item, "29 CFR 1910.132") for item in missing_ppe]
    return "; ".join(set(codes)) if codes else "29 CFR 1910.132"


def _build_violation_description(
    worker_id: str, missing_ppe: list[str], zone: str, severity: str
) -> str:
    """Build human-readable violation description."""
    ppe_str = ", ".join(missing_ppe) if missing_ppe else "unspecified PPE"
    return (
        f"Worker {worker_id} in {zone} zone detected without: {ppe_str}. "
        f"Severity: {severity}. "
        f"OSHA requires appropriate PPE for this work environment."
    )


# ─── Temporal & Site Analysis Functions ──────────────────────────────────────

def compute_site_compliance_rate(
    violation_log: list[dict[str, Any]],
    time_window_hours: float = 8.0,
) -> float:
    """
    Compute percentage of compliant worker-moments over a time window.
    Returns compliance rate (0-100).
    """
    if not violation_log:
        return 100.0

    now = datetime.now()
    window_start = now.replace(hour=max(0, now.hour - int(time_window_hours)))

    relevant = [
        v for v in violation_log
        if datetime.fromisoformat(v["timestamp"]) >= window_start
    ]

    if not relevant:
        return 100.0

    total_checks = len(relevant)
    violations = sum(1 for v in relevant if v.get("severity", "NONE") != "NONE")
    compliance_rate = round((1.0 - violations / max(total_checks, 1)) * 100, 1)
    return compliance_rate


def compute_compliance_by_zone(
    violation_log: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Compute compliance rate per zone from violation log.
    Returns dict: {zone: {compliance_rate, violations, total, severity_breakdown}}
    """
    zone_data: dict[str, dict] = {}

    for record in violation_log:
        zone = record.get("zone", "unknown")
        if zone not in zone_data:
            zone_data[zone] = {
                "total": 0,
                "violations": 0,
                "severity_counts": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0},
            }

        zone_data[zone]["total"] += 1
        sev = record.get("severity", "NONE")
        if sev != "NONE":
            zone_data[zone]["violations"] += 1
        zone_data[zone]["severity_counts"][sev] = zone_data[zone]["severity_counts"].get(sev, 0) + 1

    result: dict[str, dict[str, Any]] = {}
    for zone, data in zone_data.items():
        total = data["total"]
        violations = data["violations"]
        result[zone] = {
            "compliance_rate": round((1.0 - violations / max(total, 1)) * 100, 1),
            "violations": violations,
            "total_checks": total,
            "severity_breakdown": data["severity_counts"],
        }

    return result


def compute_compliance_trend(
    violation_log: list[dict[str, Any]],
    period: str = "daily",
) -> list[dict[str, Any]]:
    """
    Compute compliance rate trend over time.
    period: 'hourly', 'daily', 'weekly'
    Returns sorted list of {period_label, compliance_rate, violations, total}.
    """
    import pandas as pd

    if not violation_log:
        return []

    df = pd.DataFrame(violation_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    freq_map = {"hourly": "h", "daily": "D", "weekly": "W"}
    freq = freq_map.get(period, "D")

    df["is_violation"] = df["severity"].apply(lambda s: 1 if s not in ("NONE", None) else 0)
    df["period"] = df["timestamp"].dt.to_period(freq)

    grouped = df.groupby("period").agg(
        violations=("is_violation", "sum"),
        total=("is_violation", "count"),
    ).reset_index()

    result: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        compliance = round((1.0 - row["violations"] / max(row["total"], 1)) * 100, 1)
        result.append({
            "period_label": str(row["period"]),
            "compliance_rate": compliance,
            "violations": int(row["violations"]),
            "total": int(row["total"]),
        })

    return sorted(result, key=lambda x: x["period_label"])


def identify_repeat_offenders(
    violation_log: list[dict[str, Any]],
    min_violations: int = 3,
    shift_window_hours: float = 8.0,
) -> list[dict[str, Any]]:
    """
    Identify workers with >= min_violations in a shift window.
    Returns list of {worker_id, violation_count, severity_counts, ppe_types}.
    """
    from datetime import timedelta

    now = datetime.now()
    window_start = now - timedelta(hours=shift_window_hours)

    recent = [
        v for v in violation_log
        if datetime.fromisoformat(v["timestamp"]) >= window_start
        and v.get("worker_id", "unknown") != "unknown"
    ]

    worker_stats: dict[str, dict] = {}
    for v in recent:
        wid = v["worker_id"]
        if wid not in worker_stats:
            worker_stats[wid] = {
                "violation_count": 0,
                "severity_counts": {},
                "ppe_types": [],
            }
        worker_stats[wid]["violation_count"] += 1
        sev = v.get("severity", "LOW")
        worker_stats[wid]["severity_counts"][sev] = worker_stats[wid]["severity_counts"].get(sev, 0) + 1
        worker_stats[wid]["ppe_types"].extend(v.get("missing_ppe", []))

    offenders = [
        {
            "worker_id": wid,
            "violation_count": data["violation_count"],
            "severity_counts": data["severity_counts"],
            "ppe_types": list(set(data["ppe_types"])),
            "risk_level": "HIGH" if data["violation_count"] >= 5 else "MEDIUM",
        }
        for wid, data in worker_stats.items()
        if data["violation_count"] >= min_violations
    ]

    return sorted(offenders, key=lambda x: x["violation_count"], reverse=True)


# ─── Zone Monitoring ──────────────────────────────────────────────────────────

def define_zones(zone_config_dict: dict[str, dict]) -> dict[str, dict]:
    """
    Store and validate zone polygon/rectangle definitions.

    zone_config_dict format:
    {
      "zone_name": {
        "type": "rectangle",  # or "polygon"
        "coords": [x1, y1, x2, y2],  # for rectangle
        # or "polygon": [[x1,y1], [x2,y2], ...]
        "ppe_required": ["Hardhat", "Safety_Vest"],
        "restricted": False,
      }
    }
    """
    validated: dict[str, dict] = {}
    for zone_name, config in zone_config_dict.items():
        zone_type = config.get("type", "rectangle")
        coords = config.get("coords", [0, 0, 640, 480])
        ppe_req = config.get("ppe_required",
                             PPEComplianceChecker.REQUIRED_PPE_BY_ZONE.get(zone_name, []))
        restricted = config.get("restricted", False)

        validated[zone_name] = {
            "type": zone_type,
            "coords": coords,
            "ppe_required": ppe_req,
            "restricted": restricted,
        }
        logger.info(f"Zone defined: {zone_name} | PPE: {ppe_req} | Restricted: {restricted}")

    return validated


def check_zone_entry(
    worker_position: tuple[int, int],
    zones: dict[str, dict],
) -> Optional[str]:
    """
    Determine which zone a worker is currently in.
    Returns zone name or None if not in any defined zone.
    """
    wx, wy = worker_position

    for zone_name, config in zones.items():
        coords = config.get("coords", [])
        zone_type = config.get("type", "rectangle")

        if zone_type == "rectangle" and len(coords) == 4:
            x1, y1, x2, y2 = coords
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                return zone_name

        elif zone_type == "polygon" and len(coords) >= 3:
            # Point-in-polygon using ray casting
            if _point_in_polygon(wx, wy, coords):
                return zone_name

    return None


def _point_in_polygon(x: int, y: int, polygon: list[list[int]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside


def monitor_restricted_areas(
    detections: list[Any],
    restricted_zones: dict[str, dict],
) -> list[dict[str, Any]]:
    """
    Flag workers detected in restricted areas.
    Returns list of unauthorized zone entry events.
    """
    events: list[dict[str, Any]] = []

    det_list = [d.to_dict() if hasattr(d, "to_dict") else d for d in detections]
    persons = [d for d in det_list if d["class_name"] == "Person"]

    restricted = {name: cfg for name, cfg in restricted_zones.items() if cfg.get("restricted", False)}

    for person in persons:
        cx, cy = person["center_x"], person["center_y"]
        zone = check_zone_entry((cx, cy), restricted)
        if zone:
            events.append({
                "event_type": "UNAUTHORIZED_ZONE_ENTRY",
                "zone": zone,
                "worker_position": (cx, cy),
                "worker_bbox": person["bbox"],
                "timestamp": datetime.now().isoformat(),
                "severity": "CRITICAL",
            })
            logger.warning(f"Unauthorized zone entry in {zone} at ({cx},{cy})")

    return events

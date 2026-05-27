"""
Workplace Ergonomics & Injury Prevention Platform
Alert System Module - real-time ergonomic risk alerting
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Alert type constants
ALERT_IMMEDIATE_ACTION = "IMMEDIATE_ACTION"        # REBA ≥ 11
ALERT_HIGH_RISK_SUSTAINED = "HIGH_RISK_SUSTAINED"  # REBA ≥ 8 for >30 sec
ALERT_FATIGUE_DETECTED = "FATIGUE_DETECTED"        # Score deteriorating
ALERT_ZONE_RISK = "ZONE_RISK"                       # Repeated high scores in zone


class AlertManager:
    """
    Manages real-time ergonomic risk alerts for workers on the warehouse/manufacturing floor.

    Tracks current REBA scores per worker, detects threshold crossings,
    sustained high-risk exposures, and fatigue patterns.

    Attributes:
        thresholds: Dict of alert threshold values.
        _alerts: Dict of alert_id → alert dict (active alerts).
        _history: List of all alerts (including resolved).
        _worker_state: Per-worker tracking state.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> None:
        default_thresholds: Dict[str, Any] = {
            "reba_immediate": 11,        # Immediate action threshold
            "reba_sustained": 8,         # Sustained risk threshold
            "sustained_duration_sec": 30,  # Duration before sustained alert
            "fatigue_slope_threshold": 1.0,  # REBA score increase per unit time
            "zone_repeat_count": 3,      # Repeated high-risk events in one zone to alert
        }
        self.thresholds: Dict[str, Any] = {**default_thresholds, **(thresholds or {})}
        self._alerts: Dict[str, Dict[str, Any]] = {}  # active alerts
        self._history: List[Dict[str, Any]] = []
        self._worker_state: Dict[str, Dict[str, Any]] = {}

    def check_alert(
        self,
        worker_id: str,
        reba_score: int,
        timestamp: float,
        zone: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        """
        Checks a new REBA observation and triggers alerts if thresholds are crossed.

        Args:
            worker_id: Worker identifier.
            reba_score: Current REBA score (1-15).
            timestamp: Observation timestamp (seconds since epoch or shift start).
            zone: Physical zone/location of the worker.

        Returns:
            Alert dict if a new alert was triggered, None otherwise.
        """
        # Initialize worker state if first observation
        if worker_id not in self._worker_state:
            self._worker_state[worker_id] = {
                "score_history": [],
                "timestamp_history": [],
                "high_risk_start": None,
                "zone_counts": {},
                "last_alert_type": None,
            }

        state = self._worker_state[worker_id]
        state["score_history"].append(reba_score)
        state["timestamp_history"].append(timestamp)
        # Keep only recent history (last 10 minutes worth, max 600 obs)
        if len(state["score_history"]) > 600:
            state["score_history"] = state["score_history"][-300:]
            state["timestamp_history"] = state["timestamp_history"][-300:]

        alert: Optional[Dict[str, Any]] = None

        # 1. IMMEDIATE_ACTION: REBA ≥ 11
        if reba_score >= self.thresholds["reba_immediate"]:
            alert = self._create_alert(
                alert_type=ALERT_IMMEDIATE_ACTION,
                worker_id=worker_id,
                reba_score=reba_score,
                timestamp=timestamp,
                zone=zone,
                severity="critical",
                message=(
                    f"Worker {worker_id} has REBA score {reba_score} (very high risk). "
                    f"Immediate intervention required in zone {zone}."
                ),
            )

        # 2. HIGH_RISK_SUSTAINED: REBA ≥ 8 for > sustained_duration_sec
        elif reba_score >= self.thresholds["reba_sustained"]:
            if state["high_risk_start"] is None:
                state["high_risk_start"] = timestamp
            else:
                duration = timestamp - state["high_risk_start"]
                if duration >= self.thresholds["sustained_duration_sec"]:
                    # Avoid spamming: only alert once per sustained event
                    already_active = any(
                        a["type"] == ALERT_HIGH_RISK_SUSTAINED
                        and a["worker_id"] == worker_id
                        and a["status"] == "active"
                        for a in self._alerts.values()
                    )
                    if not already_active:
                        alert = self._create_alert(
                            alert_type=ALERT_HIGH_RISK_SUSTAINED,
                            worker_id=worker_id,
                            reba_score=reba_score,
                            timestamp=timestamp,
                            zone=zone,
                            severity="high",
                            message=(
                                f"Worker {worker_id} has maintained REBA ≥ {self.thresholds['reba_sustained']} "
                                f"for {duration:.0f}s in zone {zone}. Ergonomic intervention needed."
                            ),
                            metadata={"duration_sec": round(duration, 1)},
                        )
        else:
            # Reset high-risk timer when score drops below threshold
            state["high_risk_start"] = None

        # 3. ZONE_RISK: repeated high scores in same zone
        if zone != "unknown" and reba_score >= 7:
            state["zone_counts"][zone] = state["zone_counts"].get(zone, 0) + 1
            if state["zone_counts"][zone] >= self.thresholds["zone_repeat_count"]:
                already_active = any(
                    a["type"] == ALERT_ZONE_RISK
                    and a["worker_id"] == worker_id
                    and a.get("zone") == zone
                    and a["status"] == "active"
                    for a in self._alerts.values()
                )
                if not already_active:
                    alert = self._create_alert(
                        alert_type=ALERT_ZONE_RISK,
                        worker_id=worker_id,
                        reba_score=reba_score,
                        timestamp=timestamp,
                        zone=zone,
                        severity="medium",
                        message=(
                            f"Zone {zone} shows repeated high ergonomic risk for worker {worker_id}. "
                            f"Workstation redesign recommended."
                        ),
                        metadata={"zone_high_risk_count": state["zone_counts"][zone]},
                    )

        # 4. FATIGUE_DETECTED: upward trend in scores
        if len(state["score_history"]) >= 10:
            recent_scores = state["score_history"][-10:]
            try:
                import numpy as np
                times = list(range(len(recent_scores)))
                coeffs = np.polyfit(times, recent_scores, 1)
                slope = float(coeffs[0])
                if slope >= self.thresholds["fatigue_slope_threshold"]:
                    already_active = any(
                        a["type"] == ALERT_FATIGUE_DETECTED
                        and a["worker_id"] == worker_id
                        and a["status"] == "active"
                        for a in self._alerts.values()
                    )
                    if not already_active:
                        alert = self._create_alert(
                            alert_type=ALERT_FATIGUE_DETECTED,
                            worker_id=worker_id,
                            reba_score=reba_score,
                            timestamp=timestamp,
                            zone=zone,
                            severity="medium",
                            message=(
                                f"Worker {worker_id} shows fatigue pattern: REBA score increasing "
                                f"at {slope:.2f} points per observation. Consider a rest break."
                            ),
                            metadata={"trend_slope": round(slope, 3)},
                        )
            except ImportError:
                pass

        if alert:
            self._alerts[alert["alert_id"]] = alert
            self._history.append(alert.copy())
            logger.info("Alert triggered: %s for worker %s (REBA=%d)",
                        alert["type"], worker_id, reba_score)

        return alert

    def _create_alert(
        self,
        alert_type: str,
        worker_id: str,
        reba_score: int,
        timestamp: float,
        zone: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates an alert dict with a unique ID."""
        return {
            "alert_id": str(uuid.uuid4()),
            "type": alert_type,
            "worker_id": worker_id,
            "reba_score": reba_score,
            "timestamp": timestamp,
            "zone": zone,
            "severity": severity,
            "message": message,
            "status": "active",
            "created_at": time.time(),
            "resolved_at": None,
            "metadata": metadata or {},
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Returns all currently active (unresolved) alerts.

        Returns:
            List of active alert dicts, sorted by severity (critical first).
        """
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        active = [a for a in self._alerts.values() if a["status"] == "active"]
        return sorted(active, key=lambda a: severity_order.get(a["severity"], 9))

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Marks an alert as resolved.

        Args:
            alert_id: UUID of the alert to resolve.

        Returns:
            True if resolved successfully, False if not found.
        """
        if alert_id in self._alerts:
            self._alerts[alert_id]["status"] = "resolved"
            self._alerts[alert_id]["resolved_at"] = time.time()
            logger.info("Alert %s resolved.", alert_id)
            return True
        logger.warning("Alert %s not found.", alert_id)
        return False

    def get_alert_history(self, hours: float = 8.0) -> List[Dict[str, Any]]:
        """
        Returns alert log for the past N hours.

        Args:
            hours: Time window in hours.

        Returns:
            List of alert dicts (all statuses) within the time window.
        """
        cutoff = time.time() - hours * 3600
        return [a for a in self._history if a.get("created_at", 0) >= cutoff]

    def get_worker_alert_summary(self, worker_id: str) -> Dict[str, Any]:
        """
        Returns a summary of alerts for a specific worker.

        Args:
            worker_id: Worker identifier.

        Returns:
            Dict with total_alerts, critical, high, medium, low counts, and recent_alerts list.
        """
        worker_alerts = [a for a in self._history if a["worker_id"] == worker_id]
        counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for a in worker_alerts:
            sev = a.get("severity", "low")
            counts[sev] = counts.get(sev, 0) + 1

        return {
            "worker_id": worker_id,
            "total_alerts": len(worker_alerts),
            "by_severity": counts,
            "recent_alerts": sorted(worker_alerts, key=lambda x: x.get("created_at", 0), reverse=True)[:10],
        }

    def clear_worker_state(self, worker_id: str) -> None:
        """Clears per-worker tracking state (end of shift)."""
        if worker_id in self._worker_state:
            del self._worker_state[worker_id]
            logger.info("Cleared state for worker %s.", worker_id)

    def get_all_worker_ids(self) -> List[str]:
        """Returns list of all worker IDs currently being tracked."""
        return list(self._worker_state.keys())

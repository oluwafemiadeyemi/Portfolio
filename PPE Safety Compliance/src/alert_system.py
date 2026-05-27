"""
Alert Management System for PPE Compliance Monitoring.
Creates, routes, escalates, and resolves safety alerts.
Generates OSHA 300 Log compatible incident reports.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages PPE compliance alerts: creation, routing, resolution, and reporting.
    In production: integrates with notification services (Twilio, PagerDuty, Slack).
    """

    SEVERITY_ESCALATION_MINUTES: dict[str, int] = {
        "CRITICAL": 5,
        "HIGH": 15,
        "MEDIUM": 60,
        "LOW": 480,  # 8 hours
    }

    def __init__(self) -> None:
        self._alerts: dict[str, dict[str, Any]] = {}
        self._alert_log: list[dict[str, Any]] = []

    def create_alert(
        self,
        violation_record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create a new alert from a violation record.

        Returns alert dict with: id, timestamp, worker_id, zone,
        violation_type, severity, status, escalation_due_at.
        """
        severity = violation_record.get("severity", "LOW")
        alert_id = f"ALT-{str(uuid.uuid4())[:8].upper()}"
        now = datetime.now()

        escalation_minutes = self.SEVERITY_ESCALATION_MINUTES.get(severity, 60)
        escalation_due = now + timedelta(minutes=escalation_minutes)

        alert: dict[str, Any] = {
            "id": alert_id,
            "timestamp": now.isoformat(),
            "worker_id": violation_record.get("worker_id", "unknown"),
            "zone": violation_record.get("zone", "unknown"),
            "violation_type": "PPE_NON_COMPLIANCE",
            "missing_ppe": violation_record.get("missing_ppe", []),
            "severity": severity,
            "status": "open",
            "osha_code": violation_record.get("osha_code", "29 CFR 1910.132"),
            "description": violation_record.get("description", "PPE non-compliance detected"),
            "escalation_due_at": escalation_due.isoformat(),
            "escalated": False,
            "resolved_at": None,
            "resolved_by": None,
            "frame_image_b64": violation_record.get("frame_image_b64"),
        }

        self._alerts[alert_id] = alert
        self._alert_log.append(alert.copy())

        logger.info(
            f"Alert created: {alert_id} | {severity} | "
            f"Worker: {alert['worker_id']} | Zone: {alert['zone']}"
        )
        return alert

    def send_alert(self, alert: dict[str, Any]) -> None:
        """
        Dispatch alert notification.
        In production: integrates with Twilio (SMS), PagerDuty, Slack, email.
        Currently: logs to console with structured format.
        """
        severity = alert.get("severity", "UNKNOWN")
        worker_id = alert.get("worker_id", "unknown")
        zone = alert.get("zone", "unknown")
        missing = ", ".join(alert.get("missing_ppe", []))

        border = "=" * 60
        if severity == "CRITICAL":
            print(f"\n{'!'*60}")
            print(f"!! CRITICAL SAFETY ALERT — IMMEDIATE ACTION REQUIRED !!")
        elif severity == "HIGH":
            print(f"\n{border}")
            print(f"HIGH SEVERITY SAFETY ALERT")
        else:
            print(f"\n{'-' * 40}")
            print(f"Safety Alert [{severity}]")

        print(f"  Alert ID  : {alert.get('id', 'N/A')}")
        print(f"  Time      : {alert.get('timestamp', 'N/A')}")
        print(f"  Worker    : {worker_id}")
        print(f"  Zone      : {zone}")
        print(f"  Missing   : {missing or 'N/A'}")
        print(f"  OSHA Code : {alert.get('osha_code', 'N/A')}")
        print(f"  Escalate  : {alert.get('escalation_due_at', 'N/A')}")
        print(f"  Status    : {alert.get('status', 'open')}")

        if severity == "CRITICAL":
            print("!! Evacuate area / stop work until PPE is on !!")
            print(f"{'!'*60}\n")
        else:
            print(f"{border}\n")

        logger.warning(f"ALERT SENT [{severity}]: {alert.get('id')} — Worker {worker_id} in {zone}")

    def get_active_alerts(
        self,
        severity_filter: Optional[str | list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Return all open (unresolved) alerts, optionally filtered by severity.
        """
        active = [a for a in self._alerts.values() if a.get("status") == "open"]

        if severity_filter:
            if isinstance(severity_filter, str):
                severity_filter = [severity_filter]
            active = [a for a in active if a.get("severity") in severity_filter]

        return sorted(active, key=lambda x: (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(x.get("severity", "LOW"), 4),
            x.get("timestamp", ""),
        ))

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str = "system",
        resolution_note: str = "",
    ) -> Optional[dict[str, Any]]:
        """
        Mark an alert as resolved.
        Returns updated alert or None if not found.
        """
        if alert_id not in self._alerts:
            logger.warning(f"Alert not found: {alert_id}")
            return None

        alert = self._alerts[alert_id]
        alert["status"] = "resolved"
        alert["resolved_at"] = datetime.now().isoformat()
        alert["resolved_by"] = resolved_by
        alert["resolution_note"] = resolution_note

        # Update log entry
        for log_entry in self._alert_log:
            if log_entry.get("id") == alert_id:
                log_entry.update(alert)
                break

        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return alert

    def check_and_escalate(self) -> list[dict[str, Any]]:
        """
        Check for alerts past their escalation deadline and escalate them.
        Returns list of newly escalated alerts.
        """
        now = datetime.now()
        escalated: list[dict[str, Any]] = []

        for alert in self._alerts.values():
            if alert.get("status") != "open" or alert.get("escalated", False):
                continue
            due_str = alert.get("escalation_due_at")
            if due_str:
                due = datetime.fromisoformat(due_str)
                if now > due:
                    alert["escalated"] = True
                    alert["severity"] = {
                        "LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "CRITICAL",
                    }.get(alert["severity"], alert["severity"])
                    escalated.append(alert)
                    self.send_alert(alert)
                    logger.warning(f"Alert escalated: {alert['id']}")

        return escalated

    def get_shift_alert_summary(self) -> dict[str, Any]:
        """
        Summarize alerts for the current shift (last 8 hours).
        Returns: total, by_severity, resolution_rate, avg_resolution_time_minutes.
        """
        now = datetime.now()
        shift_start = now - timedelta(hours=8)

        shift_alerts = [
            a for a in self._alert_log
            if datetime.fromisoformat(a["timestamp"]) >= shift_start
        ]

        total = len(shift_alerts)
        by_severity: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        resolved_count = 0
        resolution_times: list[float] = []

        for alert in shift_alerts:
            sev = alert.get("severity", "LOW")
            by_severity[sev] = by_severity.get(sev, 0) + 1

            if alert.get("status") == "resolved" and alert.get("resolved_at"):
                resolved_count += 1
                created = datetime.fromisoformat(alert["timestamp"])
                resolved = datetime.fromisoformat(alert["resolved_at"])
                resolution_times.append((resolved - created).total_seconds() / 60.0)

        resolution_rate = round(resolved_count / max(total, 1) * 100, 1)
        avg_resolution_min = round(
            sum(resolution_times) / len(resolution_times), 1
        ) if resolution_times else 0.0

        return {
            "shift_start": shift_start.isoformat(),
            "total_alerts": total,
            "open_alerts": total - resolved_count,
            "resolved_alerts": resolved_count,
            "resolution_rate_pct": resolution_rate,
            "avg_resolution_time_min": avg_resolution_min,
            "by_severity": by_severity,
            "critical_open": sum(
                1 for a in shift_alerts
                if a.get("severity") == "CRITICAL" and a.get("status") == "open"
            ),
        }

    @property
    def all_alerts(self) -> list[dict[str, Any]]:
        """Return all alerts (active + resolved)."""
        return list(self._alerts.values())

    def __len__(self) -> int:
        return len(self._alerts)


def generate_osha_incident_report(
    violation_log: list[dict[str, Any]],
    site_info: dict[str, str],
    report_date: Optional[str] = None,
) -> str:
    """
    Generate an OSHA 300 Log-compatible incident report as a formatted string.

    OSHA 300 Log tracks: case#, employee name, job title, date of injury,
    where event occurred, description, days away, type of case.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    site_name = site_info.get("site_name", "Facility")
    site_address = site_info.get("address", "N/A")
    establishment = site_info.get("establishment", "N/A")
    naics_code = site_info.get("naics_code", "N/A")

    report_lines: list[str] = [
        "=" * 80,
        "OSHA FORM 300 — LOG OF WORK-RELATED INJURIES AND ILLNESSES",
        "PPE Compliance Monitoring System — Automated Report",
        "=" * 80,
        f"",
        f"Establishment Name : {site_name}",
        f"Address            : {site_address}",
        f"NAICS Code         : {naics_code}",
        f"Report Period      : {report_date}",
        f"Generated          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        "-" * 80,
        f"{'Case#':<8} {'Worker ID':<12} {'Date':<12} {'Time':<8} {'Zone':<14} "
        f"{'Missing PPE':<30} {'Severity':<10} {'OSHA Code':<22} {'Status':<10}",
        "-" * 80,
    ]

    # Filter to significant violations only
    reportable = [
        v for v in violation_log
        if v.get("severity") in ("CRITICAL", "HIGH", "MEDIUM")
    ]

    for i, v in enumerate(reportable, start=1):
        case_num = f"{i:04d}"
        worker_id = v.get("worker_id", "UNK")[:11]
        date = v.get("date", report_date)[:12]
        time_str = v.get("time", "00:00:00")[:8]
        zone = v.get("zone", "N/A")[:13]
        missing = ", ".join(v.get("missing_ppe", []))[:29]
        severity = v.get("severity", "N/A")[:9]
        osha_code = v.get("osha_code", "N/A")[:21]
        status = v.get("status", "open")[:9]

        report_lines.append(
            f"{case_num:<8} {worker_id:<12} {date:<12} {time_str:<8} {zone:<14} "
            f"{missing:<30} {severity:<10} {osha_code:<22} {status:<10}"
        )

    report_lines.extend([
        "-" * 80,
        f"",
        f"SUMMARY STATISTICS:",
        f"  Total incidents reported : {len(reportable)}",
        f"  CRITICAL                 : {sum(1 for v in reportable if v.get('severity') == 'CRITICAL')}",
        f"  HIGH                     : {sum(1 for v in reportable if v.get('severity') == 'HIGH')}",
        f"  MEDIUM                   : {sum(1 for v in reportable if v.get('severity') == 'MEDIUM')}",
        f"  Resolved                 : {sum(1 for v in reportable if v.get('status') == 'resolved')}",
        f"  Open                     : {sum(1 for v in reportable if v.get('status') == 'open')}",
        f"",
        f"COMPLIANCE NOTES:",
        f"  - This report is generated by automated PPE detection AI system.",
        f"  - All CRITICAL and HIGH severity incidents require supervisor review.",
        f"  - Per OSHA 29 CFR 1910.132: Employer must provide and ensure use of PPE.",
        f"  - Workers with 3+ violations in one shift must receive retraining.",
        f"",
        "=" * 80,
        f"Authorized by: Safety Management System | {site_name}",
        "=" * 80,
    ])

    return "\n".join(report_lines)

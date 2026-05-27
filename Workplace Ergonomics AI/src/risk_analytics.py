"""
Workplace Ergonomics & Injury Prevention Platform
Risk Analytics Module - session analysis, cumulative exposure, fatigue, reporting
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Keypoints = Dict[str, Dict[str, float]]
RebaResult = Dict[str, Any]


def analyze_workstation_risk(
    keypoints_series: List[Keypoints],
    worker_id: str,
    timestamp_series: List[float],
) -> Dict[str, Any]:
    """
    Processes a sequence of poses for one worker session and computes risk profile.

    Args:
        keypoints_series: List of keypoint dicts (one per frame/observation).
        worker_id: Unique identifier for the worker.
        timestamp_series: Corresponding timestamps in seconds.

    Returns:
        Session risk analysis dict with per-frame REBA scores, cumulative exposure,
        peak risk, fatigue trend, and high-risk timestamps.
    """
    from reba_rula import compute_reba_score  # type: ignore

    if len(keypoints_series) != len(timestamp_series):
        raise ValueError("keypoints_series and timestamp_series must have the same length.")

    reba_scores: List[int] = []
    reba_results: List[RebaResult] = []

    for kps in keypoints_series:
        result = compute_reba_score(kps)
        reba_scores.append(result["reba_score"])
        reba_results.append(result)

    cumulative_exposure = compute_cumulative_exposure(reba_scores, timestamp_series)
    fatigue_info = compute_posture_fatigue(reba_scores, timestamp_series)
    high_risk_moments = identify_high_risk_moments(reba_scores, timestamp_series, threshold=7)

    return {
        "worker_id": worker_id,
        "session_start": timestamp_series[0] if timestamp_series else 0.0,
        "session_end": timestamp_series[-1] if timestamp_series else 0.0,
        "session_duration_sec": (timestamp_series[-1] - timestamp_series[0]) if len(timestamp_series) > 1 else 0.0,
        "reba_scores": reba_scores,
        "reba_results": reba_results,
        "peak_reba": max(reba_scores) if reba_scores else 0,
        "avg_reba": round(float(np.mean(reba_scores)), 2) if reba_scores else 0.0,
        "cumulative_exposure": cumulative_exposure,
        "fatigue_info": fatigue_info,
        "high_risk_moments": high_risk_moments,
        "n_observations": len(reba_scores),
    }


def compute_cumulative_exposure(
    reba_scores_series: List[int],
    timestamps: List[float],
) -> Dict[str, float]:
    """
    Computes cumulative time (minutes) spent at each REBA risk level.

    Args:
        reba_scores_series: List of integer REBA scores per observation.
        timestamps: Corresponding timestamps in seconds.

    Returns:
        Dict with risk_level → cumulative_minutes.
    """
    from reba_rula import get_reba_risk_level  # type: ignore

    exposure: Dict[str, float] = {
        "negligible": 0.0,
        "low": 0.0,
        "medium": 0.0,
        "high": 0.0,
        "very_high": 0.0,
    }

    if len(reba_scores_series) == 0:
        return exposure

    # Integrate time between consecutive observations
    for i in range(len(reba_scores_series)):
        score = reba_scores_series[i]
        risk = get_reba_risk_level(score)

        if i < len(timestamps) - 1:
            dt_sec = abs(timestamps[i + 1] - timestamps[i])
        else:
            # Last observation: assume same interval as the previous
            dt_sec = abs(timestamps[-1] - timestamps[-2]) if len(timestamps) > 1 else 1.0

        exposure[risk] = exposure.get(risk, 0.0) + dt_sec / 60.0

    return {k: round(v, 3) for k, v in exposure.items()}


def compute_posture_fatigue(
    reba_scores_series: List[int],
    timestamps: List[float],
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Detects degradation in posture quality over a shift using linear regression trend.

    Args:
        reba_scores_series: List of REBA scores over time.
        timestamps: Corresponding timestamps in seconds.
        window_size: Number of observations to average for smoothing.

    Returns:
        Dict with slope (positive = deteriorating), is_fatiguing, trend_label.
    """
    if len(reba_scores_series) < 4:
        return {
            "slope": 0.0,
            "is_fatiguing": False,
            "trend_label": "insufficient_data",
            "r_squared": 0.0,
        }

    scores = np.array(reba_scores_series, dtype=np.float64)
    times = np.array(timestamps, dtype=np.float64)

    # Normalize times to 0-1
    t_range = times[-1] - times[0]
    if t_range == 0:
        t_norm = np.linspace(0, 1, len(times))
    else:
        t_norm = (times - times[0]) / t_range

    # Linear regression
    coeffs = np.polyfit(t_norm, scores, 1)
    slope = float(coeffs[0])

    # R² for quality
    predicted = np.polyval(coeffs, t_norm)
    ss_res = float(np.sum((scores - predicted) ** 2))
    ss_tot = float(np.sum((scores - np.mean(scores)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Fatigue threshold: positive slope > 0.5 points per unit time
    is_fatiguing = slope > 0.5

    if slope > 1.5:
        trend_label = "significant_deterioration"
    elif slope > 0.5:
        trend_label = "mild_deterioration"
    elif slope < -0.5:
        trend_label = "improvement"
    else:
        trend_label = "stable"

    return {
        "slope": round(slope, 4),
        "is_fatiguing": is_fatiguing,
        "trend_label": trend_label,
        "r_squared": round(r_squared, 4),
        "start_avg_reba": round(float(np.mean(scores[:window_size])), 2),
        "end_avg_reba": round(float(np.mean(scores[-window_size:])), 2),
    }


def identify_high_risk_moments(
    reba_scores: List[int],
    timestamps: List[float],
    threshold: int = 7,
) -> List[Dict[str, Any]]:
    """
    Returns timestamps where REBA score exceeds the threshold.

    Args:
        reba_scores: List of REBA scores per observation.
        timestamps: Corresponding timestamps in seconds.
        threshold: REBA score above which a moment is flagged.

    Returns:
        List of dicts: {timestamp, reba_score, risk_level}.
    """
    from reba_rula import get_reba_risk_level  # type: ignore

    high_risk: List[Dict[str, Any]] = []
    for i, score in enumerate(reba_scores):
        if score > threshold:
            ts = timestamps[i] if i < len(timestamps) else float(i)
            high_risk.append({
                "timestamp": round(ts, 2),
                "reba_score": score,
                "risk_level": get_reba_risk_level(score),
            })
    return high_risk


def compute_shift_summary(
    worker_sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes a comprehensive shift summary across all sessions for a worker.

    Args:
        worker_sessions: List of session analysis dicts from analyze_workstation_risk().

    Returns:
        Shift summary with total exposure, peak REBA, improvement vs. yesterday.
    """
    if not worker_sessions:
        return {
            "total_exposure_minutes": {},
            "peak_reba": 0,
            "avg_reba": 0.0,
            "time_in_high_risk_pct": 0.0,
            "improvement_vs_yesterday": 0.0,
            "n_high_risk_events": 0,
        }

    all_reba: List[int] = []
    total_exposure: Dict[str, float] = {
        "negligible": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0, "very_high": 0.0
    }

    for session in worker_sessions:
        all_reba.extend(session.get("reba_scores", []))
        exp = session.get("cumulative_exposure", {})
        for k in total_exposure:
            total_exposure[k] = total_exposure[k] + exp.get(k, 0.0)

    peak_reba = max(all_reba) if all_reba else 0
    avg_reba = float(np.mean(all_reba)) if all_reba else 0.0

    total_min = sum(total_exposure.values())
    high_risk_min = total_exposure.get("high", 0.0) + total_exposure.get("very_high", 0.0)
    time_in_high_risk_pct = (high_risk_min / total_min * 100) if total_min > 0 else 0.0

    n_high_risk_events = sum(
        len(s.get("high_risk_moments", [])) for s in worker_sessions
    )

    # Improvement vs. yesterday: compare first vs. last session avg
    improvement = 0.0
    if len(worker_sessions) >= 2:
        first_avg = worker_sessions[0].get("avg_reba", 0)
        last_avg = worker_sessions[-1].get("avg_reba", 0)
        improvement = round(first_avg - last_avg, 2)  # Positive = improvement

    return {
        "total_exposure_minutes": {k: round(v, 2) for k, v in total_exposure.items()},
        "peak_reba": peak_reba,
        "avg_reba": round(avg_reba, 2),
        "time_in_high_risk_pct": round(time_in_high_risk_pct, 1),
        "improvement_vs_yesterday": improvement,
        "n_high_risk_events": n_high_risk_events,
        "total_time_minutes": round(total_min, 2),
    }


def compare_job_roles(
    sessions_by_role: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Computes average risk profile by job role across multiple worker sessions.

    Args:
        sessions_by_role: Dict of role_name → list of session analysis dicts.

    Returns:
        Dict of role_name → {avg_reba, peak_reba, time_in_high_risk_pct, n_workers}.
    """
    role_profiles: Dict[str, Dict[str, Any]] = {}

    for role, sessions in sessions_by_role.items():
        if not sessions:
            continue

        all_reba: List[int] = []
        total_high_risk_min = 0.0
        total_min = 0.0

        for session in sessions:
            all_reba.extend(session.get("reba_scores", []))
            exp = session.get("cumulative_exposure", {})
            total_min += sum(exp.values())
            total_high_risk_min += exp.get("high", 0.0) + exp.get("very_high", 0.0)

        role_profiles[role] = {
            "avg_reba": round(float(np.mean(all_reba)), 2) if all_reba else 0.0,
            "peak_reba": max(all_reba) if all_reba else 0,
            "time_in_high_risk_pct": round(
                (total_high_risk_min / total_min * 100) if total_min > 0 else 0.0, 1
            ),
            "n_workers": len(sessions),
            "n_observations": len(all_reba),
        }

    return role_profiles


def generate_ergonomics_report(
    worker_id: str,
    shift_summary: Dict[str, Any],
    high_risk_moments: List[Dict[str, Any]],
) -> str:
    """
    Generates a formatted ergonomics assessment report string.

    Args:
        worker_id: Worker identifier.
        shift_summary: Shift summary from compute_shift_summary().
        high_risk_moments: High-risk moments from identify_high_risk_moments().

    Returns:
        Multi-line formatted report string.
    """
    from reba_rula import get_reba_risk_level, get_reba_action  # type: ignore

    peak = shift_summary.get("peak_reba", 0)
    avg = shift_summary.get("avg_reba", 0)
    high_pct = shift_summary.get("time_in_high_risk_pct", 0)
    improvement = shift_summary.get("improvement_vs_yesterday", 0)
    exposure = shift_summary.get("total_exposure_minutes", {})

    risk = get_reba_risk_level(int(round(avg)))
    recommendation = get_reba_action(risk)

    lines = [
        "=" * 65,
        f"  ERGONOMICS ASSESSMENT REPORT — Worker: {worker_id}",
        "=" * 65,
        "",
        "SHIFT SUMMARY",
        "-" * 40,
        f"  Average REBA Score  : {avg:.2f} ({risk.upper()} risk)",
        f"  Peak REBA Score     : {peak}",
        f"  Time at High Risk   : {high_pct:.1f}%",
        f"  Improvement vs Yesterday: {'+' if improvement > 0 else ''}{improvement:.2f} REBA points",
        "",
        "CUMULATIVE EXPOSURE (minutes)",
        "-" * 40,
    ]

    for level in ["negligible", "low", "medium", "high", "very_high"]:
        mins = exposure.get(level, 0.0)
        total = sum(exposure.values()) or 1
        pct = mins / total * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {level:<12}: {mins:6.1f} min ({pct:4.1f}%) {bar}")

    lines += [
        "",
        "HIGH-RISK EVENTS",
        "-" * 40,
    ]

    if high_risk_moments:
        for evt in high_risk_moments[:10]:
            lines.append(
                f"  t={evt['timestamp']:8.1f}s  REBA={evt['reba_score']:2d}  ({evt['risk_level']})"
            )
        if len(high_risk_moments) > 10:
            lines.append(f"  ... and {len(high_risk_moments) - 10} more high-risk events")
    else:
        lines.append("  No high-risk events detected during this shift.")

    lines += [
        "",
        "RECOMMENDED ACTION",
        "-" * 40,
        f"  {recommendation}",
        "",
        "ERGONOMIC RECOMMENDATIONS",
        "-" * 40,
    ]

    if avg > 7:
        lines += [
            "  1. Immediately review workstation design and task layout.",
            "  2. Introduce mechanical lifting aids for loads > 5kg.",
            "  3. Implement mandatory rest breaks every 30 minutes.",
            "  4. Schedule ergonomics specialist assessment within 48 hours.",
        ]
    elif avg > 4:
        lines += [
            "  1. Review and optimize workstation height and reach distances.",
            "  2. Provide anti-fatigue mats and supportive footwear guidance.",
            "  3. Implement job rotation to distribute physical load.",
            "  4. Schedule follow-up assessment within 2 weeks.",
        ]
    else:
        lines += [
            "  1. Continue current workstation setup.",
            "  2. Maintain regular monitoring cadence.",
            "  3. Review posture coaching materials with worker.",
        ]

    lines += ["", "=" * 65]
    return "\n".join(lines)

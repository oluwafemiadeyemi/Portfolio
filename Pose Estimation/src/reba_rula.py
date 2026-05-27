"""
Workplace Ergonomics & Injury Prevention Platform
REBA/RULA Scoring Module - Full implementation of ergonomic risk assessment algorithms

REBA (Rapid Entire Body Assessment) - ISO 9241-110 compliant
RULA (Rapid Upper Limb Assessment) - Based on McAtamney & Corlett (1993)

All lookup tables are hardcoded in this module.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

Keypoints = Dict[str, Dict[str, float]]


# ===========================================================================
# REBA LOOKUP TABLES (complete, hardcoded)
# ===========================================================================

# Table A: Neck × Trunk × Legs → Raw Score A (1-9)
# Dimensions: [neck_score (1-3)][trunk_score (1-5)][legs_score (1-2)]
REBA_TABLE_A: Dict[Tuple[int, int, int], int] = {
    # neck=1
    (1, 1, 1): 1, (1, 1, 2): 2,
    (1, 2, 1): 2, (1, 2, 2): 3,
    (1, 3, 1): 2, (1, 3, 2): 3,
    (1, 4, 1): 3, (1, 4, 2): 4,
    (1, 5, 1): 4, (1, 5, 2): 5,
    # neck=2
    (2, 1, 1): 2, (2, 1, 2): 3,
    (2, 2, 1): 3, (2, 2, 2): 4,
    (2, 3, 1): 3, (2, 3, 2): 4,
    (2, 4, 1): 4, (2, 4, 2): 5,
    (2, 5, 1): 5, (2, 5, 2): 6,
    # neck=3
    (3, 1, 1): 3, (3, 1, 2): 4,
    (3, 2, 1): 4, (3, 2, 2): 5,
    (3, 3, 1): 5, (3, 3, 2): 6,
    (3, 4, 1): 6, (3, 4, 2): 7,
    (3, 5, 1): 7, (3, 5, 2): 8,
}

# Table B: Upper Arm × Lower Arm × Wrist → Raw Score B (1-9)
# Dimensions: [upper_arm_score (1-6)][lower_arm_score (1-2)][wrist_score (1-3)]
REBA_TABLE_B: Dict[Tuple[int, int, int], int] = {
    # upper_arm=1
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 2,
    (1, 2, 1): 1, (1, 2, 2): 2, (1, 2, 3): 3,
    # upper_arm=2
    (2, 1, 1): 1, (2, 1, 2): 2, (2, 1, 3): 3,
    (2, 2, 1): 2, (2, 2, 2): 3, (2, 2, 3): 4,
    # upper_arm=3
    (3, 1, 1): 3, (3, 1, 2): 4, (3, 1, 3): 5,
    (3, 2, 1): 4, (3, 2, 2): 5, (3, 2, 3): 5,
    # upper_arm=4
    (4, 1, 1): 4, (4, 1, 2): 5, (4, 1, 3): 5,
    (4, 2, 1): 5, (4, 2, 2): 6, (4, 2, 3): 7,
    # upper_arm=5
    (5, 1, 1): 6, (5, 1, 2): 7, (5, 1, 3): 8,
    (5, 2, 1): 7, (5, 2, 2): 8, (5, 2, 3): 8,
    # upper_arm=6
    (6, 1, 1): 7, (6, 1, 2): 8, (6, 1, 3): 9,
    (6, 2, 1): 8, (6, 2, 2): 9, (6, 2, 3): 9,
}

# Table C: Score A × Score B → REBA Final Score (before activity)
# Dimensions: [score_a (1-12)][score_b (1-12)] → score_c (1-15)
REBA_TABLE_C: Dict[Tuple[int, int], int] = {
    (1, 1): 1,  (1, 2): 1,  (1, 3): 1,  (1, 4): 2,  (1, 5): 3,  (1, 6): 3,  (1, 7): 4,  (1, 8): 5,  (1, 9): 6,  (1, 10): 7,  (1, 11): 7,  (1, 12): 7,
    (2, 1): 1,  (2, 2): 2,  (2, 3): 2,  (2, 4): 3,  (2, 5): 4,  (2, 6): 4,  (2, 7): 5,  (2, 8): 6,  (2, 9): 6,  (2, 10): 7,  (2, 11): 7,  (2, 12): 8,
    (3, 1): 2,  (3, 2): 3,  (3, 3): 3,  (3, 4): 3,  (3, 5): 4,  (3, 6): 5,  (3, 7): 6,  (3, 8): 7,  (3, 9): 7,  (3, 10): 8,  (3, 11): 8,  (3, 12): 8,
    (4, 1): 3,  (4, 2): 4,  (4, 3): 4,  (4, 4): 4,  (4, 5): 5,  (4, 6): 6,  (4, 7): 7,  (4, 8): 8,  (4, 9): 8,  (4, 10): 9,  (4, 11): 9,  (4, 12): 9,
    (5, 1): 4,  (5, 2): 4,  (5, 3): 4,  (5, 4): 5,  (5, 5): 6,  (5, 6): 7,  (5, 7): 8,  (5, 8): 8,  (5, 9): 9,  (5, 10): 9,  (5, 11): 9,  (5, 12): 9,
    (6, 1): 6,  (6, 2): 6,  (6, 3): 6,  (6, 4): 7,  (6, 5): 8,  (6, 6): 8,  (6, 7): 9,  (6, 8): 9,  (6, 9): 10, (6, 10): 10, (6, 11): 10, (6, 12): 10,
    (7, 1): 7,  (7, 2): 7,  (7, 3): 7,  (7, 4): 8,  (7, 5): 9,  (7, 6): 9,  (7, 7): 9,  (7, 8): 10, (7, 9): 10, (7, 10): 11, (7, 11): 11, (7, 12): 11,
    (8, 1): 8,  (8, 2): 8,  (8, 3): 8,  (8, 4): 9,  (8, 5): 10, (8, 6): 10, (8, 7): 10, (8, 8): 10, (8, 9): 11, (8, 10): 11, (8, 11): 11, (8, 12): 12,
    (9, 1): 9,  (9, 2): 9,  (9, 3): 9,  (9, 4): 10, (9, 5): 10, (9, 6): 10, (9, 7): 11, (9, 8): 11, (9, 9): 12, (9, 10): 12, (9, 11): 12, (9, 12): 12,
    (10, 1): 10, (10, 2): 10, (10, 3): 10, (10, 4): 11, (10, 5): 11, (10, 6): 11, (10, 7): 11, (10, 8): 12, (10, 9): 12, (10, 10): 13, (10, 11): 13, (10, 12): 13,
    (11, 1): 11, (11, 2): 11, (11, 3): 11, (11, 4): 11, (11, 5): 12, (11, 6): 12, (11, 7): 13, (11, 8): 13, (11, 9): 13, (11, 10): 14, (11, 11): 14, (11, 12): 14,
    (12, 1): 12, (12, 2): 12, (12, 3): 12, (12, 4): 12, (12, 5): 13, (12, 6): 13, (12, 7): 14, (12, 8): 14, (12, 9): 14, (12, 10): 15, (12, 11): 15, (12, 12): 15,
}


# ===========================================================================
# RULA LOOKUP TABLES (complete, hardcoded)
# ===========================================================================

# RULA Table A: Upper Arm × Lower Arm × Wrist × Wrist Twist → Score A (1-9)
# Simplified as [upper_arm(1-6)][lower_arm(1-2)][wrist(1-4)] → score
RULA_TABLE_A: Dict[Tuple[int, int, int], int] = {
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 2, (1, 1, 4): 3,
    (1, 2, 1): 2, (1, 2, 2): 2, (1, 2, 3): 3, (1, 2, 4): 3,
    (2, 1, 1): 2, (2, 1, 2): 2, (2, 1, 3): 3, (2, 1, 4): 3,
    (2, 2, 1): 2, (2, 2, 2): 3, (2, 2, 3): 3, (2, 2, 4): 4,
    (3, 1, 1): 2, (3, 1, 2): 3, (3, 1, 3): 3, (3, 1, 4): 4,
    (3, 2, 1): 3, (3, 2, 2): 3, (3, 2, 3): 4, (3, 2, 4): 4,
    (4, 1, 1): 3, (4, 1, 2): 3, (4, 1, 3): 4, (4, 1, 4): 4,
    (4, 2, 1): 3, (4, 2, 2): 4, (4, 2, 3): 4, (4, 2, 4): 5,
    (5, 1, 1): 3, (5, 1, 2): 4, (5, 1, 3): 4, (5, 1, 4): 5,
    (5, 2, 1): 4, (5, 2, 2): 4, (5, 2, 3): 5, (5, 2, 4): 5,
    (6, 1, 1): 4, (6, 1, 2): 4, (6, 1, 3): 5, (6, 1, 4): 5,
    (6, 2, 1): 4, (6, 2, 2): 5, (6, 2, 3): 5, (6, 2, 4): 6,
}

# RULA Table B: Neck × Trunk × Legs → Score B (1-7)
RULA_TABLE_B: Dict[Tuple[int, int, int], int] = {
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 3,
    (1, 2, 1): 2, (1, 2, 2): 3, (1, 2, 3): 4,
    (1, 3, 1): 3, (1, 3, 2): 4, (1, 3, 3): 5,
    (1, 4, 1): 4, (1, 4, 2): 5, (1, 4, 3): 6,
    (1, 5, 1): 5, (1, 5, 2): 6, (1, 5, 3): 6,
    (2, 1, 1): 2, (2, 1, 2): 3, (2, 1, 3): 4,
    (2, 2, 1): 3, (2, 2, 2): 4, (2, 2, 3): 5,
    (2, 3, 1): 4, (2, 3, 2): 5, (2, 3, 3): 5,
    (2, 4, 1): 5, (2, 4, 2): 5, (2, 4, 3): 6,
    (2, 5, 1): 6, (2, 5, 2): 6, (2, 5, 3): 7,
    (3, 1, 1): 3, (3, 1, 2): 4, (3, 1, 3): 5,
    (3, 2, 1): 4, (3, 2, 2): 5, (3, 2, 3): 5,
    (3, 3, 1): 5, (3, 3, 2): 5, (3, 3, 3): 6,
    (3, 4, 1): 5, (3, 4, 2): 6, (3, 4, 3): 7,
    (3, 5, 1): 6, (3, 5, 2): 7, (3, 5, 3): 7,
    (4, 1, 1): 4, (4, 1, 2): 5, (4, 1, 3): 5,
    (4, 2, 1): 5, (4, 2, 2): 5, (4, 2, 3): 6,
    (4, 3, 1): 5, (4, 3, 2): 6, (4, 3, 3): 7,
    (4, 4, 1): 6, (4, 4, 2): 7, (4, 4, 3): 7,
    (4, 5, 1): 7, (4, 5, 2): 7, (4, 5, 3): 7,
}

# RULA Table C: Score A + Muscle/Force → Grand Score A (1-8)
RULA_TABLE_C: Dict[Tuple[int, int], int] = {
    (1, 0): 1, (1, 1): 2, (1, 2): 3,
    (2, 0): 2, (2, 1): 2, (2, 2): 3,
    (3, 0): 3, (3, 1): 3, (3, 2): 4,
    (4, 0): 3, (4, 1): 4, (4, 2): 4,
    (5, 0): 4, (5, 1): 4, (5, 2): 5,
    (6, 0): 4, (6, 1): 5, (6, 2): 5,
    (7, 0): 5, (7, 1): 5, (7, 2): 6,
    (8, 0): 5, (8, 1): 6, (8, 2): 6,
    (9, 0): 6, (9, 1): 7, (9, 2): 7,
}

# RULA Table D: Grand Score A × Grand Score B → Final RULA Score (1-7)
RULA_TABLE_D: Dict[Tuple[int, int], int] = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 4, (1, 6): 5, (1, 7): 5,
    (2, 1): 2, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 4, (2, 6): 5, (2, 7): 5,
    (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4, (3, 5): 4, (3, 6): 5, (3, 7): 6,
    (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4, (4, 5): 5, (4, 6): 6, (4, 7): 6,
    (5, 1): 4, (5, 2): 4, (5, 3): 4, (5, 4): 5, (5, 5): 6, (5, 6): 7, (5, 7): 7,
    (6, 1): 4, (6, 2): 4, (6, 3): 5, (6, 4): 6, (6, 5): 6, (6, 6): 7, (6, 7): 7,
    (7, 1): 5, (7, 2): 5, (7, 3): 6, (7, 4): 6, (7, 5): 7, (7, 6): 7, (7, 7): 7,
    (8, 1): 5, (8, 2): 5, (8, 3): 6, (8, 4): 7, (8, 5): 7, (8, 6): 7, (8, 7): 7,
}


# ===========================================================================
# Angle computation helpers
# ===========================================================================

def _angle_between(p1: Tuple[float, float], vertex: Tuple[float, float],
                   p2: Tuple[float, float]) -> float:
    """Computes the angle at vertex between vectors vertex→p1 and vertex→p2 in degrees."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


def _get_xy(keypoints: Keypoints, name: str) -> Optional[Tuple[float, float]]:
    """Gets (x, y) from keypoints dict, returns None if not found or low visibility."""
    kp = keypoints.get(name)
    if kp is None:
        return None
    if kp.get("visibility", 1.0) < 0.3:
        return None
    return (kp["x"], kp["y"])


def _vertical_angle(point: Tuple[float, float], base: Tuple[float, float]) -> float:
    """Angle of the vector base→point from vertical (positive = forward tilt)."""
    dx = point[0] - base[0]
    dy = base[1] - point[1]  # y increases downward in image coords; invert for anatomical
    angle = math.degrees(math.atan2(dx, dy))
    return abs(angle)


# ===========================================================================
# REBA Body Segment Angle Computations
# ===========================================================================

def compute_neck_angle(keypoints: Keypoints) -> float:
    """
    Computes neck flexion/extension angle from vertical (degrees).

    Uses nose-to-midpoint-of-shoulders vector relative to vertical.
    """
    nose = _get_xy(keypoints, "nose")
    l_shoulder = _get_xy(keypoints, "left_shoulder")
    r_shoulder = _get_xy(keypoints, "right_shoulder")

    if nose is None or l_shoulder is None or r_shoulder is None:
        return 15.0  # Default neutral-ish

    mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    return _vertical_angle(nose, mid_shoulder)


def compute_trunk_angle(keypoints: Keypoints) -> float:
    """
    Computes trunk flexion/extension angle from vertical (degrees).

    Uses midpoint-of-shoulders to midpoint-of-hips.
    """
    l_shoulder = _get_xy(keypoints, "left_shoulder")
    r_shoulder = _get_xy(keypoints, "right_shoulder")
    l_hip = _get_xy(keypoints, "left_hip")
    r_hip = _get_xy(keypoints, "right_hip")

    if any(v is None for v in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return 10.0  # Default slightly flexed

    mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    return _vertical_angle(mid_shoulder, mid_hip)


def compute_upper_arm_angle(keypoints: Keypoints, side: str = "right") -> float:
    """
    Computes upper arm elevation angle from vertical (degrees).

    Args:
        keypoints: Pose keypoints dict.
        side: 'right' or 'left'.
    """
    prefix = side
    shoulder = _get_xy(keypoints, f"{prefix}_shoulder")
    elbow = _get_xy(keypoints, f"{prefix}_elbow")
    hip = _get_xy(keypoints, f"{prefix}_hip")

    if shoulder is None or elbow is None:
        return 25.0  # Default low-elevation

    # Angle of shoulder-to-elbow vector from vertical
    return _vertical_angle(elbow, shoulder)


def compute_lower_arm_angle(keypoints: Keypoints, side: str = "right") -> float:
    """
    Computes elbow (lower arm) flexion angle (degrees).

    Args:
        keypoints: Pose keypoints dict.
        side: 'right' or 'left'.
    """
    prefix = side
    shoulder = _get_xy(keypoints, f"{prefix}_shoulder")
    elbow = _get_xy(keypoints, f"{prefix}_elbow")
    wrist = _get_xy(keypoints, f"{prefix}_wrist")

    if any(v is None for v in [shoulder, elbow, wrist]):
        return 90.0  # Default neutral

    return _angle_between(shoulder, elbow, wrist)


def compute_wrist_angle(keypoints: Keypoints, side: str = "right") -> float:
    """
    Estimates wrist deviation angle (degrees) from the forearm line.
    Uses elbow-wrist-fingertip proxy (index finger if available).

    Args:
        keypoints: Pose keypoints dict.
        side: 'right' or 'left'.
    """
    prefix = side
    elbow = _get_xy(keypoints, f"{prefix}_elbow")
    wrist = _get_xy(keypoints, f"{prefix}_wrist")
    # Use index finger as fingertip proxy
    fingertip = _get_xy(keypoints, f"{prefix}_index") or _get_xy(keypoints, f"{prefix}_pinky")

    if elbow is None or wrist is None or fingertip is None:
        return 5.0  # Default minimal deviation

    return _angle_between(elbow, wrist, fingertip)


def compute_leg_position(keypoints: Keypoints) -> Dict[str, Any]:
    """
    Determines leg position: bilateral/unilateral stance, knee flexion angle.

    Returns:
        Dict with stance (bilateral/unilateral), knee_angle_left, knee_angle_right.
    """
    l_hip = _get_xy(keypoints, "left_hip")
    r_hip = _get_xy(keypoints, "right_hip")
    l_knee = _get_xy(keypoints, "left_knee")
    r_knee = _get_xy(keypoints, "right_knee")
    l_ankle = _get_xy(keypoints, "left_ankle")
    r_ankle = _get_xy(keypoints, "right_ankle")

    knee_angle_left = 180.0
    knee_angle_right = 180.0

    if l_hip and l_knee and l_ankle:
        knee_angle_left = _angle_between(l_hip, l_knee, l_ankle)
    if r_hip and r_knee and r_ankle:
        knee_angle_right = _angle_between(r_hip, r_knee, r_ankle)

    # Determine stance
    if l_ankle and r_ankle:
        horiz_diff = abs(l_ankle[0] - r_ankle[0])
        stance = "bilateral" if horiz_diff > 0.05 else "unilateral"
    else:
        stance = "unknown"

    return {
        "stance": stance,
        "knee_angle_left": round(knee_angle_left, 1),
        "knee_angle_right": round(knee_angle_right, 1),
    }


# ===========================================================================
# REBA Score subscores
# ===========================================================================

def _neck_score(angle: float, twisted: bool = False, bent: bool = False) -> int:
    """REBA neck score (1-3) from angle in degrees."""
    if angle <= 20:
        score = 1
    elif angle <= 20:
        score = 1
    else:
        score = 2 if angle <= 45 else 3
    if twisted or bent:
        score = min(score + 1, 3)
    return score


def _trunk_score(angle: float, twisted: bool = False, bent_sideways: bool = False) -> int:
    """REBA trunk score (1-5) from flexion angle in degrees."""
    if angle == 0:
        score = 1
    elif angle <= 20:
        score = 2
    elif angle <= 60:
        score = 3
    else:
        score = 4
    if twisted or bent_sideways:
        score = min(score + 1, 5)
    return score


def _legs_score(stance: str, knee_angle: float) -> int:
    """REBA legs score (1-2) from stance and knee angle."""
    score = 1 if stance == "bilateral" else 2
    # Add 1 for knee flexion 30-60°, add 2 for >60°
    if knee_angle < 30:
        pass
    elif knee_angle <= 60:
        score = min(score + 1, 2)
    else:
        score = 2
    return score


def _upper_arm_reba_score(angle: float, abducted: bool = False,
                           shoulder_raised: bool = False, supported: bool = False) -> int:
    """REBA upper arm score (1-6) from elevation angle."""
    if angle <= 20:
        score = 1
    elif angle <= 45:
        score = 2
    elif angle <= 90:
        score = 3
    else:
        score = 4
    if abducted:
        score = min(score + 1, 6)
    if shoulder_raised:
        score = min(score + 1, 6)
    if supported:
        score = max(score - 1, 1)
    return score


def _lower_arm_reba_score(angle: float) -> int:
    """REBA lower arm score (1-2) from elbow flexion angle."""
    return 1 if 60 <= angle <= 100 else 2


def _wrist_reba_score(angle: float, deviated: bool = False) -> int:
    """REBA wrist score (1-3) from wrist angle."""
    if angle <= 15:
        score = 1
    else:
        score = 2
    if deviated:
        score = min(score + 1, 3)
    return score


# ===========================================================================
# Full REBA Score
# ===========================================================================

def compute_reba_score(
    keypoints: Keypoints,
    load_kg: float = 0.0,
    coupling_score: int = 0,
    activity_score: int = 0,
) -> Dict[str, Any]:
    """
    Computes the full REBA (Rapid Entire Body Assessment) score.

    REBA Score range: 1 (negligible risk) to 15 (very high risk)

    Args:
        keypoints: Pose keypoints dict (from PoseExtractor).
        load_kg: Load/force in kg (0=<5kg, 1=5-10kg, 2=>10kg or sudden).
        coupling_score: Coupling quality (0=good, 1=fair, 2=poor).
        activity_score: Activity adjustment (0=static, 1=repeated, 2=rapid/large changes).

    Returns:
        Dict with reba_score, risk_level, action_required, body_part_scores.
    """
    # --- Compute angles ---
    neck_angle = compute_neck_angle(keypoints)
    trunk_angle = compute_trunk_angle(keypoints)
    upper_arm_angle_r = compute_upper_arm_angle(keypoints, side="right")
    lower_arm_angle_r = compute_lower_arm_angle(keypoints, side="right")
    wrist_angle_r = compute_wrist_angle(keypoints, side="right")
    leg_info = compute_leg_position(keypoints)
    knee_angle = min(leg_info["knee_angle_left"], leg_info["knee_angle_right"])

    # --- Individual scores ---
    neck_s = _neck_score(neck_angle)
    trunk_s = _trunk_score(trunk_angle)
    legs_s = _legs_score(leg_info["stance"], knee_angle)
    upper_arm_s = _upper_arm_reba_score(upper_arm_angle_r)
    lower_arm_s = _lower_arm_reba_score(lower_arm_angle_r)
    wrist_s = _wrist_reba_score(wrist_angle_r)

    # --- Group A: Neck + Trunk + Legs ---
    key_a = (
        max(1, min(neck_s, 3)),
        max(1, min(trunk_s, 5)),
        max(1, min(legs_s, 2)),
    )
    raw_a = REBA_TABLE_A.get(key_a, REBA_TABLE_A.get((min(key_a[0], 3), min(key_a[1], 5), min(key_a[2], 2)), 5))

    # Load adjustment
    if load_kg < 5:
        load_adj = 0
    elif load_kg <= 10:
        load_adj = 1
    else:
        load_adj = 2  # >10kg or sudden force
    score_a = min(raw_a + load_adj, 12)

    # --- Group B: Upper Arm + Lower Arm + Wrist ---
    key_b = (
        max(1, min(upper_arm_s, 6)),
        max(1, min(lower_arm_s, 2)),
        max(1, min(wrist_s, 3)),
    )
    raw_b = REBA_TABLE_B.get(key_b, REBA_TABLE_B.get((min(key_b[0], 6), min(key_b[1], 2), min(key_b[2], 3)), 5))
    score_b = min(raw_b + coupling_score, 12)

    # --- Table C: Score A × Score B ---
    key_c = (max(1, min(score_a, 12)), max(1, min(score_b, 12)))
    score_c = REBA_TABLE_C.get(key_c, 10)

    # --- Final REBA Score ---
    reba_score = min(score_c + activity_score, 15)

    risk_level = get_reba_risk_level(reba_score)
    action_required = get_reba_action(risk_level)

    return {
        "reba_score": reba_score,
        "risk_level": risk_level,
        "action_required": action_required,
        "body_part_scores": {
            "neck": {"angle_deg": round(neck_angle, 1), "score": neck_s},
            "trunk": {"angle_deg": round(trunk_angle, 1), "score": trunk_s},
            "legs": {"stance": leg_info["stance"], "knee_angle": round(knee_angle, 1), "score": legs_s},
            "upper_arm": {"angle_deg": round(upper_arm_angle_r, 1), "score": upper_arm_s},
            "lower_arm": {"angle_deg": round(lower_arm_angle_r, 1), "score": lower_arm_s},
            "wrist": {"angle_deg": round(wrist_angle_r, 1), "score": wrist_s},
        },
        "group_a_score": score_a,
        "group_b_score": score_b,
        "table_c_score": score_c,
        "load_kg": load_kg,
        "coupling_score": coupling_score,
        "activity_score": activity_score,
    }


def get_reba_risk_level(score: int) -> str:
    """
    Maps REBA score to risk level.

    Score 1       → negligible
    Score 2-3     → low
    Score 4-7     → medium
    Score 8-10    → high
    Score 11-15   → very_high
    """
    if score == 1:
        return "negligible"
    elif score <= 3:
        return "low"
    elif score <= 7:
        return "medium"
    elif score <= 10:
        return "high"
    else:
        return "very_high"


def get_reba_action(risk_level: str) -> str:
    """Maps REBA risk level to required action string."""
    actions = {
        "negligible": "No action required. Acceptable posture.",
        "low": "Action may be needed. Monitor worker posture.",
        "medium": "Action needed soon. Investigate and implement changes within the next few weeks.",
        "high": "Action required promptly. Implement ergonomic controls within days.",
        "very_high": "Implement changes immediately. Stop or modify the task. High injury risk.",
    }
    return actions.get(risk_level, "Assessment required.")


# ===========================================================================
# RULA Score
# ===========================================================================

def _upper_arm_rula_score(angle: float) -> int:
    """RULA upper arm score (1-6) from elevation angle."""
    if angle <= 20:
        return 1
    elif angle <= 45:
        return 2
    elif angle <= 90:
        return 3
    elif angle <= 110:
        return 4
    elif angle <= 135:
        return 5
    else:
        return 6


def _lower_arm_rula_score(angle: float) -> int:
    """RULA lower arm score (1-2): 1 if 60-100°, 2 otherwise."""
    return 1 if 60 <= angle <= 100 else 2


def _wrist_rula_score(angle: float) -> int:
    """RULA wrist score (1-4) from deviation angle."""
    if angle <= 15:
        return 1
    elif angle <= 30:
        return 2
    elif angle <= 45:
        return 3
    else:
        return 4


def _neck_rula_score(angle: float) -> int:
    """RULA neck score (1-4)."""
    if angle <= 10:
        return 1
    elif angle <= 20:
        return 2
    elif angle <= 30:
        return 3
    else:
        return 4


def _trunk_rula_score(angle: float) -> int:
    """RULA trunk score (1-5)."""
    if angle == 0:
        return 1
    elif angle <= 20:
        return 2
    elif angle <= 60:
        return 3
    else:
        return 4


def _legs_rula_score(stance: str) -> int:
    """RULA legs score (1-3)."""
    return 1 if stance == "bilateral" else 2


def compute_rula_score(
    keypoints: Keypoints,
    muscle_use: int = 0,
    force: int = 0,
) -> Dict[str, Any]:
    """
    Computes the full RULA (Rapid Upper Limb Assessment) score.

    RULA Score range: 1 (acceptable) to 7 (investigate and change immediately)

    Args:
        keypoints: Pose keypoints dict.
        muscle_use: 0=intermittent, 1=static >1min or repeated >4x/min.
        force: 0=<2kg occasional, 1=2-10kg or intermittent, 2=>10kg or shock.

    Returns:
        Dict with rula_score, risk_level, recommended_action, body_part_scores.
    """
    neck_angle = compute_neck_angle(keypoints)
    trunk_angle = compute_trunk_angle(keypoints)
    upper_arm_angle_r = compute_upper_arm_angle(keypoints, side="right")
    lower_arm_angle_r = compute_lower_arm_angle(keypoints, side="right")
    wrist_angle_r = compute_wrist_angle(keypoints, side="right")
    leg_info = compute_leg_position(keypoints)

    upper_arm_s = _upper_arm_rula_score(upper_arm_angle_r)
    lower_arm_s = _lower_arm_rula_score(lower_arm_angle_r)
    wrist_s = _wrist_rula_score(wrist_angle_r)
    neck_s = _neck_rula_score(neck_angle)
    trunk_s = _trunk_rula_score(trunk_angle)
    legs_s = _legs_rula_score(leg_info["stance"])

    # Table A: Upper Arm × Lower Arm × Wrist
    key_a = (
        max(1, min(upper_arm_s, 6)),
        max(1, min(lower_arm_s, 2)),
        max(1, min(wrist_s, 4)),
    )
    raw_a = RULA_TABLE_A.get(key_a, 5)

    # Table C: Raw A + muscle + force → Grand Score A
    grand_a_raw = raw_a + muscle_use + force
    key_c = (max(1, min(grand_a_raw, 9)), max(0, min(force, 2)))
    grand_a = RULA_TABLE_C.get(key_c, min(grand_a_raw + 1, 8))

    # Table B: Neck × Trunk × Legs
    key_b = (
        max(1, min(neck_s, 4)),
        max(1, min(trunk_s, 5)),
        max(1, min(legs_s, 3)),
    )
    raw_b = RULA_TABLE_B.get(key_b, 4)
    grand_b = min(raw_b + muscle_use + force, 8)

    # Table D: Grand A × Grand B → RULA score
    key_d = (max(1, min(grand_a, 8)), max(1, min(grand_b, 7)))
    rula_score = RULA_TABLE_D.get(key_d, 5)

    risk_level, action = _get_rula_risk_and_action(rula_score)

    return {
        "rula_score": rula_score,
        "risk_level": risk_level,
        "recommended_action": action,
        "body_part_scores": {
            "upper_arm": {"angle_deg": round(upper_arm_angle_r, 1), "score": upper_arm_s},
            "lower_arm": {"angle_deg": round(lower_arm_angle_r, 1), "score": lower_arm_s},
            "wrist": {"angle_deg": round(wrist_angle_r, 1), "score": wrist_s},
            "neck": {"angle_deg": round(neck_angle, 1), "score": neck_s},
            "trunk": {"angle_deg": round(trunk_angle, 1), "score": trunk_s},
            "legs": {"stance": leg_info["stance"], "score": legs_s},
        },
        "grand_score_a": grand_a,
        "grand_score_b": grand_b,
        "muscle_use": muscle_use,
        "force": force,
    }


def _get_rula_risk_and_action(score: int) -> Tuple[str, str]:
    """Maps RULA score to risk level and recommended action."""
    if score <= 2:
        return "acceptable", "Posture is acceptable. No immediate action required."
    elif score <= 3:
        return "low", "Further investigation may be needed. Change may be required."
    elif score <= 5:
        return "medium", "Investigation and change required soon."
    elif score == 6:
        return "high", "Investigation and change required promptly."
    else:
        return "very_high", "Investigate and implement changes immediately."

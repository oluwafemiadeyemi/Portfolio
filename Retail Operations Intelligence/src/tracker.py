"""
Retail Operations Intelligence & Loss Prevention Platform
Tracker Module - Simple multi-object tracker using IoU matching + Hungarian algorithm
No ByteTrack dependency; implemented from scratch using scipy.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]
Track = Dict[str, Any]


class SimpleTracker:
    """
    Simple multi-object tracker using IoU-based Hungarian algorithm matching.

    Attributes:
        max_age: Number of frames a track can be unmatched before deletion.
        min_hits: Minimum number of matched frames before a track is reported.
        iou_threshold: Minimum IoU for detection-track matching.
        tracks: List of currently active track dicts.
        frame_count: Total frames processed.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.frame_count: int = 0
        self._next_id: int = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new frame detections.

        Args:
            detections: List of detection dicts with 'bbox': [x1, y1, x2, y2].

        Returns:
            List of active tracks (dict with track_id, bbox, hits, age, etc.).
        """
        self.frame_count += 1

        # Predict next positions for all existing tracks
        for track in self.tracks:
            track["predicted_bbox"] = self._predict_next_position(track)
            track["age"] += 1

        if len(detections) == 0:
            # Age all tracks, remove stale
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return self._get_active_tracks()

        det_bboxes = [d["bbox"] for d in detections]
        track_bboxes = [t["predicted_bbox"] for t in self.tracks]

        if len(self.tracks) == 0:
            # All detections are new
            for det in detections:
                self.tracks.append(self._create_track(det))
        else:
            # Build IoU cost matrix: rows = tracks, cols = detections
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for ti, t_bbox in enumerate(track_bboxes):
                for di, d_bbox in enumerate(det_bboxes):
                    iou_matrix[ti, di] = self._compute_iou(t_bbox, d_bbox)

            # Hungarian assignment (minimize cost = maximize IoU → negate)
            from scipy.optimize import linear_sum_assignment  # type: ignore
            cost_matrix = 1.0 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_track_ids: set = set()
            matched_det_ids: set = set()

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.iou_threshold:
                    # Valid match
                    track = self.tracks[row]
                    det = detections[col]
                    track["bbox"] = det["bbox"]
                    track["predicted_bbox"] = det["bbox"]
                    track["hits"] += 1
                    track["age"] = 0
                    track["consecutive_misses"] = 0
                    track["class_name"] = det.get("class_name", track["class_name"])
                    track["confidence"] = det.get("confidence", 0.0)
                    track["center_x"] = det.get("center_x", (det["bbox"][0] + det["bbox"][2]) / 2)
                    track["center_y"] = det.get("center_y", (det["bbox"][1] + det["bbox"][3]) / 2)
                    track["velocity_x"] = track["center_x"] - track.get("prev_center_x", track["center_x"])
                    track["velocity_y"] = track["center_y"] - track.get("prev_center_y", track["center_y"])
                    track["prev_center_x"] = track["center_x"]
                    track["prev_center_y"] = track["center_y"]
                    matched_track_ids.add(row)
                    matched_det_ids.add(col)

            # Create new tracks for unmatched detections
            for di, det in enumerate(detections):
                if di not in matched_det_ids:
                    self.tracks.append(self._create_track(det))

            # Increment miss counter for unmatched tracks
            for ti, track in enumerate(self.tracks):
                if ti not in matched_track_ids and "hits" in track:
                    track["consecutive_misses"] = track.get("consecutive_misses", 0) + 1

        # Remove tracks that have been missing too long
        self.tracks = [t for t in self.tracks if t.get("consecutive_misses", 0) <= self.max_age]

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Track]:
        """Return tracks that have enough hits to be reported."""
        active = []
        for track in self.tracks:
            if track["hits"] >= self.min_hits or self.frame_count <= self.min_hits:
                active.append({
                    "track_id": track["track_id"],
                    "bbox": track["bbox"],
                    "center_x": track.get("center_x", 0.0),
                    "center_y": track.get("center_y", 0.0),
                    "class_name": track.get("class_name", "unknown"),
                    "confidence": track.get("confidence", 0.0),
                    "hits": track["hits"],
                    "age": track["age"],
                    "velocity_x": track.get("velocity_x", 0.0),
                    "velocity_y": track.get("velocity_y", 0.0),
                })
        return active

    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Computes Intersection over Union between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU value in [0, 1].
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
        area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0

    def _create_track(self, detection: Detection) -> Track:
        """
        Creates a new track from a detection.

        Args:
            detection: Detection dict with bbox, class_name, confidence, etc.

        Returns:
            New track dict.
        """
        bbox = detection.get("bbox", [0.0, 0.0, 0.0, 0.0])
        cx = detection.get("center_x", (bbox[0] + bbox[2]) / 2)
        cy = detection.get("center_y", (bbox[1] + bbox[3]) / 2)

        track: Track = {
            "track_id": self._next_id,
            "bbox": bbox,
            "predicted_bbox": bbox,
            "class_name": detection.get("class_name", "unknown"),
            "confidence": detection.get("confidence", 0.0),
            "hits": 1,
            "age": 0,
            "consecutive_misses": 0,
            "center_x": cx,
            "center_y": cy,
            "prev_center_x": cx,
            "prev_center_y": cy,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_frame": self.frame_count,
        }
        self._next_id += 1
        return track

    def _predict_next_position(self, track: Track) -> List[float]:
        """
        Predicts next bounding box position using constant velocity model.

        Args:
            track: Current track dict with velocity information.

        Returns:
            Predicted [x1, y1, x2, y2] for the next frame.
        """
        bbox = track["bbox"]
        vx = track.get("velocity_x", 0.0)
        vy = track.get("velocity_y", 0.0)

        # Dampen velocity (objects tend to slow)
        damping = 0.8
        vx *= damping
        vy *= damping

        return [
            bbox[0] + vx,
            bbox[1] + vy,
            bbox[2] + vx,
            bbox[3] + vy,
        ]

    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracks = []
        self.frame_count = 0
        self._next_id = 1

    def get_track_count(self) -> int:
        """Returns number of currently active tracks."""
        return len(self.tracks)

"""
Workplace Ergonomics & Injury Prevention Platform
Pose Extractor Module - MediaPipe and YOLOv8-pose keypoint extraction
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe 33 landmark names (in order)
# ---------------------------------------------------------------------------
MEDIAPIPE_LANDMARK_NAMES: List[str] = [
    "nose",                   # 0
    "left_eye_inner",         # 1
    "left_eye",               # 2
    "left_eye_outer",         # 3
    "right_eye_inner",        # 4
    "right_eye",              # 5
    "right_eye_outer",        # 6
    "left_ear",               # 7
    "right_ear",              # 8
    "mouth_left",             # 9
    "mouth_right",            # 10
    "left_shoulder",          # 11
    "right_shoulder",         # 12
    "left_elbow",             # 13
    "right_elbow",            # 14
    "left_wrist",             # 15
    "right_wrist",            # 16
    "left_pinky",             # 17
    "right_pinky",            # 18
    "left_index",             # 19
    "right_index",            # 20
    "left_thumb",             # 21
    "right_thumb",            # 22
    "left_hip",               # 23
    "right_hip",              # 24
    "left_knee",              # 25
    "right_knee",             # 26
    "left_ankle",             # 27
    "right_ankle",            # 28
    "left_heel",              # 29
    "right_heel",             # 30
    "left_foot_index",        # 31
    "right_foot_index",       # 32
]

# COCO 17 keypoint names (YOLOv8-pose)
COCO_KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

Keypoints = Dict[str, Dict[str, float]]  # name → {x, y, z, visibility}


class PoseExtractor:
    """
    Extracts human body keypoints from images and video using MediaPipe or YOLOv8-pose.

    Attributes:
        model_type: 'mediapipe' or 'yolov8'
        confidence: Minimum detection/tracking confidence threshold
    """

    def __init__(self, model_type: str = "mediapipe", confidence: float = 0.5) -> None:
        self.model_type = model_type.lower()
        self.confidence = confidence
        self._mp_pose: Optional[Any] = None
        self._mp_pose_solution: Optional[Any] = None
        self._yolo_model: Optional[Any] = None
        self._initialized = False
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the underlying pose estimation model."""
        if self.model_type == "mediapipe":
            self._init_mediapipe()
        elif self.model_type == "yolov8":
            self._init_yolov8()
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Use 'mediapipe' or 'yolov8'.")

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
            # mediapipe >= 0.10 uses tasks API; fall back to legacy solutions if available
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self._mp_pose_solution = mp.solutions.pose
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=self.confidence,
                    min_tracking_confidence=self.confidence,
                )
                self._mp_backend = "solutions"
            else:
                # New tasks-based API — fall back to YOLOv8 pose instead
                logger.warning("mediapipe.solutions not available (new API); falling back to YOLOv8 pose")
                self.model_type = "yolov8"
                self._init_yolov8()
                return
            self._initialized = True
            logger.info("MediaPipe Pose initialized (confidence=%.2f)", self.confidence)
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
            self._initialized = False

    def _init_yolov8(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._yolo_model = YOLO("yolov8n-pose.pt")
            self._initialized = True
            logger.info("YOLOv8-pose initialized.")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            self._initialized = False

    def extract_keypoints(self, image: np.ndarray) -> Keypoints:
        """
        Extracts pose keypoints from a single image array.

        Args:
            image: BGR or RGB numpy array (HxWxC).

        Returns:
            Dict of keypoint_name → {x, y, z, visibility}.
            Coordinates are normalized 0-1 relative to image dimensions.
            Returns empty dict if no pose detected.
        """
        if not self._initialized:
            logger.warning("Model not initialized; returning empty keypoints.")
            return {}

        if self.model_type == "mediapipe":
            return self._extract_mediapipe(image)
        else:
            return self._extract_yolov8(image)

    def _extract_mediapipe(self, image: np.ndarray) -> Keypoints:
        """Extract keypoints using MediaPipe Pose."""
        import cv2  # type: ignore

        # MediaPipe expects RGB
        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        results = self._mp_pose.process(rgb)

        if results.pose_landmarks is None:
            logger.debug("No pose detected in image.")
            return {}

        keypoints: Keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx < len(MEDIAPIPE_LANDMARK_NAMES):
                name = MEDIAPIPE_LANDMARK_NAMES[idx]
                keypoints[name] = {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": float(landmark.visibility),
                }
        return keypoints

    def _extract_yolov8(self, image: np.ndarray) -> Keypoints:
        """Extract keypoints using YOLOv8-pose."""
        if self._yolo_model is None:
            return {}

        results = self._yolo_model.predict(image, conf=self.confidence, verbose=False)
        H, W = image.shape[:2]
        keypoints: Keypoints = {}

        for result in results:
            if result.keypoints is None:
                continue
            kp_data = result.keypoints.xy
            kp_conf = result.keypoints.conf

            if len(kp_data) == 0:
                continue

            # Take the first (most confident) detected person
            kps = kp_data[0].cpu().numpy()
            confs = kp_conf[0].cpu().numpy() if kp_conf is not None else np.ones(len(kps))

            for idx, (xy, c) in enumerate(zip(kps, confs)):
                if idx < len(COCO_KEYPOINT_NAMES):
                    name = COCO_KEYPOINT_NAMES[idx]
                    keypoints[name] = {
                        "x": float(xy[0]) / max(W, 1),
                        "y": float(xy[1]) / max(H, 1),
                        "z": 0.0,
                        "visibility": float(c),
                    }
            break  # Only first person

        return keypoints

    def extract_from_image(self, image_path: str) -> Keypoints:
        """
        Extracts keypoints from a single image file.

        Args:
            image_path: Path to image file.

        Returns:
            Keypoints dict.
        """
        try:
            import cv2  # type: ignore
            image = cv2.imread(image_path)
            if image is None:
                raise OSError(f"Cannot read image: {image_path}")
        except ImportError:
            from PIL import Image  # type: ignore
            pil = Image.open(image_path).convert("RGB")
            image = np.array(pil)[:, :, ::-1]

        return self.extract_keypoints(image)

    def extract_from_video(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts keypoints from a video file.

        Args:
            video_path: Path to video file.
            sample_rate: Process every nth frame (1 = every frame).
            max_frames: Maximum number of frames to process.

        Returns:
            List of dicts: {frame_idx, timestamp_sec, keypoints}.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError("opencv-python required for video processing") from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results: List[Dict[str, Any]] = []
        frame_idx = 0
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and processed >= max_frames:
                    break

                if frame_idx % sample_rate == 0:
                    kps = self.extract_keypoints(frame)
                    results.append({
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                        "keypoints": kps,
                    })
                    processed += 1

                    if processed % 100 == 0:
                        logger.info("Processed %d frames from %s", processed, video_path)

                frame_idx += 1
        finally:
            cap.release()

        logger.info("Video extraction complete: %d frames processed from %s", processed, video_path)
        return results

    def batch_extract(self, image_dir: str) -> Dict[str, Keypoints]:
        """
        Extracts keypoints from all images in a directory.

        Args:
            image_dir: Directory containing image files.

        Returns:
            Dict mapping filename → keypoints dict.
        """
        img_dir = Path(image_dir)
        if not img_dir.exists():
            raise OSError(f"Directory not found: {image_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
        logger.info("Batch extracting keypoints from %d images in %s", len(image_paths), image_dir)

        results: Dict[str, Keypoints] = {}
        for i, img_path in enumerate(sorted(image_paths)):
            try:
                kps = self.extract_from_image(str(img_path))
                results[img_path.name] = kps
            except Exception as exc:
                logger.warning("Failed to extract keypoints from %s: %s", img_path.name, exc)
                results[img_path.name] = {}

            if (i + 1) % 50 == 0:
                logger.info("Batch extraction: %d / %d done", i + 1, len(image_paths))

        return results

    def get_landmark_names(self) -> List[str]:
        """
        Returns the list of landmark names for the active model.

        Returns:
            List of 33 MediaPipe landmark names, or 17 COCO keypoint names for YOLOv8.
        """
        if self.model_type == "mediapipe":
            return list(MEDIAPIPE_LANDMARK_NAMES)
        else:
            return list(COCO_KEYPOINT_NAMES)

    def close(self) -> None:
        """Release resources."""
        if self._mp_pose is not None:
            try:
                self._mp_pose.close()
            except Exception:
                pass
            self._mp_pose = None

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

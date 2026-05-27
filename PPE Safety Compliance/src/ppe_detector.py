"""
PPE Detector — YOLOv8-based Personal Protective Equipment Detection.
Supports image, batch, and video inference with optional training.
"""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("ultralytics found — using YOLOv8 detection.")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed — using OpenCV HOG-based fallback detector.")


class Detection:
    """Single object detection result."""

    __slots__ = ("class_name", "confidence", "bbox", "center_x", "center_y", "class_id")

    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: tuple[int, int, int, int],  # x1, y1, x2, y2
        class_id: int = 0,
    ) -> None:
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.center_x = (bbox[0] + bbox[2]) // 2
        self.center_y = (bbox[1] + bbox[3]) // 2
        self.class_id = class_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "class_id": self.class_id,
        }

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox})"
        )


class _FallbackDetector:
    """
    Simplified detection using OpenCV's background subtraction + contour analysis.
    Used when ultralytics is not installed. Detects 'Person' class only.
    """

    def __init__(self, conf_threshold: float = 0.4) -> None:
        self.conf_threshold = conf_threshold
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        # Person detector via HOG
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        h, w = image.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        small = cv2.resize(image, (int(w * scale), int(h * scale)))

        try:
            rects, weights = self.hog.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            for (rx, ry, rw, rh), weight in zip(rects, weights):
                if float(weight[0]) >= self.conf_threshold:
                    x1, y1 = int(rx / scale), int(ry / scale)
                    x2, y2 = int((rx + rw) / scale), int((ry + rh) / scale)
                    detections.append(Detection(
                        class_name="Person",
                        confidence=float(np.clip(weight[0] / 2.0, 0, 1)),
                        bbox=(x1, y1, x2, y2),
                        class_id=4,
                    ))
        except Exception as exc:
            logger.debug(f"HOG detection error: {exc}")

        return detections


class PPEDetector:
    """
    YOLOv8-based PPE (Personal Protective Equipment) detector.
    Detects: Hardhat, Safety_Vest, Safety_Glasses, Gloves, Person, No_Hardhat, No_Safety_Vest.
    """

    DEFAULT_CLASS_NAMES = [
        "Hardhat", "Safety_Vest", "Safety_Glasses", "Gloves",
        "Person", "No_Hardhat", "No_Safety_Vest",
    ]

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.model: Any = None
        self._fallback = _FallbackDetector(conf_threshold=conf_threshold)

        if weights_path is not None:
            self.load_model(weights_path)
        elif ULTRALYTICS_AVAILABLE:
            # Load pretrained YOLOv8n as base model (transfer learning target)
            try:
                logger.info("Loading YOLOv8n pretrained model...")
                self.model = YOLO("yolov8n.pt")
                logger.info("YOLOv8n loaded (pretrained COCO — fine-tune on PPE data for production)")
            except Exception as exc:
                logger.warning(f"YOLOv8n load failed: {exc} — using fallback detector")
                self.model = None

    def load_model(self, weights_path: str | Path) -> None:
        """Load a trained YOLOv8 model from weights file."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics not available — cannot load YOLO weights.")
            return

        logger.info(f"Loading YOLO model from {weights_path}...")
        self.model = YOLO(str(weights_path))
        # Update class names from model if available
        if hasattr(self.model, "names") and self.model.names:
            self.class_names = list(self.model.names.values())
        logger.info(f"Model loaded. Classes: {self.class_names}")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run detection on a single image (numpy BGR array).
        Returns list of Detection objects.
        """
        if image is None or image.size == 0:
            return []

        if self.model is None or not ULTRALYTICS_AVAILABLE:
            return self._fallback.detect(image)

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=False,
            )
            detections = self._parse_results(results[0])
            elapsed = time.perf_counter() - t0
            logger.debug(f"Detection: {len(detections)} objects in {elapsed*1000:.1f}ms")
            return detections
        except Exception as exc:
            logger.warning(f"YOLO inference error: {exc} — using fallback")
            return self._fallback.detect(image)

    def _parse_results(self, result: Any) -> list[Detection]:
        """Parse ultralytics result object into Detection list."""
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Get model class names
        model_names = result.names if hasattr(result, "names") else {}

        for box, conf, cid in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = model_names.get(cid, self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}")

            # Map COCO person class to our Person class
            if class_name.lower() == "person":
                class_name = "Person"

            detections.append(Detection(
                class_name=class_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                class_id=int(cid),
            ))

        return detections

    def detect_batch(self, image_list: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of images."""
        if not image_list:
            return []

        if self.model is None or not ULTRALYTICS_AVAILABLE:
            return [self._fallback.detect(img) for img in image_list]

        try:
            results = self.model.predict(
                source=image_list,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=False,
            )
            return [self._parse_results(r) for r in results]
        except Exception as exc:
            logger.warning(f"Batch detection error: {exc}")
            return [self._fallback.detect(img) for img in image_list]

    def detect_from_video(
        self,
        video_path: str | Path,
        sample_every_n_frames: int = 5,
        max_frames: int = 500,
        output_path: Optional[str | Path] = None,
    ) -> list[dict[str, Any]]:
        """
        Analyze a video file, sampling every n frames.

        Returns list of frame analysis results:
        [{frame_idx, timestamp_sec, detections, n_persons, n_violations}]
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {total_frames} frames at {fps:.1f} FPS, sampling every {sample_every_n_frames} frames")

        frame_results: list[dict[str, Any]] = []
        frame_idx = 0
        processed = 0

        writer = None
        if output_path:
            output_path = Path(output_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                detections = self.detect(frame)
                n_persons = sum(1 for d in detections if d.class_name == "Person")
                n_violations = sum(1 for d in detections if d.class_name.startswith("No_"))

                frame_results.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "detections": [d.to_dict() for d in detections],
                    "n_persons": n_persons,
                    "n_violations": n_violations,
                    "n_ppe_items": len(detections) - n_persons,
                })
                processed += 1

                if writer is not None:
                    from src.visualization import draw_ppe_detections
                    from src.compliance_engine import PPEComplianceChecker
                    checker = PPEComplianceChecker()
                    compliance = checker.check_worker_compliance(detections)
                    annotated = draw_ppe_detections(frame, detections, compliance)
                    writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        logger.info(f"Video analysis complete: {processed} frames processed")
        return frame_results

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 50,
        imgsz: int = 640,
        batch_size: int = 16,
        project_name: str = "ppe_detection",
        experiment_name: str = "train",
    ) -> Any:
        """
        Train YOLOv8 model on PPE detection dataset.
        Saves best weights to runs/detect/{experiment_name}/weights/best.pt

        Returns training results object.
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for training. pip install ultralytics")

        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        if self.model is None:
            logger.info("Loading YOLOv8n for training...")
            self.model = YOLO("yolov8n.pt")

        logger.info(
            f"Starting YOLOv8 training: epochs={epochs}, imgsz={imgsz}, "
            f"batch={batch_size}, data={data_yaml}"
        )

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project_name,
            name=experiment_name,
            device=self.device,
            workers=4,
            patience=15,
            save=True,
            exist_ok=True,
        )

        logger.info(f"Training complete. Results saved to: {project_name}/{experiment_name}")
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        if self.model is None:
            return {"status": "not_loaded", "backend": "fallback_hog"}

        info: dict[str, Any] = {
            "status": "loaded",
            "backend": "ultralytics_yolov8" if ULTRALYTICS_AVAILABLE else "fallback",
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "class_names": self.class_names,
            "n_classes": len(self.class_names),
        }

        if ULTRALYTICS_AVAILABLE and hasattr(self.model, "info"):
            try:
                model_info = self.model.info(verbose=False)
                info["parameters"] = model_info[0] if model_info else "N/A"
            except Exception:
                pass

        return info

"""
Retail Operations Intelligence & Loss Prevention Platform
Detector Module - YOLOv8 model loading, training, and inference
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default model weights for each size variant
YOLOV8_WEIGHTS: Dict[str, str] = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


def load_yolov8_model(
    weights_path: Optional[str] = None,
    model_size: str = "n",
) -> Any:
    """
    Loads a YOLOv8 model. Downloads pretrained COCO weights if not found locally.

    Args:
        weights_path: Path to custom .pt weights file. If None, uses pretrained.
        model_size: One of 'n', 's', 'm', 'l', 'x' (nano is default for speed).

    Returns:
        Loaded YOLO model object.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError("ultralytics is required: pip install ultralytics") from exc

    if weights_path is not None:
        wpath = Path(weights_path)
        if wpath.exists():
            logger.info("Loading custom weights from %s", weights_path)
            model = YOLO(str(wpath))
            return model
        else:
            logger.warning("Weights file not found at %s; falling back to pretrained.", weights_path)

    pretrained_name = YOLOV8_WEIGHTS.get(model_size, "yolov8n.pt")
    logger.info("Loading pretrained YOLOv8 weights: %s", pretrained_name)
    model = YOLO(pretrained_name)
    return model


def train_retail_model(
    data_yaml_path: str,
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
    project: str = "runs/detect",
    name: str = "retail_model",
) -> Dict[str, Any]:
    """
    Trains a YOLOv8 model on the retail dataset with augmentation enabled.

    Args:
        data_yaml_path: Path to dataset YAML file (Roboflow / demo format).
        epochs: Number of training epochs.
        imgsz: Input image size.
        device: 'cpu', 'cuda', or device index.
        project: Output project directory.
        name: Experiment name.

    Returns:
        Dict with best_model_path, metrics, training_plots_dir.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError("ultralytics is required: pip install ultralytics") from exc

    if not Path(data_yaml_path).exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml_path}")

    logger.info("Starting YOLOv8 training: epochs=%d, imgsz=%d, device=%s", epochs, imgsz, device)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        batch=16,
        workers=0,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Save settings
        save=True,
        save_period=10,
        plots=True,
        exist_ok=True,
    )

    run_dir = Path(project) / name
    best_weights = run_dir / "weights" / "best.pt"
    plots_dir = run_dir

    output: Dict[str, Any] = {
        "best_model_path": str(best_weights) if best_weights.exists() else None,
        "last_model_path": str(run_dir / "weights" / "last.pt"),
        "training_plots_dir": str(plots_dir),
        "metrics": {},
    }

    # Extract metrics if available
    try:
        metrics_dict = results.results_dict if hasattr(results, "results_dict") else {}
        output["metrics"] = {
            "mAP50": metrics_dict.get("metrics/mAP50(B)", 0.0),
            "mAP50_95": metrics_dict.get("metrics/mAP50-95(B)", 0.0),
            "precision": metrics_dict.get("metrics/precision(B)", 0.0),
            "recall": metrics_dict.get("metrics/recall(B)", 0.0),
        }
    except Exception as exc:
        logger.warning("Could not extract training metrics: %s", exc)

    logger.info("Training complete. Best model: %s", output["best_model_path"])
    return output


def run_inference(
    model: Any,
    image_path_or_array: Union[str, np.ndarray],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Runs YOLOv8 inference on a single image.

    Args:
        model: Loaded YOLO model.
        image_path_or_array: Image file path or numpy array (BGR).
        conf_threshold: Minimum confidence to include detection.
        iou_threshold: NMS IoU threshold.

    Returns:
        List of detection dicts with keys:
            class_name, class_id, confidence, bbox [x1,y1,x2,y2],
            center_x, center_y, width, height
    """
    start = time.perf_counter()

    results = model.predict(
        source=image_path_or_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []
    class_names: List[str] = model.names if hasattr(model, "names") else {}

    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes
        for i in range(len(boxes)):
            try:
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = class_names[cls_id] if isinstance(class_names, (list, dict)) and cls_id in (
                    range(len(class_names)) if isinstance(class_names, list) else class_names
                ) else str(cls_id)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                detections.append({
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                })
            except Exception as exc:
                logger.debug("Error parsing detection %d: %s", i, exc)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("Inference completed in %.1f ms, %d detections", elapsed_ms, len(detections))
    return detections


def run_video_inference(
    model: Any,
    video_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Processes a video file frame by frame.

    Args:
        model: Loaded YOLO model.
        video_path: Path to input video file.
        output_path: If provided, saves annotated video to this path.
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        max_frames: Optional limit on number of frames to process.

    Returns:
        List of per-frame detection dicts:
            {frame_idx, timestamp_sec, detections: List[...]}
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("opencv-python is required: pip install opencv-python") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[Any] = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_frame_detections: List[Dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            frame_dets = run_inference(model, frame, conf_threshold, iou_threshold)
            all_frame_detections.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "detections": frame_dets,
            })

            if writer is not None:
                annotated = _draw_boxes_cv2(frame, frame_dets)
                writer.write(annotated)

            if frame_idx % 100 == 0:
                logger.info("Processed frame %d (%.1f sec)", frame_idx, frame_idx / fps)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    logger.info("Video inference complete: %d frames processed", frame_idx)
    return all_frame_detections


def batch_inference(
    model: Any,
    image_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs inference on all images in a directory.

    Args:
        model: Loaded YOLO model.
        image_dir: Directory containing image files.
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.

    Returns:
        Dict mapping image filename → list of detections.
    """
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise OSError(f"Image directory not found: {image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
    logger.info("Running batch inference on %d images in %s", len(image_paths), image_dir)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for i, img_path in enumerate(sorted(image_paths)):
        try:
            dets = run_inference(model, str(img_path), conf_threshold, iou_threshold)
            results[img_path.name] = dets
        except Exception as exc:
            logger.warning("Inference failed for %s: %s", img_path.name, exc)
            results[img_path.name] = []

        if (i + 1) % 50 == 0:
            logger.info("Batch inference: %d / %d images done", i + 1, len(image_paths))

    logger.info("Batch inference complete: %d images processed", len(image_paths))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_boxes_cv2(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes on a frame using OpenCV (no external deps)."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return frame

    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out

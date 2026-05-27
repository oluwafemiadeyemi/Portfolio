from ultralytics import YOLO
import json, pathlib

base = pathlib.Path(r"c:\Users\Artstanding\Documents\MIT Applied AI and Data Science\Data Science Projects\Face Detection")
data_yaml = base / "data" / "raw" / "ppe_yolo" / "data.yaml"
model_dir = base / "data" / "models"

model = YOLO("yolov8n.pt")
results = model.train(
    data=str(data_yaml),
    epochs=5,
    imgsz=416,
    device="cpu",
    project=str(model_dir),
    name="ppe_safety",
    exist_ok=True,
    verbose=False,
    workers=2,
    batch=16,
)
metrics = {
    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
    "mAP50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
    "precision": results.results_dict.get("metrics/precision(B)", 0),
    "recall": results.results_dict.get("metrics/recall(B)", 0),
}
(model_dir / "metrics.json").write_text(json.dumps(metrics))
print("PPE Safety YOLO done:", metrics)

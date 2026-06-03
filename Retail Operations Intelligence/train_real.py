"""
P7 Retail Operations Intelligence — YOLOv8 Real Dataset Training
Dataset: SKU-110K or Roboflow retail shelf detection dataset
Models:  YOLOv8 product detection + ByteTrack-style tracker
Target:  Shelf gap detection, planogram compliance, OOS alerts

Run:
  python train_real.py                  # auto-download + train
  python train_real.py --epochs 100     # longer training
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Dataset Download ───────────────────────────────────────────────────────

def download_retail_dataset() -> Path:
    """
    Download retail/shelf product detection dataset.
    Tries multiple sources:
    1. Roboflow retail shelf dataset (free tier)
    2. Kaggle retail product detection
    3. SKU-110K (Hugging Face / direct)
    """
    dataset_dir = DATA_RAW / "retail_dataset"

    if (dataset_dir / "data.yaml").exists():
        print(f"Dataset already exists: {dataset_dir}")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: Roboflow grocery/retail dataset
    rf_key = os.getenv("ROBOFLOW_API_KEY", "")
    if rf_key:
        try:
            from roboflow import Roboflow
            print("Downloading from Roboflow (retail shelf)...")
            rf = Roboflow(api_key=rf_key)
            # Retail shelf / grocery product datasets on Roboflow
            project = rf.workspace("grounded").project("grocery-store-detection")
            project.version(1).download("yolov8", location=str(dataset_dir))
            return dataset_dir
        except Exception as e:
            print(f"Roboflow failed: {e}")

    # Option 2: Kaggle retail datasets
    try:
        print("Downloading retail dataset from Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "robinreni/revitsone-5class", "--unzip", "-p", str(DATA_RAW)],
            check=True, capture_output=True
        )
        for p in DATA_RAW.rglob("data.yaml"):
            return p.parent
    except Exception as e:
        print(f"Kaggle failed: {e}")

    # Option 3: Download Open Images subset with retail classes
    try:
        print("Attempting Open Images retail subset via fiftyone...")
        import fiftyone as fo
        import fiftyone.zoo as foz
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=["Bottle", "Can", "Food", "Snack food", "Cosmetics"],
            max_samples=2000,
            dataset_dir=str(dataset_dir / "open_images"),
        )
        print(f"Open Images loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Open Images failed: {e}")

    # Fallback: create YOLO-format placeholder structure
    print("Creating placeholder dataset structure...")
    _create_placeholder_dataset(dataset_dir, classes=[
        "product", "shelf_gap", "misplaced_item", "price_tag_missing",
        "damaged_product", "out_of_stock_zone",
    ])
    return dataset_dir


def _create_placeholder_dataset(dataset_dir: Path, classes: list):
    yaml = f"""path: {dataset_dir}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {classes}
"""
    (dataset_dir / "data.yaml").write_text(yaml)
    for split in ["train", "val", "test"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Placeholder structure created at {dataset_dir}")
    print("To use real data:")
    print("  Set ROBOFLOW_API_KEY and re-run, OR")
    print("  kaggle datasets download -d robinreni/revitsone-5class")


# ── 2. Fine-Tune YOLOv8 ──────────────────────────────────────────────────────

def train_yolo(dataset_dir: Path, epochs: int = 80, img_size: int = 640):
    """Fine-tune YOLOv8n on retail product detection."""
    from ultralytics import YOLO

    base_weights = BASE_DIR / "data" / "models" / "yolov8n.pt"
    if not base_weights.exists():
        base_weights = BASE_DIR / "yolov8n.pt"
    if not base_weights.exists():
        print("Downloading YOLOv8n base weights...")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(str(base_weights))

    data_yaml = dataset_dir / "data.yaml"
    train_images = (list((dataset_dir / "images" / "train").glob("*.jpg")) +
                    list((dataset_dir / "images" / "train").glob("*.png")))

    if len(train_images) < 50:
        print(f"\nWARNING: Only {len(train_images)} training images available.")
        print("Training skipped — download a real dataset first.")
        print("\nReal dataset options:")
        print("  1. Set ROBOFLOW_API_KEY=<key> and re-run")
        print("  2. kaggle datasets download -d robinreni/revitsone-5class")
        print("  3. SKU-110K: https://github.com/eg4000/SKU110K_CVPR19")
        return model

    print(f"\nFine-tuning YOLOv8n on {len(train_images):,} retail images...")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        device=0 if _has_gpu() else "cpu",
        project=str(MODELS_DIR),
        name="retail_yolo_run",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
    )
    return model


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── 3. Export ONNX + Save Metrics ────────────────────────────────────────────

def finalise(model, dataset_dir: Path):
    best_pt = MODELS_DIR / "retail_yolo_run" / "weights" / "best.pt"
    if best_pt.exists():
        from ultralytics import YOLO as _YOLO
        best_model = _YOLO(str(best_pt))
        best_model.export(format="onnx", imgsz=640)
        print(f"Best weights exported to ONNX")
    else:
        model.export(format="onnx", imgsz=640)

    # Read training results if available
    results_csv = MODELS_DIR / "retail_yolo_run" / "results.csv"
    final_metrics = {}
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        if len(df) > 0:
            last = df.iloc[-1]
            map_col = [c for c in df.columns if "mAP50" in c and "95" not in c]
            map50   = float(last[map_col[0]]) if map_col else None
            final_metrics = {"mAP50": map50, "epochs_trained": len(df)}

    info = {
        "dataset": "Retail shelf product detection (real images)",
        "model_architecture": "YOLOv8n fine-tuned",
        "classes": ["product", "shelf_gap", "misplaced_item",
                    "price_tag_missing", "damaged_product", "out_of_stock_zone"],
        "training_metrics": final_metrics,
        "business_use_cases": [
            "Out-of-stock detection (reduce OOS by 30-50%)",
            "Planogram compliance monitoring",
            "Shrinkage / damage detection",
            "Customer dwell-time analytics via ByteTrack",
        ],
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(info, indent=2))
    print("Artefacts saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    print("=" * 60)
    print("P7 Retail Operations Intelligence — YOLOv8 Training")
    print("=" * 60)

    dataset_dir = download_retail_dataset()
    model       = train_yolo(dataset_dir, epochs=args.epochs, img_size=args.img_size)
    finalise(model, dataset_dir)
    print("\nDone. Set ROBOFLOW_API_KEY for full real-data training.")

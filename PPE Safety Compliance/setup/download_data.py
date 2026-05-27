"""
Data Download Script — PPE Compliance Monitoring System
Downloads PPE detection datasets from Kaggle and Roboflow.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def run_kaggle_command(cmd: list[str], description: str) -> bool:
    """Run a kaggle CLI command."""
    logger.info(f"Downloading: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"FAILED: {description}\n{exc.stderr}")
        return False
    except FileNotFoundError:
        logger.error("kaggle CLI not found. Install: pip install kaggle")
        return False


def check_kaggle_setup() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
            "Setup: go to https://www.kaggle.com/account → Create New API Token"
        )
        return False
    return True


def download_hard_hat_detection(data_dir: Path) -> bool:
    """
    Dataset 1: Hard Hat Detection (PASCAL VOC format)
    URL: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
    ~5,000 annotated images with hard_hat and no_hard_hat classes
    """
    return run_kaggle_command(
        ["kaggle", "datasets", "download",
         "-d", "andrewmvd/hard-hat-detection",
         "-p", str(data_dir), "--unzip"],
        "Hard Hat Detection Dataset (andrewmvd)",
    )


def download_construction_safety_roboflow(data_dir: Path) -> bool:
    """
    Dataset 2: Construction Site Safety (Roboflow via Kaggle)
    URL: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
    ~2,400 images with 10 PPE classes in YOLO format
    Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
             Safety Cone, Safety Vest, machinery, vehicle
    """
    return run_kaggle_command(
        ["kaggle", "datasets", "download",
         "-d", "snehilsanyal/construction-site-safety-image-dataset-roboflow",
         "-p", str(data_dir), "--unzip"],
        "Construction Site Safety Dataset (Roboflow via Kaggle)",
    )


def download_via_roboflow_api() -> None:
    """
    Download directly via Roboflow API (requires roboflow package + API key).
    Dataset: https://universe.roboflow.com/joseph-nelson/hard-hat-workers
    """
    print("\n" + "=" * 70)
    print("ROBOFLOW API DOWNLOAD")
    print("=" * 70)
    print()
    print("1. Install: pip install roboflow")
    print("2. Get API key from: https://app.roboflow.com/settings/api")
    print()
    print("Then run:")
    print("  from roboflow import Roboflow")
    print("  rf = Roboflow(api_key='YOUR_API_KEY')")
    print("  project = rf.workspace().project('hard-hat-workers')")
    print("  dataset = project.version(1).download('yolov8')")
    print()
    print("Alternative — PPE Detection Dataset:")
    print("  project = rf.workspace('roboflow-universe-projects').project('ppe-detection-oqeqb')")
    print("  dataset = project.version(2).download('yolov8')")
    print()
    print("=" * 70)


def create_demo_dataset(data_dir: Path) -> None:
    """
    Create a minimal demo dataset structure with synthetic annotations
    for testing when no real data is available.
    """
    import cv2
    import numpy as np

    logger.info("Creating demo synthetic dataset...")
    train_img_dir = data_dir / "images" / "train"
    val_img_dir = data_dir / "images" / "val"
    train_lbl_dir = data_dir / "labels" / "train"
    val_lbl_dir = data_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    class_names = ["Hardhat", "Safety_Vest", "Safety_Glasses", "Gloves",
                   "Person", "No_Hardhat", "No_Safety_Vest"]

    def _make_synthetic_image_and_label(
        img_dir: Path, lbl_dir: Path, idx: int, n_objects: int = 3
    ) -> None:
        # Create synthetic construction scene
        img = np.ones((480, 640, 3), dtype=np.uint8) * 180
        # Sky
        img[:160] = [200, 220, 240]
        # Ground
        img[380:] = [100, 90, 80]

        labels = []
        for _ in range(n_objects):
            cls_id = int(rng.integers(0, len(class_names)))
            cx = float(rng.uniform(0.1, 0.9))
            cy = float(rng.uniform(0.2, 0.8))
            w = float(rng.uniform(0.05, 0.2))
            h = float(rng.uniform(0.05, 0.3))

            # Draw colored rectangle
            x1 = int((cx - w / 2) * 640)
            y1 = int((cy - h / 2) * 480)
            x2 = int((cx + w / 2) * 640)
            y2 = int((cy + h / 2) * 480)
            color = (int(rng.integers(50, 200)),
                     int(rng.integers(50, 200)),
                     int(rng.integers(50, 200)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.putText(img, class_names[cls_id][:4], (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        img_path = img_dir / f"demo_{idx:04d}.jpg"
        lbl_path = lbl_dir / f"demo_{idx:04d}.txt"
        cv2.imwrite(str(img_path), img)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

    for i in range(50):
        _make_synthetic_image_and_label(train_img_dir, train_lbl_dir, i)
    for i in range(10):
        _make_synthetic_image_and_label(val_img_dir, val_lbl_dir, i)

    # Create data.yaml
    from src.data_loader import create_data_yaml
    yaml_path = create_data_yaml(data_dir, class_names)

    logger.info(f"Demo dataset created: 50 train + 10 val synthetic images in {data_dir}")
    print(f"\nDemo dataset created at: {data_dir}")
    print("50 synthetic training images + 10 validation images")
    print(f"data.yaml: {yaml_path}")


def print_manual_instructions() -> None:
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS — PPE Detection Datasets")
    print("=" * 70)
    print()
    print("Option 1: Hard Hat Detection (Kaggle)")
    print("  URL: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection")
    print(f"  Save to: {DATA_DIR}/")
    print()
    print("Option 2: Construction Safety Roboflow (Kaggle)")
    print("  URL: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow")
    print(f"  Save to: {DATA_DIR}/")
    print()
    print("Option 3: Roboflow Universe (Direct)")
    print("  URL: https://universe.roboflow.com/joseph-nelson/hard-hat-workers")
    print("  Download: YOLO v8 format")
    print()
    print("Option 4: Demo synthetic dataset (automatic)")
    print("  Run this script with --demo flag: python download_data.py --demo")
    print()
    print("=" * 70)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")

    use_demo = "--demo" in sys.argv

    if use_demo:
        create_demo_dataset(DATA_DIR)
        return

    if not check_kaggle_setup():
        print_manual_instructions()
        download_via_roboflow_api()
        logger.info("Creating demo dataset as fallback...")
        create_demo_dataset(DATA_DIR)
        return

    success = download_construction_safety_roboflow(DATA_DIR)
    if not success:
        success = download_hard_hat_detection(DATA_DIR)

    if not success:
        print_manual_instructions()
        download_via_roboflow_api()
        logger.info("All downloads failed — creating demo dataset...")
        create_demo_dataset(DATA_DIR)
    else:
        logger.info("Dataset downloaded successfully!")
        # Create data.yaml
        from src.data_loader import create_data_yaml, get_ppe_class_names
        create_data_yaml(DATA_DIR, get_ppe_class_names())

    downloaded = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    logger.info(f"Total images in dataset: {len(downloaded)}")


if __name__ == "__main__":
    main()

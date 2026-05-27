"""
Data Loader for PPE Compliance Monitoring System.
Supports YOLO-format datasets, Roboflow downloads, and dataset validation.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def get_ppe_class_names() -> list[str]:
    """Return canonical PPE detection class names."""
    return [
        "Hardhat",
        "Safety_Vest",
        "Safety_Glasses",
        "Gloves",
        "Person",
        "No_Hardhat",
        "No_Safety_Vest",
    ]


def load_ppe_dataset(data_dir: str | Path) -> dict[str, object]:
    """
    Load a YOLO-format PPE dataset from a directory.
    Expects subdirectories: images/train, images/val, labels/train, labels/val
    (or flat images/ and labels/).

    Returns a dict with:
      - train_images: list of image paths
      - val_images: list of image paths
      - train_labels: list of label paths
      - val_labels: list of label paths
      - class_names: list of class names (from data.yaml if present)
      - n_train: int
      - n_val: int
    """
    data_dir = Path(data_dir)
    logger.info(f"Loading PPE dataset from: {data_dir}")

    result: dict[str, object] = {
        "train_images": [],
        "val_images": [],
        "train_labels": [],
        "val_labels": [],
        "class_names": get_ppe_class_names(),
        "n_train": 0,
        "n_val": 0,
        "data_dir": str(data_dir),
    }

    # Try to load data.yaml for class names
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        if "names" in yaml_data:
            result["class_names"] = yaml_data["names"]
            logger.info(f"Loaded class names from data.yaml: {result['class_names']}")

    # Find images and labels
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ["train", "val", "test"]:
        img_dir_candidates = [
            data_dir / "images" / split,
            data_dir / split / "images",
            data_dir / "images",
            data_dir / split,
        ]
        lbl_dir_candidates = [
            data_dir / "labels" / split,
            data_dir / split / "labels",
            data_dir / "labels",
            data_dir / split,
        ]

        img_dir = next((p for p in img_dir_candidates if p.exists()), None)
        lbl_dir = next((p for p in lbl_dir_candidates if p.exists()), None)

        if img_dir:
            imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions])
            if split in ("train", "test"):
                result["train_images"] = [str(p) for p in imgs]
            else:
                result["val_images"] = [str(p) for p in imgs]
            logger.info(f"  {split}: {len(imgs)} images in {img_dir}")

        if lbl_dir:
            lbls = sorted([p for p in lbl_dir.iterdir() if p.suffix == ".txt"])
            if split in ("train", "test"):
                result["train_labels"] = [str(p) for p in lbls]
            else:
                result["val_labels"] = [str(p) for p in lbls]

    result["n_train"] = len(result["train_images"])
    result["n_val"] = len(result["val_images"])

    if result["n_train"] == 0 and result["n_val"] == 0:
        logger.warning(f"No images found in {data_dir}. Run download_data.py to get data.")
    else:
        logger.info(f"Dataset loaded: {result['n_train']} train, {result['n_val']} val images.")

    return result


def create_data_yaml(
    data_dir: str | Path,
    class_names: Optional[list[str]] = None,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
) -> Path:
    """
    Create a YOLO-compatible data.yaml configuration file.

    Returns the path to the created yaml file.
    """
    data_dir = Path(data_dir)
    if class_names is None:
        class_names = get_ppe_class_names()

    if train_path is None:
        train_path = str(data_dir / "images" / "train")
    if val_path is None:
        val_path = str(data_dir / "images" / "val")

    yaml_content = {
        "path": str(data_dir.resolve()),
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = data_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    logger.info(f"data.yaml created at {yaml_path} with {len(class_names)} classes")
    return yaml_path


def download_roboflow_sample() -> None:
    """
    Print instructions for downloading PPE detection datasets from Roboflow Universe.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════════════════╗
║           PPE Detection Dataset Download Instructions                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  OPTION 1: Roboflow Universe (Recommended)                               ║
║  Dataset: Hard Hat Workers                                               ║
║  URL: https://universe.roboflow.com/joseph-nelson/hard-hat-workers       ║
║                                                                          ║
║  Steps:                                                                  ║
║  1. Create free account at roboflow.com                                  ║
║  2. Go to above URL → Export → YOLO v8 format                           ║
║  3. pip install roboflow                                                 ║
║  4. from roboflow import Roboflow                                        ║
║     rf = Roboflow(api_key="YOUR_API_KEY")                               ║
║     project = rf.workspace().project("hard-hat-workers")                ║
║     dataset = project.version(1).download("yolov8")                     ║
║                                                                          ║
║  OPTION 2: Kaggle Datasets                                               ║
║  kaggle datasets download -d andrewmvd/hard-hat-detection \\             ║
║    -p data/raw/ --unzip                                                  ║
║                                                                          ║
║  OPTION 3: PPE Detection (Roboflow via Kaggle)                          ║
║  kaggle datasets download \\                                             ║
║    -d snehilsanyal/construction-site-safety-image-dataset-roboflow \\   ║
║    -p data/raw/ --unzip                                                  ║
║                                                                          ║
║  After downloading, place files in: data/raw/                            ║
║  Expected structure:                                                     ║
║    data/raw/images/train/*.jpg                                           ║
║    data/raw/images/val/*.jpg                                             ║
║    data/raw/labels/train/*.txt                                           ║
║    data/raw/labels/val/*.txt                                             ║
║    data/raw/data.yaml                                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)
    logger.info("Roboflow download instructions printed.")


def validate_dataset(data_dir: str | Path) -> dict[str, object]:
    """
    Validate a YOLO-format PPE dataset.
    Checks:
    - Image-label pair matching
    - Label format validity (class_id cx cy w h)
    - Class distribution
    - Image readability

    Returns validation report dict.
    """
    data_dir = Path(data_dir)
    dataset = load_ppe_dataset(data_dir)
    class_names = dataset["class_names"]

    report: dict[str, object] = {
        "data_dir": str(data_dir),
        "n_train": dataset["n_train"],
        "n_val": dataset["n_val"],
        "issues": [],
        "class_distribution": {cls: 0 for cls in class_names},
        "missing_labels": 0,
        "invalid_labels": 0,
        "total_annotations": 0,
        "valid": True,
    }

    all_images = list(dataset["train_images"]) + list(dataset["val_images"])

    for img_path_str in all_images:
        img_path = Path(img_path_str)

        # Check label exists
        for label_dir in [
            img_path.parent.parent / "labels" / img_path.parent.name,
            img_path.parent / "labels",
            img_path.parent,
        ]:
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                break
        else:
            report["missing_labels"] = int(report["missing_labels"]) + 1
            report["issues"].append(f"Missing label for: {img_path.name}")
            continue

        # Parse label file
        try:
            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    report["invalid_labels"] = int(report["invalid_labels"]) + 1
                    continue

                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    report["invalid_labels"] = int(report["invalid_labels"]) + 1
                    continue

                if 0 <= class_id < len(class_names):
                    cls = class_names[class_id]
                    report["class_distribution"][cls] = report["class_distribution"].get(cls, 0) + 1
                    report["total_annotations"] = int(report["total_annotations"]) + 1

        except Exception as exc:
            report["issues"].append(f"Label parse error {label_path.name}: {exc}")

    issues_list = report["issues"]
    if len(issues_list) > 0 or report["missing_labels"] > 0:
        report["valid"] = False

    logger.info(
        f"Dataset validation: {report['n_train']} train, {report['n_val']} val, "
        f"{report['total_annotations']} annotations, "
        f"{report['missing_labels']} missing labels, "
        f"valid={report['valid']}"
    )
    return report

"""
Retail Operations Intelligence & Loss Prevention Platform
Data Loader Module - handles Roboflow datasets, sample image downloads, annotations
"""

import os
import logging
import urllib.request
import urllib.error
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 50 known retail store images from Open Images V7 CDN / public sources
SAMPLE_IMAGE_URLS: List[str] = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Two_women_looking_at_goods_in_a_supermarket.jpg/640px-Two_women_looking_at_goods_in_a_supermarket.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Supermarkt.jpg/640px-Supermarkt.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Retail_store_shelves.jpg/640px-Retail_store_shelves.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Grocery_store_shelves.jpg/640px-Grocery_store_shelves.jpg",
    "https://images.unsplash.com/photo-1542838132-92c53300491e?w=640",
    "https://images.unsplash.com/photo-1604719312566-8912e9667d9f?w=640",
    "https://images.unsplash.com/photo-1601599561213-832382fd07ba?w=640",
    "https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=640",
    "https://images.unsplash.com/photo-1578916171728-46686eac8d58?w=640",
    "https://images.unsplash.com/photo-1553546895-531931aa1aa8?w=640",
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640",
    "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=640",
    "https://images.unsplash.com/photo-1574226516831-e1dff420e562?w=640",
    "https://images.unsplash.com/photo-1580913428735-bd3c269d6a82?w=640",
    "https://images.unsplash.com/photo-1619418602850-35ad20aa1700?w=640",
    "https://images.unsplash.com/photo-1530721108087-bc25a18cce42?w=640",
    "https://images.unsplash.com/photo-1416339306562-f3d12fefd36f?w=640",
    "https://images.unsplash.com/photo-1488459716781-31db52582fe9?w=640",
    "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=640",
    "https://images.unsplash.com/photo-1583258292688-d0213dc5a3a8?w=640",
    "https://images.unsplash.com/photo-1621905252507-b35492cc74b4?w=640",
    "https://images.unsplash.com/photo-1567103472667-6898f3a79cf2?w=640",
    "https://images.unsplash.com/photo-1612392166886-ee8475b03af2?w=640",
    "https://images.unsplash.com/photo-1610348725531-843dff563e2c?w=640",
    "https://images.unsplash.com/photo-1598514982901-5a4b29e4b3d5?w=640",
    "https://images.unsplash.com/photo-1559181567-c3190ca9be46?w=640",
    "https://images.unsplash.com/photo-1550583724-b2692b85b150?w=640",
    "https://images.unsplash.com/photo-1617696861117-d6a9ecbeb4e5?w=640",
    "https://images.unsplash.com/photo-1601599561213-832382fd07ba?w=640",
    "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=640",
    "https://images.unsplash.com/photo-1506617420156-8e4536971650?w=640",
    "https://images.unsplash.com/photo-1608198093002-ad4e005484ec?w=640",
    "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=640",
    "https://images.unsplash.com/photo-1601985705806-5b9a291f8304?w=640",
    "https://images.unsplash.com/photo-1585155770447-2f66e2a397b5?w=640",
    "https://images.unsplash.com/photo-1579113800032-c38bd7635818?w=640",
    "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=640",
    "https://images.unsplash.com/photo-1556742111-a301076d9d18?w=640",
    "https://images.unsplash.com/photo-1596363505729-4190a9506133?w=640",
    "https://images.unsplash.com/photo-1587316205860-28f1d7eab11d?w=640",
    "https://images.unsplash.com/photo-1610440042657-612c34d95e9f?w=640",
    "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=640",
    "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=640",
    "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=640",
    "https://images.unsplash.com/photo-1519996529931-28324d5a630e?w=640",
    "https://images.unsplash.com/photo-1594179047519-f347310d3322?w=640",
    "https://images.unsplash.com/photo-1607623814075-e51df1bdc82f?w=640",
    "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?w=640",
    "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=640",
    "https://images.unsplash.com/photo-1627308595229-7830a5c91f9f?w=640",
]

# YOLO class names for retail detection
RETAIL_CLASSES: List[str] = [
    "product", "person", "empty_shelf", "price_tag", "shopping_cart",
    "basket", "shelf", "display", "checkout_counter", "queue_line",
]


def load_roboflow_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Loads an existing Roboflow dataset in YOLO format.
    Searches for 'data/', 'Dataset/', or common Roboflow export structures.

    Args:
        data_dir: Root directory to search for dataset.

    Returns:
        Dictionary with keys: train_images, val_images, test_images, classes, yaml_path
    """
    data_path = Path(data_dir)
    result: Dict[str, Any] = {
        "train_images": [],
        "val_images": [],
        "test_images": [],
        "classes": RETAIL_CLASSES,
        "yaml_path": None,
        "found": False,
    }

    # Candidate subdirectory names used by Roboflow exports
    candidate_roots = [
        data_path,
        data_path / "data",
        data_path / "Dataset",
        data_path / "dataset",
        data_path / "roboflow",
        data_path / "yolo",
    ]

    yaml_path: Optional[Path] = None
    for root in candidate_roots:
        if not root.exists():
            continue
        for yaml_file in root.rglob("data.yaml"):
            yaml_path = yaml_file
            break
        if yaml_path:
            break

    if yaml_path is None:
        logger.warning("No data.yaml found in %s. Dataset not loaded.", data_dir)
        return result

    logger.info("Found Roboflow dataset YAML at: %s", yaml_path)
    result["yaml_path"] = str(yaml_path)
    result["found"] = True

    # Parse YAML manually (avoid pyyaml dependency at import time)
    try:
        import yaml  # type: ignore
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        if "names" in cfg:
            result["classes"] = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else cfg["names"]
    except ImportError:
        logger.warning("PyYAML not installed; using default class names.")

    dataset_root = yaml_path.parent

    for split, key in [("train", "train_images"), ("valid", "val_images"), ("test", "test_images")]:
        for subdir_name in [split, "val" if split == "valid" else split]:
            img_dir = dataset_root / subdir_name / "images"
            if not img_dir.exists():
                img_dir = dataset_root / subdir_name
            if img_dir.exists():
                images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
                result[key] = [str(p) for p in images]
                logger.info("Split '%s': %d images found.", split, len(result[key]))
                break

    return result


def download_sample_images(output_dir: str, n: int = 50) -> List[str]:
    """
    Downloads n sample retail images from hardcoded Open Images / Unsplash CDN URLs.

    Args:
        output_dir: Directory to save downloaded images.
        n: Number of images to download (max 50).

    Returns:
        List of paths to successfully downloaded images.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    urls = SAMPLE_IMAGE_URLS[:min(n, len(SAMPLE_IMAGE_URLS))]
    downloaded: List[str] = []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RetailAI/1.0)"}

    for idx, url in enumerate(urls):
        ext = ".jpg"
        dest = out_path / f"sample_{idx:04d}{ext}"
        if dest.exists():
            downloaded.append(str(dest))
            continue
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(dest, "wb") as f:
                    f.write(response.read())
            downloaded.append(str(dest))
            logger.info("Downloaded %s → %s", url, dest.name)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", url, exc)

    logger.info("Downloaded %d / %d sample images to %s", len(downloaded), len(urls), output_dir)
    return downloaded


def load_annotations(label_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads YOLO format .txt annotation files from a directory.

    YOLO format per line: <class_id> <cx> <cy> <w> <h> (all normalized 0-1)

    Args:
        label_dir: Directory containing .txt annotation files.

    Returns:
        Dict mapping filename stem → list of annotation dicts.
    """
    label_path = Path(label_dir)
    if not label_path.exists():
        logger.warning("Label directory does not exist: %s", label_dir)
        return {}

    annotations: Dict[str, List[Dict[str, Any]]] = {}
    txt_files = list(label_path.glob("*.txt"))
    logger.info("Loading %d annotation files from %s", len(txt_files), label_dir)

    for txt_file in txt_files:
        stem = txt_file.stem
        boxes: List[Dict[str, Any]] = []
        try:
            with open(txt_file, "r") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        logger.debug("Skipping malformed line %d in %s", line_no, txt_file.name)
                        continue
                    class_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append({
                        "class_id": class_id,
                        "cx": cx,
                        "cy": cy,
                        "width": w,
                        "height": h,
                    })
        except OSError as exc:
            logger.error("Cannot read %s: %s", txt_file, exc)
        annotations[stem] = boxes

    return annotations


def create_demo_dataset(output_dir: str) -> str:
    """
    Creates a minimal demo dataset with synthetic images and YOLO bounding box annotations.
    Produces a valid data.yaml so that YOLOv8 training can run without real data.

    Args:
        output_dir: Root directory for the demo dataset.

    Returns:
        Path to the generated data.yaml file.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        logger.warning("Pillow not available; creating placeholder PNG files.")
        Image = None  # type: ignore

    root = Path(output_dir)
    splits = ["train", "valid", "test"]
    n_per_split = {"train": 30, "valid": 10, "test": 10}
    num_classes = len(RETAIL_CLASSES)

    rng = random.Random(42)

    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_per_split[split]):
            img_name = f"{split}_{i:04d}.jpg"
            lbl_name = f"{split}_{i:04d}.txt"
            img_path = img_dir / img_name
            lbl_path = lbl_dir / lbl_name

            # Create synthetic image
            if Image is not None:
                w, h = 640, 640
                img = Image.new("RGB", (w, h), color=(
                    rng.randint(180, 230), rng.randint(180, 230), rng.randint(180, 230)
                ))
                draw = ImageDraw.Draw(img)
                # Draw synthetic shelf lines
                for row in range(4):
                    y = int(h * (0.2 + row * 0.18))
                    draw.line([(0, y), (w, y)], fill=(100, 100, 100), width=3)
                # Draw synthetic product rectangles
                n_boxes = rng.randint(3, 8)
                with open(lbl_path, "w") as lf:
                    for _ in range(n_boxes):
                        cls = rng.randint(0, num_classes - 1)
                        bw = rng.uniform(0.05, 0.15)
                        bh = rng.uniform(0.06, 0.18)
                        cx = rng.uniform(bw / 2, 1 - bw / 2)
                        cy = rng.uniform(bh / 2, 1 - bh / 2)
                        px1, py1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                        px2, py2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                        color = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
                        draw.rectangle([px1, py1, px2, py2], fill=color, outline=(0, 0, 0), width=2)
                        lf.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                img.save(str(img_path), "JPEG")
            else:
                # Fallback: write empty placeholder
                with open(img_path, "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 20)  # minimal JPEG header
                n_boxes = rng.randint(3, 8)
                with open(lbl_path, "w") as lf:
                    for _ in range(n_boxes):
                        cls = rng.randint(0, num_classes - 1)
                        cx = rng.uniform(0.1, 0.9)
                        cy = rng.uniform(0.1, 0.9)
                        bw = rng.uniform(0.05, 0.2)
                        bh = rng.uniform(0.05, 0.2)
                        lf.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Write data.yaml
    yaml_path = root / "data.yaml"
    names_str = "\n".join(f"  {i}: {name}" for i, name in enumerate(RETAIL_CLASSES))
    yaml_content = (
        f"path: {root.resolve()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {num_classes}\n"
        f"names:\n{names_str}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info("Demo dataset created at %s (%d train, %d val, %d test images)",
                output_dir, n_per_split["train"], n_per_split["valid"], n_per_split["test"])
    return str(yaml_path)


def validate_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Checks dataset structure and reports class distribution.

    Args:
        data_dir: Root dataset directory.

    Returns:
        Validation report dict with structure_ok, class_distribution, split_counts, issues list.
    """
    data_path = Path(data_dir)
    report: Dict[str, Any] = {
        "structure_ok": False,
        "class_distribution": {},
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "issues": [],
        "total_annotations": 0,
    }

    if not data_path.exists():
        report["issues"].append(f"Directory does not exist: {data_dir}")
        return report

    # Check for data.yaml
    yaml_files = list(data_path.rglob("data.yaml"))
    if not yaml_files:
        report["issues"].append("No data.yaml found.")
    else:
        report["yaml_path"] = str(yaml_files[0])

    # Count images and annotations per split
    class_counts: Dict[int, int] = {}
    total_annotations = 0

    for split, key in [("train", "train"), ("valid", "val"), ("test", "test")]:
        for subdir in [split, "val" if split == "valid" else split]:
            img_dir = data_path / subdir / "images"
            lbl_dir = data_path / subdir / "labels"
            if not img_dir.exists():
                img_dir = data_path / subdir
                lbl_dir = data_path / subdir

            if img_dir.exists():
                images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                report["split_counts"][key] = len(images)

                if lbl_dir.exists():
                    for lbl in lbl_dir.glob("*.txt"):
                        try:
                            with open(lbl, "r") as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        cls = int(parts[0])
                                        class_counts[cls] = class_counts.get(cls, 0) + 1
                                        total_annotations += 1
                        except (OSError, ValueError):
                            pass
                break

    report["total_annotations"] = total_annotations
    report["class_distribution"] = {
        RETAIL_CLASSES[k] if k < len(RETAIL_CLASSES) else str(k): v
        for k, v in sorted(class_counts.items())
    }

    total_images = sum(report["split_counts"].values())
    if total_images == 0:
        report["issues"].append("No images found in any split.")
    else:
        report["structure_ok"] = True

    if report["split_counts"]["train"] == 0:
        report["issues"].append("No training images found.")
        report["structure_ok"] = False

    logger.info("Dataset validation: %d total images, %d annotations, issues: %s",
                total_images, total_annotations, report["issues"] or "none")
    return report

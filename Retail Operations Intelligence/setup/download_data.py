"""
Retail Operations Intelligence & Loss Prevention Platform
Data Download Script

Instructions:
  1. SKU-110K Dataset (retail product images, 110,628 images)
  2. Open Images V7 (retail category subset)
  3. Both saved to data/raw/

Usage:
    python setup/download_data.py --dataset sku110k
    python setup/download_data.py --dataset openimages
    python setup/download_data.py --demo
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root (one level up from setup/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# SKU-110K
# ---------------------------------------------------------------------------
SKU110K_README = """
SKU-110K Dataset — Retail Product Detection
============================================
Source: https://github.com/eg4000/SKU110K_CVPR19
Paper: "Precise Detection in Densely Packed Scenes" (CVPR 2019)

Manual Download Instructions:
------------------------------
1. Visit: https://github.com/eg4000/SKU110K_CVPR19#dataset
2. Click the Google Drive download link on the README page
3. Download SKU110K_fixed.tar.gz (~11 GB)
4. Extract to: {data_raw}/sku110k/
   tar -xzf SKU110K_fixed.tar.gz -C {data_raw}/sku110k/

OR use the script approach below (requires gdown):
    pip install gdown
    gdown "https://drive.google.com/uc?id=1iq93lCdhaPUN0fWbLieMtzfB1850pKwd"
    tar -xzf SKU110K_fixed.tar.gz -C {data_raw}/sku110k/

Expected directory structure after extraction:
    data/raw/sku110k/
        images/
            train/    (8233 images)
            val/      (588 images)
            test/     (2936 images)
        annotations/
            annotations_train.csv
            annotations_val.csv
            annotations_test.csv

Dataset statistics:
    - 110,712 images total
    - Dense retail shelf scenes
    - Bounding box annotations for all products
    - CVPR 2019 benchmark dataset
""".format(data_raw=DATA_RAW)


def download_sku110k(output_dir: Path) -> bool:
    """
    Attempts to download SKU-110K dataset programmatically via gdown (Google Drive).
    Falls back to printing manual instructions if gdown is unavailable.
    """
    logger.info("=== SKU-110K Dataset Download ===")

    sku_dir = output_dir / "sku110k"
    sku_dir.mkdir(parents=True, exist_ok=True)

    readme_path = sku_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_path, "w") as f:
        f.write(SKU110K_README)
    logger.info("Instructions written to %s", readme_path)

    try:
        import gdown  # type: ignore
        # SKU-110K Google Drive file ID
        file_id = "1iq93lCdhaPUN0fWbLieMtzfB1850pKwd"
        url = f"https://drive.google.com/uc?id={file_id}"
        archive_path = sku_dir / "SKU110K_fixed.tar.gz"

        logger.info("Downloading SKU-110K via gdown (~11 GB)...")
        gdown.download(url, str(archive_path), quiet=False)

        if archive_path.exists() and archive_path.stat().st_size > 1_000_000:
            logger.info("Extracting archive...")
            shutil.unpack_archive(str(archive_path), str(sku_dir))
            logger.info("SKU-110K extracted to %s", sku_dir)
            return True
        else:
            logger.error("Download may have failed; archive too small.")
            return False

    except ImportError:
        logger.warning("gdown not installed (pip install gdown). Please follow manual instructions:")
        print(SKU110K_README)
        return False
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        print(SKU110K_README)
        return False


# ---------------------------------------------------------------------------
# Open Images V7
# ---------------------------------------------------------------------------
OPENIMAGES_README = """
Open Images V7 — Retail Category Subset
=========================================
Source: https://storage.googleapis.com/openimages/web/index.html
Documentation: https://github.com/cvdfoundation/open-images-dataset

Recommended Tool: OIDv7 Toolkit (pip install openimages)
---------------------------------------------------------
Install:
    pip install openimages

Download retail-relevant categories:
    python -m openimages download --category "Bottle,Can,Food,Drink,Bread,Cake,Cookie,Dairy Product,Fruit,Vegetable,Tin can" --type train --limit 5000 --dest {data_raw}/openimages/

Categories relevant for retail:
    - Product / Packaged goods
    - Person (for customer tracking)
    - Shopping cart / Basket
    - Food items (produce, packaged food)
    - Bottle, Can, Tin can

Alternative: fiftyone library
------------------------------
    pip install fiftyone
    import fiftyone.zoo as foz
    dataset = foz.load_zoo_dataset("open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Bottle","Person","Shopping cart","Tin can"],
        max_samples=2000,
        dataset_dir="{data_raw}/openimages/",
    )

Manual download (subset via CSV):
    1. Download class descriptions: https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
    2. Download train annotations: https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
    3. Use the official scripts at: https://github.com/openimages/dataset/blob/master/downloader.py

Expected output structure:
    data/raw/openimages/
        images/
        labels/  (YOLO format after conversion)
        metadata.json
""".format(data_raw=DATA_RAW)


def download_openimages(output_dir: Path, categories: list, limit: int = 500) -> bool:
    """
    Downloads Open Images V7 retail subset using the openimages package or fiftyone.
    """
    logger.info("=== Open Images V7 Download ===")
    oi_dir = output_dir / "openimages"
    oi_dir.mkdir(parents=True, exist_ok=True)

    readme_path = oi_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_path, "w") as f:
        f.write(OPENIMAGES_README)

    # Try fiftyone first
    try:
        import fiftyone.zoo as foz  # type: ignore
        logger.info("Using fiftyone to download Open Images V7...")
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=categories,
            max_samples=limit,
            dataset_dir=str(oi_dir),
        )
        logger.info("Downloaded %d samples via fiftyone", len(dataset))
        return True
    except ImportError:
        logger.warning("fiftyone not installed. Trying openimages package...")

    # Try openimages package
    try:
        import openimages  # type: ignore  # noqa
        cats_str = ",".join(categories)
        cmd = [
            sys.executable, "-m", "openimages", "download",
            "--category", cats_str,
            "--type", "train",
            "--limit", str(limit),
            "--dest", str(oi_dir),
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except ImportError:
        logger.warning("openimages not installed. Please follow manual instructions.")
        print(OPENIMAGES_README)
        return False
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def create_demo_data(output_dir: Path) -> str:
    """Creates a minimal demo dataset for testing the pipeline without real data."""
    logger.info("Creating demo dataset at %s", output_dir)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from data_loader import create_demo_dataset  # type: ignore
        yaml_path = create_demo_dataset(str(output_dir / "demo"))
        logger.info("Demo dataset ready: %s", yaml_path)
        return yaml_path
    except Exception as exc:
        logger.error("Failed to create demo dataset: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download retail datasets for the Object Detection platform."
    )
    parser.add_argument(
        "--dataset", choices=["sku110k", "openimages", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Create a minimal demo dataset for testing (no download required)",
    )
    parser.add_argument(
        "--output", default=str(DATA_RAW),
        help=f"Output directory (default: {DATA_RAW})",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max number of images for Open Images download (default: 500)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    if args.demo:
        yaml_path = create_demo_data(output_dir)
        if yaml_path:
            print(f"\nDemo dataset ready. Use this YAML path for training:\n  {yaml_path}")
        return

    if args.dataset in ("sku110k", "all"):
        success = download_sku110k(output_dir)
        status = "SUCCESS" if success else "FAILED (see instructions above)"
        logger.info("SKU-110K download: %s", status)

    if args.dataset in ("openimages", "all"):
        retail_categories = [
            "Bottle", "Can", "Food", "Drink", "Person",
            "Shopping cart", "Tin can", "Shelf", "Fruit", "Vegetable",
        ]
        success = download_openimages(output_dir, retail_categories, limit=args.limit)
        status = "SUCCESS" if success else "FAILED (see instructions above)"
        logger.info("Open Images V7 download: %s", status)

    if not args.dataset and not args.demo:
        parser.print_help()


if __name__ == "__main__":
    main()

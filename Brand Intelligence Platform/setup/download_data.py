"""
Data download script for the Enterprise Brand Intelligence Platform.
Downloads the Yelp Open Dataset from Kaggle or provides manual instructions.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_RAW = _BASE_DIR / "data" / "raw"

KAGGLE_DATASET = "yelp-dataset/yelp-dataset"
EXPECTED_FILES = [
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_user.json",
]

MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD INSTRUCTIONS — YELP OPEN DATASET          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Option A: Kaggle CLI                                                ║
║  1. Create a Kaggle account at https://www.kaggle.com               ║
║  2. Go to: https://www.kaggle.com/settings → API → Create New Token ║
║  3. Save the downloaded kaggle.json to: ~/.kaggle/kaggle.json        ║
║     Windows: C:\\Users\\<YourUser>\\.kaggle\\kaggle.json              ║
║  4. Run:                                                             ║
║     pip install kaggle                                               ║
║     kaggle datasets download -d yelp-dataset/yelp-dataset \\         ║
║       -p data/raw/ --unzip                                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Option B: Direct Download                                           ║
║  1. Visit: https://www.yelp.com/dataset/download                    ║
║  2. Fill in the access form and download the dataset                 ║
║  3. Extract all JSON files to:                                       ║
║     data/raw/yelp_academic_dataset_review.json                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Option C: Use Synthetic Demo Data                                   ║
║  The platform auto-generates 2000 realistic synthetic reviews        ║
║  when the Yelp dataset is not present. No download required.         ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def install_kaggle() -> bool:
    """Attempt to install the kaggle package."""
    logger.info("Attempting to install kaggle package...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
            timeout=120,
        )
        logger.info("kaggle installed successfully")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to install kaggle: %s", exc)
        return False
    except subprocess.TimeoutExpired:
        logger.error("kaggle installation timed out")
        return False


def check_kaggle_credentials() -> bool:
    """Check that ~/.kaggle/kaggle.json exists."""
    from pathlib import Path as P
    candidates = [
        P.home() / ".kaggle" / "kaggle.json",
        P("~/.kaggle/kaggle.json").expanduser(),
    ]
    for p in candidates:
        if p.exists():
            logger.info("Kaggle credentials found at %s", p)
            return True
    logger.warning("Kaggle credentials not found at ~/.kaggle/kaggle.json")
    return False


def download_via_kaggle() -> bool:
    """Download Yelp dataset using the Kaggle CLI."""
    _DATA_RAW.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Yelp dataset from Kaggle to %s", _DATA_RAW)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(_DATA_RAW),
                "--unzip",
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            logger.info("Download complete:\n%s", result.stdout)
            return True
        else:
            logger.error("Kaggle download failed:\n%s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("Kaggle download timed out after 600 seconds")
        return False
    except FileNotFoundError as exc:
        logger.error("Kaggle CLI not found: %s", exc)
        return False


def verify_files() -> bool:
    """Verify that expected dataset files exist."""
    all_found = True
    for filename in EXPECTED_FILES:
        path = _DATA_RAW / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("FOUND: %s (%.1f MB)", filename, size_mb)
        else:
            logger.warning("MISSING: %s", filename)
            all_found = False
    return all_found


def main() -> None:
    """Main entry point for data download script."""
    logger.info("=== Enterprise Brand Intelligence Platform — Data Downloader ===")
    logger.info("Target directory: %s", _DATA_RAW)

    # Check if already downloaded
    if verify_files():
        logger.info("All dataset files already present. Nothing to download.")
        return

    # Try to import kaggle
    try:
        import kaggle  # noqa: F401
        logger.info("kaggle package already installed")
    except ImportError:
        logger.info("kaggle package not installed")
        if not install_kaggle():
            print(MANUAL_INSTRUCTIONS)
            sys.exit(1)

    # Check credentials
    if not check_kaggle_credentials():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    # Download
    if download_via_kaggle():
        if verify_files():
            logger.info("All files verified. Dataset ready.")
        else:
            logger.warning("Some files are missing after download.")
            print(MANUAL_INSTRUCTIONS)
    else:
        logger.error("Download failed.")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Data Download Script — Customer Lifetime Value & Retention Platform
Downloads KKBox and IBM Telco datasets from Kaggle.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"SUCCESS: {description}")
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"FAILED: {description}")
        logger.error(f"stderr: {exc.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"kaggle CLI not found. Install with: pip install kaggle")
        return False


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
            "Setup instructions:\n"
            "  1. Go to https://www.kaggle.com/account\n"
            "  2. Click 'Create New API Token'\n"
            "  3. Move downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "  4. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)"
        )
        return False
    return True


def download_kkbox(data_dir: Path) -> bool:
    """
    Download KKBox Music Streaming Churn dataset.
    Competition: https://www.kaggle.com/c/kkbox-churn-prediction-challenge
    Files: train.csv, members_v3.csv, transactions_v2.csv, user_logs_v2.csv
    """
    return run_command(
        [
            "kaggle", "competitions", "download",
            "-c", "kkbox-churn-prediction-challenge",
            "-p", str(data_dir),
            "--unzip",
        ],
        "KKBox Churn Prediction Challenge dataset",
    )


def download_ibm_telco(data_dir: Path) -> bool:
    """
    Download IBM Telco Customer Churn dataset.
    Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    File: WA_Fn-UseC_-Telco-Customer-Churn.csv (7,043 customers)
    """
    return run_command(
        [
            "kaggle", "datasets", "download",
            "-d", "blastchar/telco-customer-churn",
            "-p", str(data_dir),
            "--unzip",
        ],
        "IBM Telco Customer Churn dataset",
    )


def print_manual_instructions() -> None:
    """Print manual download instructions if automated download fails."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Option 1 — KKBox Music Streaming Churn (2.6M users)")
    print("  URL: https://www.kaggle.com/c/kkbox-churn-prediction-challenge")
    print("  Files to download: train.csv, members_v3.csv, transactions_v2.csv")
    print(f"  Save to: {DATA_DIR}")
    print()
    print("Option 2 — IBM Telco Customer Churn (7,043 customers)")
    print("  URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("  File: WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"  Save to: {DATA_DIR}")
    print()
    print("Option 3 — Use existing customer_churn.csv in project root (auto-detected)")
    print()
    print("Option 4 — Synthetic data (automatic fallback, 200,000 users)")
    print("  No download needed — generated automatically if no data found")
    print()
    print("=" * 70)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")

    if not check_kaggle_credentials():
        print_manual_instructions()
        sys.exit(1)

    success_kkbox = download_kkbox(DATA_DIR)
    if not success_kkbox:
        logger.warning("KKBox download failed — trying IBM Telco...")
        success_telco = download_ibm_telco(DATA_DIR)
        if not success_telco:
            logger.warning("IBM Telco download also failed.")
            print_manual_instructions()
            logger.info("Will use synthetic data (200k users) as fallback.")
        else:
            logger.info("IBM Telco dataset downloaded successfully.")
    else:
        logger.info("KKBox dataset downloaded successfully.")

    # List downloaded files
    downloaded = list(DATA_DIR.glob("*.csv"))
    if downloaded:
        logger.info(f"Files in {DATA_DIR}:")
        for f in downloaded:
            logger.info(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

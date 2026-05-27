"""
Dataset Download Script
Real-Time Credit Risk & Fraud Detection Intelligence Platform

Automates downloading the two primary datasets from Kaggle.
Falls back to detailed manual instructions if Kaggle CLI is not available.

Usage:
    python setup/download_data.py

Requirements:
    pip install kaggle
    Configure ~/.kaggle/kaggle.json with your API credentials.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("download_data")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ─────────────────────────────────────────────
# Helper: run subprocess command
# ─────────────────────────────────────────────

def run_cmd(cmd: list[str], cwd: Path | None = None) -> bool:
    """
    Run a shell command and return True on success.

    Args:
        cmd: Command as list of strings.
        cwd: Working directory.

    Returns:
        True if exit code 0, False otherwise.
    """
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info(f"Success: {cmd_str}")
            if result.stdout.strip():
                logger.debug(result.stdout.strip())
            return True
        else:
            logger.error(f"Failed (exit {result.returncode}): {cmd_str}")
            logger.error(f"STDERR: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {cmd_str}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error running command: {exc}")
        return False


# ─────────────────────────────────────────────
# Check Kaggle CLI
# ─────────────────────────────────────────────

def check_kaggle_cli() -> bool:
    """Check whether the Kaggle CLI is installed and configured."""
    result = subprocess.run(
        ["kaggle", "--version"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.warning("Kaggle CLI not found. Install with: pip install kaggle")
        return False

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.warning(
            f"Kaggle credentials not found at {kaggle_json}. "
            "Create your API token at: https://www.kaggle.com/settings → API → Create New Token"
        )
        return False

    # Check permissions (Unix only)
    if os.name != "nt":
        perms = oct(kaggle_json.stat().st_mode)[-3:]
        if perms != "600":
            logger.warning(f"Setting kaggle.json permissions to 600")
            os.chmod(kaggle_json, 0o600)

    logger.info("Kaggle CLI is configured.")
    return True


# ─────────────────────────────────────────────
# Dataset 1: IEEE-CIS Fraud Detection
# ─────────────────────────────────────────────

def download_ieee_cis() -> bool:
    """
    Download IEEE-CIS Fraud Detection dataset from Kaggle competition.

    Competition: https://www.kaggle.com/c/ieee-fraud-detection
    Files: train_transaction.csv (~440 MB), train_identity.csv (~50 MB)

    NOTE: You must accept the competition rules at:
    https://www.kaggle.com/c/ieee-fraud-detection/rules
    before the CLI download will succeed.

    Returns:
        True if download succeeded.
    """
    logger.info("=" * 60)
    logger.info("Downloading IEEE-CIS Fraud Detection Dataset")
    logger.info("Competition: https://www.kaggle.com/c/ieee-fraud-detection")
    logger.info(
        "IMPORTANT: You must accept competition rules at "
        "https://www.kaggle.com/c/ieee-fraud-detection/rules"
    )
    logger.info("=" * 60)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    success = run_cmd(
        [
            "kaggle", "competitions", "download",
            "-c", "ieee-fraud-detection",
            "-p", str(RAW_DATA_DIR),
            "--unzip",
        ]
    )

    if success:
        trans_file = RAW_DATA_DIR / "train_transaction.csv"
        ident_file = RAW_DATA_DIR / "train_identity.csv"
        if trans_file.exists() and ident_file.exists():
            logger.info(f"IEEE-CIS files ready: {trans_file} | {ident_file}")
            return True
        else:
            logger.warning("Download reported success but expected files not found")
            return False
    return False


# ─────────────────────────────────────────────
# Dataset 2: Credit Card Fraud 2023
# ─────────────────────────────────────────────

def download_creditcard_2023() -> bool:
    """
    Download Credit Card Fraud Detection Dataset 2023 from Kaggle.

    Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
    File: creditcard_2023.csv (~140 MB, 568k rows)

    Returns:
        True if download succeeded.
    """
    logger.info("=" * 60)
    logger.info("Downloading Credit Card Fraud 2023 Dataset")
    logger.info("URL: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    logger.info("=" * 60)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    success = run_cmd(
        [
            "kaggle", "datasets", "download",
            "-d", "nelgiriyewithana/credit-card-fraud-detection-dataset-2023",
            "-p", str(RAW_DATA_DIR),
            "--unzip",
        ]
    )

    if success:
        cc_file = RAW_DATA_DIR / "creditcard_2023.csv"
        if cc_file.exists():
            logger.info(f"Credit Card 2023 file ready: {cc_file}")
            return True
        else:
            logger.warning("Download reported success but creditcard_2023.csv not found")
            return False
    return False


# ─────────────────────────────────────────────
# Manual Instructions
# ─────────────────────────────────────────────

MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║         MANUAL DATASET DOWNLOAD INSTRUCTIONS                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION A: Install Kaggle CLI (Recommended)                  ║
║  ─────────────────────────────────────────                   ║
║  1. pip install kaggle                                       ║
║                                                              ║
║  2. Create Kaggle API token:                                 ║
║     → https://www.kaggle.com/settings → API                  ║
║     → Click "Create New Token"                               ║
║     → Save kaggle.json to ~/.kaggle/kaggle.json              ║
║     → chmod 600 ~/.kaggle/kaggle.json   (Linux/Mac)          ║
║                                                              ║
║  3. Accept IEEE-CIS competition rules:                       ║
║     → https://www.kaggle.com/c/ieee-fraud-detection/rules    ║
║                                                              ║
║  4. Run download commands:                                   ║
║     kaggle competitions download \\                           ║
║       -c ieee-fraud-detection \\                              ║
║       -p data/raw/ --unzip                                   ║
║                                                              ║
║     kaggle datasets download \\                               ║
║       -d nelgiriyewithana/credit-card-fraud-detection-dataset-2023 \\
║       -p data/raw/ --unzip                                   ║
║                                                              ║
║  OPTION B: Manual Browser Download                           ║
║  ─────────────────────────────────                           ║
║  IEEE-CIS Fraud Detection:                                   ║
║  → https://www.kaggle.com/c/ieee-fraud-detection/data        ║
║  → Download: train_transaction.csv, train_identity.csv       ║
║  → Place files in: data/raw/                                 ║
║                                                              ║
║  Credit Card Fraud 2023:                                     ║
║  → https://www.kaggle.com/datasets/nelgiriyewithana/         ║
║      credit-card-fraud-detection-dataset-2023                ║
║  → Download: creditcard_2023.csv                             ║
║  → Place file in: data/raw/                                  ║
║                                                              ║
║  OPTION C: Use Synthetic Data (No Download Required)         ║
║  ──────────────────────────────────────────────────          ║
║  The platform auto-generates 500,000 synthetic transactions  ║
║  with realistic fraud patterns if real data is not found.    ║
║  Simply proceed with training:                               ║
║    python src/train.py                                       ║
║                                                              ║
║  Expected file locations after download:                     ║
║    data/raw/train_transaction.csv   (~440 MB)                ║
║    data/raw/train_identity.csv      (~50 MB)                 ║
║    data/raw/creditcard_2023.csv     (~140 MB)                ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    """
    Main entry point: detect Kaggle CLI, download datasets, or print manual instructions.
    """
    logger.info("Credit Risk & Fraud Detection — Data Download Script")
    logger.info(f"Target directory: {RAW_DATA_DIR}")
    logger.info("")

    # Check if files already exist
    trans_exists = (RAW_DATA_DIR / "train_transaction.csv").exists()
    ident_exists = (RAW_DATA_DIR / "train_identity.csv").exists()
    cc_exists = (RAW_DATA_DIR / "creditcard_2023.csv").exists()

    if trans_exists and ident_exists:
        logger.info("IEEE-CIS dataset already present — skipping download")
    if cc_exists:
        logger.info("Credit Card 2023 dataset already present — skipping download")

    if trans_exists and ident_exists and cc_exists:
        logger.info("All datasets present. You are ready to train!")
        logger.info("Run: python src/train.py")
        return

    # Check Kaggle CLI
    if not check_kaggle_cli():
        print(MANUAL_INSTRUCTIONS)
        logger.info(
            "Kaggle CLI not available. See manual instructions above. "
            "Or proceed with synthetic data: python src/train.py"
        )
        sys.exit(0)

    # Try downloads
    ieee_ok = False
    cc_ok = False

    if not (trans_exists and ident_exists):
        ieee_ok = download_ieee_cis()
        if not ieee_ok:
            logger.warning(
                "IEEE-CIS download failed. Common causes:\n"
                "  1. You haven't accepted competition rules at "
                "https://www.kaggle.com/c/ieee-fraud-detection/rules\n"
                "  2. Kaggle credentials expired — generate a new token\n"
                "  3. Network connectivity issue"
            )
    else:
        ieee_ok = True

    if not cc_exists:
        cc_ok = download_creditcard_2023()
        if not cc_ok:
            logger.warning(
                "Credit Card 2023 download failed. "
                "Try manual download from: "
                "https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
            )
    else:
        cc_ok = True

    # Final status
    logger.info("")
    logger.info("=" * 50)
    logger.info("DOWNLOAD STATUS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"IEEE-CIS Fraud Detection:    {'OK' if ieee_ok else 'FAILED'}")
    logger.info(f"Credit Card Fraud 2023:      {'OK' if cc_ok else 'FAILED'}")

    if not ieee_ok and not cc_ok:
        logger.info("")
        logger.info("No real data downloaded. Synthetic data will be used automatically.")
        logger.info("Proceed with: python src/train.py")
        print(MANUAL_INSTRUCTIONS)
    else:
        logger.info("")
        logger.info("Ready to train! Run: python src/train.py")


if __name__ == "__main__":
    main()

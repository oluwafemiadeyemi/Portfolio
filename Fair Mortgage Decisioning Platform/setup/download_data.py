"""
HMDA 2022 Modified LAR Data Download Instructions and Automation
Source: CFPB FFIEC HMDA Data Publication
URL: https://ffiec.cfpb.gov/data-publication/modified-lar/2022
"""

import logging
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Download configuration ─────────────────────────────────────────────────────
HMDA_BASE_URL = "https://s3.amazonaws.com/cfpb-hmda-public/prod/lar/2022/2022_public_lar.zip"
HMDA_DIRECT_CSV = "https://ffiec.cfpb.gov/data-publication/modified-lar/2022"
TARGET_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Alternative state-level download for smaller files
STATE_DOWNLOAD_TEMPLATE = (
    "https://ffiec.cfpb.gov/data-browser/data/2022/derived-mlar/states/{state_code}"
    "?actions_taken=1,2,3&years=2022&output=csv"
)

MAJOR_STATES = {
    "CA": "California (largest market)",
    "TX": "Texas",
    "FL": "Florida",
    "NY": "New York",
    "IL": "Illinois",
}


def print_manual_download_instructions() -> None:
    """Prints step-by-step instructions for manual HMDA data download."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           HMDA 2022 MODIFIED LAR DATA DOWNLOAD INSTRUCTIONS                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

OPTION 1 — FFIEC HMDA Data Browser (Recommended, Filtered)
──────────────────────────────────────────────────────────
1. Go to: https://ffiec.cfpb.gov/data-browser/data/2022
2. Select: "HMDA Data" → "Modified LAR"
3. Under "Geography": Select "Nationwide" OR specific states
4. Under "Filters":
   - Actions Taken: 1 (Loan Originated), 2 (Approved Not Accepted), 3 (Application Denied)
5. Click "Download Data" → Choose CSV format
6. Save as: data/raw/hmda_2022.csv

OPTION 2 — Direct S3 Download (Full dataset, ~5GB)
──────────────────────────────────────────────────
URL: https://s3.amazonaws.com/cfpb-hmda-public/prod/lar/2022/2022_public_lar.zip

Warning: This is the complete national dataset (~14M records, ~5GB compressed).
Recommended for production use only.

Steps:
  wget https://s3.amazonaws.com/cfpb-hmda-public/prod/lar/2022/2022_public_lar.zip
  unzip 2022_public_lar.zip -d data/raw/
  mv data/raw/2022_public_lar.txt data/raw/2022_public_lar.csv

OPTION 3 — State-Level Files (Smaller, Faster)
───────────────────────────────────────────────
For a representative sample, download top 5 states:

California:
  https://ffiec.cfpb.gov/data-browser/data/2022/derived-mlar/states/CA?actions_taken=1,2,3&years=2022&output=csv

Texas:
  https://ffiec.cfpb.gov/data-browser/data/2022/derived-mlar/states/TX?actions_taken=1,2,3&years=2022&output=csv

Florida:
  https://ffiec.cfpb.gov/data-browser/data/2022/derived-mlar/states/FL?actions_taken=1,2,3&years=2022&output=csv

Merge downloaded files and save as: data/raw/hmda_2022.csv

OPTION 4 — Automated (this script)
────────────────────────────────────
Run: python setup/download_data.py --auto

IMPORTANT COLUMN NAMES (HMDA 2022 Modified LAR):
  - action_taken (1=originated, 2=approved/not accepted, 3=denied)
  - loan_amount (in dollars, 2022+ uses full dollars not thousands)
  - income (in thousands; multiply by 1000 after loading)
  - debt_to_income_ratio (reported as range or percentage)
  - loan_to_value_ratio (numeric percentage)
  - applicant_sex (1=male, 2=female, 3=info not provided)
  - applicant_race_1 (1-7 race codes)
  - applicant_age (age ranges: 25-34, 35-44, etc.)
  - county_code (5-digit FIPS county code)
  - census_tract (11-digit census tract)

REGULATORY NOTES:
  - Modified LAR excludes direct identifiers to protect applicant privacy
  - Complete LAR (with institution identifiers) requires HMDA Reporter agreement
  - Data dictionary: https://ffiec.cfpb.gov/documentation/publications/modified-lar/
  - HMDA Filing Instructions: https://www.consumerfinance.gov/data-research/hmda/
"""
    print(instructions)


def download_state_sample(state_code: str, output_dir: Path) -> bool:
    """
    Attempts to download HMDA data for a specific state via the CFPB API.

    Args:
        state_code: Two-letter state code (e.g., 'CA')
        output_dir: Directory to save the downloaded file

    Returns:
        True if download successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hmda_2022_{state_code.lower()}.csv"

    url = (
        f"https://ffiec.cfpb.gov/data-browser/data/2022/derived-mlar/states/{state_code}"
        f"?actions_taken=1,2,3&years=2022&output=csv"
    )

    logger.info(f"Downloading HMDA 2022 data for {state_code}...")
    logger.info(f"URL: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Python HMDA Downloader; research purposes)",
            "Accept": "text/csv,application/csv",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read()
        with open(output_path, "wb") as f:
            f.write(content)
        file_size = output_path.stat().st_size
        logger.info(f"Downloaded {file_size:,} bytes to {output_path}")
        return True
    except Exception as exc:
        logger.warning(f"Download failed for {state_code}: {exc}")
        return False


def verify_hmda_data(filepath: Path) -> dict:
    """
    Verifies downloaded HMDA data file integrity and column presence.

    Args:
        filepath: Path to HMDA CSV file

    Returns:
        Dict with verification results
    """
    import pandas as pd

    result = {"valid": False, "n_rows": 0, "columns_found": [], "missing_columns": []}
    required_cols = [
        "action_taken",
        "loan_amount",
        "income",
        "applicant_sex",
        "applicant_race_1",
        "county_code",
    ]

    try:
        df = pd.read_csv(filepath, nrows=100, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]

        result["columns_found"] = list(df.columns)
        result["missing_columns"] = [c for c in required_cols if c not in df.columns]
        result["n_rows"] = len(pd.read_csv(filepath, usecols=[0], low_memory=False))
        result["valid"] = len(result["missing_columns"]) == 0
        result["file_size_mb"] = round(filepath.stat().st_size / 1e6, 2)

        if result["valid"]:
            logger.info(f"HMDA data verified: {result['n_rows']:,} rows, {len(result['columns_found'])} columns")
        else:
            logger.warning(f"Missing columns: {result['missing_columns']}")

    except Exception as exc:
        result["error"] = str(exc)
        logger.error(f"Verification failed: {exc}")

    return result


def auto_download(output_dir: Path = TARGET_DIR, states: list = None) -> bool:
    """
    Automatically downloads HMDA data for specified states.

    Args:
        output_dir: Target directory
        states: List of state codes (defaults to top 3 states)

    Returns:
        True if at least one download succeeded
    """
    if states is None:
        states = ["CA", "TX", "FL"]

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for state in states:
        if download_state_sample(state, output_dir):
            success_count += 1

    if success_count > 0:
        # Merge state files
        import pandas as pd
        dfs = []
        for state in states:
            fp = output_dir / f"hmda_2022_{state.lower()}.csv"
            if fp.exists():
                try:
                    df = pd.read_csv(fp, low_memory=False, nrows=200000)
                    df["state_source"] = state
                    dfs.append(df)
                    logger.info(f"Loaded {len(df):,} rows from {state}")
                except Exception as exc:
                    logger.warning(f"Could not read {fp}: {exc}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            merged_path = output_dir / "hmda_2022.csv"
            combined.to_csv(merged_path, index=False)
            logger.info(f"Merged dataset saved: {merged_path} ({len(combined):,} rows)")
            return True

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HMDA 2022 Data Download Utility")
    parser.add_argument("--auto", action="store_true", help="Auto-download top 3 states")
    parser.add_argument("--states", nargs="+", default=["CA", "TX", "FL"], help="States to download")
    parser.add_argument("--output-dir", default=str(TARGET_DIR), help="Output directory")
    args = parser.parse_args()

    print_manual_download_instructions()

    if args.auto:
        output_path = Path(args.output_dir)
        logger.info(f"Auto-downloading HMDA data for: {args.states}")
        success = auto_download(output_path, args.states)
        if success:
            logger.info("Download complete!")
        else:
            logger.warning("Auto-download failed. The app will use synthetic data as fallback.")
    else:
        logger.info("Run with --auto flag to attempt automatic download.")
        logger.info("The application will generate synthetic data if no HMDA file is found.")

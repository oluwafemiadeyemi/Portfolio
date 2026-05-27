"""
setup/download_data.py
======================
Instructions and helpers for downloading real Parkinson's Disease datasets.

Datasets:
  1. mPower (9,500+ participants) — Synapse platform
  2. UCI Parkinson's Telemonitoring (5,875 recordings)
  3. PPMI (Parkinson's Progression Markers Initiative)
"""

from __future__ import annotations

import logging
import sys
import textwrap
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset 1: mPower (Synapse)
# ---------------------------------------------------------------------------

MPOWER_INSTRUCTIONS = """
============================================================
  mPower Parkinson's mHealth Research Study
  9,500+ participants | Synapse Platform
============================================================

Steps to access:
  1. Register for a free account at https://www.synapse.org
  2. Complete the user certification quiz (required for data access)
  3. Navigate to project syn4993293 (mPower Public Researcher Portal)
  4. Apply for data access — approval is typically within 48 hours
  5. Install the Synapse Python client:

       pip install synapseclient

  6. Download the data:

       import synapseclient
       syn = synapseclient.Synapse()
       syn.login("your_username", "your_password")

       # Download walking activity table
       walking = syn.tableQuery("SELECT * FROM syn10308918")
       walking_df = walking.asDataFrame()

       # Download voice activity table
       voice = syn.tableQuery("SELECT * FROM syn10308920")
       voice_df = voice.asDataFrame()

  7. Save files to: {data_dir}

  Data citation:
    Bot et al. (2016). The mPower study, Parkinson disease mobile data
    collected using ResearchKit. Scientific Data, 3, 160011.
    https://doi.org/10.1038/sdata.2016.11

  Note: Data use requires agreeing to data use terms. Must not attempt
  to re-identify participants. HIPAA considerations apply.
============================================================
""".format(data_dir=DATA_DIR)


# ---------------------------------------------------------------------------
# Dataset 2: UCI Parkinson's Telemonitoring
# ---------------------------------------------------------------------------

UCI_TELE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)

UCI_TELE_INSTRUCTIONS = """
============================================================
  UCI Parkinson's Telemonitoring Dataset
  5,875 recordings | 42 PD subjects | UPDRS scores
  Oxford/Intel Research 2009
============================================================

Direct download (no registration required):

  URL: {url}

  Python download:
      import urllib.request
      urllib.request.urlretrieve(
          "{url}",
          "{data_dir}/parkinsons_updrs.data"
      )

  Citation:
    Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O. (2009).
    Accurate Telemonitoring of Parkinson's Disease Progression by
    Non-invasive Speech Tests. IEEE Transactions on Biomedical Engineering.
    https://doi.org/10.1109/TBME.2009.2036000

  Columns: subject#, age, sex, test_time, motor_UPDRS, total_UPDRS,
           Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP,
           Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5,
           Shimmer:APQ11, Shimmer:DDA, NHR, HNR, RPDE, DFA, PPE
============================================================
""".format(url=UCI_TELE_URL, data_dir=DATA_DIR)


# ---------------------------------------------------------------------------
# Dataset 3: PPMI
# ---------------------------------------------------------------------------

PPMI_INSTRUCTIONS = """
============================================================
  Parkinson's Progression Markers Initiative (PPMI)
  The Michael J. Fox Foundation | Multi-site longitudinal study
============================================================

Steps to access:
  1. Register at https://www.ppmi-info.org
  2. Complete required CITI training courses:
     - Human Subjects Research — Social/Behavioral
     - Good Clinical Practice
  3. Sign the Data Use Agreement
  4. Access data through the PPMI Study Data Download portal
  5. Key datasets to download:
     - Motor assessments (UPDRS)
     - DaTscan imaging data
     - CSF biomarkers
     - Accelerometry / gait assessments

  Python access via PPMI API (after obtaining credentials):

      pip install ppmi-downloader

      from ppmi_downloader import PPMIDownloader
      dl = PPMIDownloader()
      dl.download_imaging_data(subject_ids=[...])

  Citation:
    Marek, K. et al. (2011). The Parkinson Progression Marker Initiative
    (PPMI). Progress in Neurobiology, 95(4), 629–635.
    https://doi.org/10.1016/j.pneurobio.2011.09.005

  Note: PPMI data requires IRB compliance and annual renewal of DUA.
============================================================
"""


# ---------------------------------------------------------------------------
# Auto-downloader for UCI (no auth required)
# ---------------------------------------------------------------------------

def download_uci_telemonitoring(output_dir: Path | None = None) -> Path:
    """Download the UCI Parkinson's Telemonitoring dataset automatically.

    Parameters
    ----------
    output_dir: Directory to save the file. Defaults to ``data/raw/``.

    Returns
    -------
    Path to the downloaded file.
    """
    out_dir = Path(output_dir) if output_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parkinsons_updrs.data"

    if out_path.exists():
        logger.info("UCI Telemonitoring file already exists at %s", out_path)
        return out_path

    logger.info("Downloading UCI Telemonitoring dataset from %s …", UCI_TELE_URL)
    try:
        urllib.request.urlretrieve(UCI_TELE_URL, out_path)
        logger.info("Downloaded successfully to %s", out_path)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        raise

    return out_path


# ---------------------------------------------------------------------------
# Main entry point — print all instructions
# ---------------------------------------------------------------------------

def main() -> None:
    """Print dataset download instructions to stdout."""
    print(MPOWER_INSTRUCTIONS)
    print(UCI_TELE_INSTRUCTIONS)
    print(PPMI_INSTRUCTIONS)

    print("\n" + "=" * 60)
    print("  Attempting auto-download of UCI Telemonitoring dataset…")
    print("=" * 60)
    try:
        path = download_uci_telemonitoring()
        print(f"\nSuccess! File saved to: {path}")
    except Exception as exc:
        print(f"\nAuto-download failed: {exc}")
        print("Please download manually using the instructions above.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

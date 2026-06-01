"""
Master runner for Projects 11-17.
Runs data pipelines, trains models, then starts all APIs and dashboards.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

BASE = Path(__file__).parent
PYTHON = sys.executable

PROJECTS = {
    11: {"name": "Marketing Campaign Intelligence",   "api_port": 8001, "dash_port": 8501},
    12: {"name": "Automotive Pricing Intelligence",   "api_port": 8002, "dash_port": 8502},
    13: {"name": "Loan Default Prediction",           "api_port": 8003, "dash_port": 8503},
    14: {"name": "Malaria Detection",                 "api_port": 8004, "dash_port": 8504},
    15: {"name": "Facial Emotion Detection",          "api_port": 8005, "dash_port": 8505},
    16: {"name": "Music Recommendation System",       "api_port": 8006, "dash_port": 8506},
    17: {"name": "Customer Review Categorisation",    "api_port": 8007, "dash_port": 8507},
}


def run(cmd, cwd=None, label="", timeout=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, cwd=cwd, shell=True,
        capture_output=False, timeout=timeout,
    )
    return result.returncode == 0


def start_bg(cmd, cwd=None):
    return subprocess.Popen(cmd, cwd=cwd, shell=True,
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                            if sys.platform == "win32" else 0)


# ── Step 1: Data Pipelines ────────────────────────────────────────────────────

def run_pipelines():
    # Project 11: Marketing
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import generate_synthetic_events, build_rfm_features, build_feature_matrix; import pandas as pd; from pathlib import Path; DATA_PROC=Path(\"data/processed\"); DATA_PROC.mkdir(parents=True,exist_ok=True); df=generate_synthetic_events(1_000_000); df.to_parquet(DATA_PROC/\"synthetic_events.parquet\",index=False); rfm=build_rfm_features(df); rfm.to_parquet(DATA_PROC/\"rfm_features.parquet\",index=False); fm,_=build_feature_matrix(df); import numpy as np; idx=np.random.default_rng(42).choice(len(fm),size=50000,replace=False); fm.iloc[idx].reset_index(drop=True).to_parquet(DATA_PROC/\"sample_features.parquet\",index=False); print(\"Marketing data done\")"',
        cwd=BASE / "Marketing Campaign Intelligence",
        label="P11: Generating 1M marketing events")

    # Project 12: Automotive
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import prepare_all; prepare_all(500_000)"',
        cwd=BASE / "Automotive Pricing Intelligence",
        label="P12: Generating 500k vehicle listings")

    # Project 13: Loans
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import prepare_all; prepare_all(500_000)"',
        cwd=BASE / "Loan Default Prediction",
        label="P13: Generating 500k loan applications")

    # Project 14: Malaria (synthetic only — no GPU training)
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import prepare_all; prepare_all(use_synthetic=True)"',
        cwd=BASE / "Malaria Detection",
        label="P14: Generating synthetic malaria cell images")

    # Project 15: Emotion (synthetic faces)
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import prepare_all; prepare_all(n_per_class=500)"',
        cwd=BASE / "Facial Emotion Detection",
        label="P15: Generating synthetic emotion face images")

    # Project 16: Music
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import prepare_all; prepare_all(target_events=2_000_000)"',
        cwd=BASE / "Music Recommendation System",
        label="P16: Generating 2M music listening events")

    # Project 17: Reviews
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from data_pipeline import prepare_all; prepare_all(n=100_000, sample_for_rag=10_000)"',
        cwd=BASE / "Customer Review Categorisation",
        label="P17: Generating 100k customer reviews")


# ── Step 2: Model Training ────────────────────────────────────────────────────

def train_models():
    # P11: RFM + Basket + Attribution (skip slow UMAP segmentation)
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from rfm_analysis import run_rfm_analysis; run_rfm_analysis()"',
        cwd=BASE / "Marketing Campaign Intelligence",
        label="P11: Computing RFM scores")

    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from basket_analysis import run_basket_analysis; run_basket_analysis()"',
        cwd=BASE / "Marketing Campaign Intelligence",
        label="P11: Running FP-Growth basket analysis")

    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from attribution import run_attribution_analysis; run_attribution_analysis()"',
        cwd=BASE / "Marketing Campaign Intelligence",
        label="P11: Computing multi-touch attribution")

    # P11: Segmentation (UMAP + HDBSCAN — sample of 30k for speed)
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from segmentation import run_segmentation; run_segmentation(sample_size=30000)"',
        cwd=BASE / "Marketing Campaign Intelligence",
        label="P11: UMAP + HDBSCAN segmentation (30k sample)")

    # P12: Automotive model (no Optuna tuning for speed)
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from model import load_data, train_ensemble, compute_shap_values, conformal_prediction_intervals; import joblib; X_tr,y_tr,X_te,y_te,fc=load_data(); models,metrics=train_ensemble(X_tr,y_tr,X_te,y_te,fc,tune=False); compute_shap_values(models,X_te); conformal_prediction_intervals(models,X_te,y_te)"',
        cwd=BASE / "Automotive Pricing Intelligence",
        label="P12: Training LightGBM+XGBoost+CatBoost ensemble")

    # P13: Loan model
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from model import run_training; run_training()"',
        cwd=BASE / "Loan Default Prediction",
        label="P13: Training credit risk ensemble + SHAP + fairness")

    # P14 + P15: Skip deep learning training (CPU too slow; demo mode works fine)
    print("\n[P14 Malaria] Skipping deep learning training on CPU — dashboard runs in demo mode.")
    print("[P15 Emotion] Skipping deep learning training on CPU — dashboard runs in demo mode.")

    # P16: ALS + Content-based FAISS
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); from recommender import train_all; train_all()"',
        cwd=BASE / "Music Recommendation System",
        label="P16: Training ALS + FAISS content-based index")

    # P17: Build ChromaDB vector store (Ollama not needed for indexing)
    run(f'"{PYTHON}" -c "import sys; sys.path.insert(0,\"src\"); import pandas as pd; from pathlib import Path; from rag_pipeline import build_vector_store; df=pd.read_parquet(Path(\"data/processed/rag_sample.parquet\")); build_vector_store(df)"',
        cwd=BASE / "Customer Review Categorisation",
        label="P17: Building ChromaDB vector store")


# ── Step 3: Start Services ────────────────────────────────────────────────────

def start_services():
    procs = []
    for num, p in PROJECTS.items():
        proj_dir = BASE / p["name"]
        api_port  = p["api_port"]
        dash_port = p["dash_port"]

        api_cmd  = f'"{PYTHON}" -m uvicorn src.api:app --host 0.0.0.0 --port {api_port}'
        dash_cmd = f'"{PYTHON}" -m streamlit run dashboard/app.py --server.port {dash_port} --server.headless true'

        print(f"Starting P{num} API  → http://localhost:{api_port}/docs")
        procs.append(start_bg(api_cmd, cwd=proj_dir))
        time.sleep(1)

        print(f"Starting P{num} Dash → http://localhost:{dash_port}")
        procs.append(start_bg(dash_cmd, cwd=proj_dir))
        time.sleep(1)

    return procs


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data",   action="store_true")
    parser.add_argument("--skip-train",  action="store_true")
    parser.add_argument("--skip-serve",  action="store_true")
    args = parser.parse_args()

    if not args.skip_data:
        print("\n>>> PHASE 1: DATA PIPELINES")
        run_pipelines()

    if not args.skip_train:
        print("\n>>> PHASE 2: MODEL TRAINING")
        train_models()

    if not args.skip_serve:
        print("\n>>> PHASE 3: STARTING ALL SERVICES")
        procs = start_services()

        print("\n" + "="*60)
        print("ALL 7 PROJECTS RUNNING")
        print("="*60)
        for num, p in PROJECTS.items():
            print(f"  P{num} {p['name'][:35]:<35} API:{p['api_port']}  Dash:{p['dash_port']}")
        print("\nPress Ctrl+C to stop all services.")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nShutting down...")
            for proc in procs:
                proc.terminate()

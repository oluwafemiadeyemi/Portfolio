# Real-Time Fraud Detection Platform

> **IEEE-CIS Scale Fraud Scoring with Drift Monitoring**

Score 590K transactions at sub-20ms latency with XGBoost/LightGBM ensemble, Evidently drift detection, and SHAP adverse action codes.

---

## Executive Summary

Payment fraud costs the global economy $32 billion annually. Traditional rule-based systems produce 70-80% false positive rates, blocking legitimate customers and generating $118 in review costs per flagged transaction. ML models without monitoring degrade silently — a distribution shift in merchant categories or device fingerprints can increase fraud rates 3× before anyone notices.

### Target Buyers
**JPMorgan Chase, American Express, Stripe, PayPal, Mastercard**

### Business ROI
Reducing false positives by 1% saves a mid-size bank $4M/year in review costs. AUC improvement from 0.88 to 0.94 translates to $12M annual fraud savings at $10B transaction volume.

---

## Screenshots

| Dashboard View |
|---|
| ![00 Overview](../screenshots/00_overview.png) |
| ![01 Overview](../screenshots/01_overview.png) |
| ![01 Roc Curve](../screenshots/01_roc_curve.png) |
| ![02 Confusion Matrix](../screenshots/02_confusion_matrix.png) |
| ![02 Transaction Analysis](../screenshots/02_transaction_analysis.png) |
| ![03 Feature Importance](../screenshots/03_feature_importance.png) |

---

## Dashboard Demo

> **Screen Recording** — Full navigation through all 6 dashboard tabs

[Watch Dashboard Demo](../recordings/P02_dashboard.mp4)

*The recording shows: `Overview` → `Transaction Analysis` → `Model Performance` → `Fairness Monitor` → `Drift Monitor` → `Alert Queue`*


---

## Problem Statement

Payment fraud costs the global economy $32 billion annually. Traditional rule-based systems produce 70-80% false positive rates, blocking legitimate customers and generating $118 in review costs per flagged transaction. ML models without monitoring degrade silently — a distribution shift in merchant categories or device fingerprints can increase fraud rates 3× before anyone notices.

## Technical Solution

A production-grade fraud scoring engine built on **XGBoost + LightGBM ensemble** trained on 590K IEEE-CIS transactions. Features include device fingerprinting, velocity signals, IP geolocation distance, and behavioral embeddings. **Evidently AI** provides Population Stability Index (PSI) monitoring for feature drift. **SHAP adverse action codes** explain every decline to satisfy Regulation E requirements.

## Dataset

IEEE-CIS Fraud Detection Dataset — 590,540 transactions, 433 engineered features, 3.5% fraud rate. Industry-standard benchmark used by JPMorgan and Stripe research teams.

## Tech Stack

`XGBoost, LightGBM, SHAP, Evidently AI, FastAPI, Streamlit, Plotly, scikit-learn`

## Key Results

| Metric | Value |
|---|---|
| **Ensemble AUC-ROC** | 0.9412 on held-out test set |
| **Precision at 10% Recall** | 0.891 — optimal for high-value transactions |
| **False Positive Rate** | 2.3% at operating threshold |
| **Inference Latency** | <20ms p99 (FastAPI + model cache) |
| **PSI Monitoring** | 42 features tracked — alert at PSI > 0.20 |

---

## Architecture Overview

```
Real-Time Fraud Detection/
├── dashboard/app.py          # Streamlit — port 8511
├── src/
│   ├── api.py                # FastAPI — port 8001
│   ├── model.py              # ML pipeline
│   └── data_pipeline.py     # ETL & preprocessing
├── models/                   # Trained model artifacts
├── data/
│   ├── raw/                  # Source datasets
│   └── processed/            # Feature-engineered data
├── docs/
│   ├── screenshots/          # Dashboard UI screenshots
│   └── recordings/           # Screen recording MP4
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the portfolio
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Real-Time Fraud Detection"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py --server.port 8511

# Launch API (separate terminal)
uvicorn src.api:app --port 8001 --reload
```

---

*Project P02 of 17 — Part of the [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

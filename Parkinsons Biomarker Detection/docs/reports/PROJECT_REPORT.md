# Parkinson's Disease Biomarker Detection

> **mPower Clinical Biomarkers from Voice, Gait & Tremor**

Detect Parkinson's biomarkers from smartphone voice recordings and gait data with multimodal ML — 87.3% accuracy on 9,500 mPower participants.

---

## Executive Summary

Parkinson's disease affects 10 million people worldwide with no cure. Early detection is critical — neuroprotective therapies are most effective in the first 2-5 years. Current diagnosis requires specialist neurological examination (average wait: 3-6 months), making early-stage detection inaccessible. Remote digital biomarkers from smartphones could enable population-scale screening at near-zero marginal cost.

### Target Buyers
**Pfizer, Johnson & Johnson, Roche, Apple Health Research, NIH**

### Business ROI
Digital biomarker screening at $0.01/test vs. $800 specialist neurological exam. Early detection enables $150K+ savings per patient in delayed disease progression costs.

---

## Screenshots

| Dashboard View |
|---|
| ![00 Overview](../screenshots/00_overview.png) |
| ![01 Biomarker Scatter](../screenshots/01_biomarker_scatter.png) |
| ![01 Population Distribution](../screenshots/01_population_distribution.png) |
| ![02 Roc Curve](../screenshots/02_roc_curve.png) |
| ![02 Violin Plots](../screenshots/02_violin_plots.png) |
| ![03 Correlation Matrix](../screenshots/03_correlation_matrix.png) |

---

## Dashboard Demo

> **Screen Recording** — Full navigation through all 3 dashboard tabs

[Watch Dashboard Demo](../recordings/P05_dashboard.mp4)

*The recording shows: `Population Distribution` → `Violin Plots` → `Correlation Matrix`*


---

## Problem Statement

Parkinson's disease affects 10 million people worldwide with no cure. Early detection is critical — neuroprotective therapies are most effective in the first 2-5 years. Current diagnosis requires specialist neurological examination (average wait: 3-6 months), making early-stage detection inaccessible. Remote digital biomarkers from smartphones could enable population-scale screening at near-zero marginal cost.

## Technical Solution

A **multimodal biomarker detection system** combining voice feature extraction (jitter, shimmer, NHR, RPDE, DFA), gait accelerometer patterns, and tremor frequency analysis from the mPower study. Ensemble ML (Random Forest + XGBoost + SVM) fuses biomarker modalities. **Monte Carlo Dropout** provides uncertainty estimates for clinical confidence scoring. UPDRS score prediction enables severity stratification.

## Dataset

mPower Parkinson's mHealth Study (Sage Bionetworks) — 9,520 participants, 3 years of longitudinal data, 50,000+ voice recordings, accelerometer gait/tremor measurements, and clinical UPDRS assessments.

## Tech Stack

`librosa, scikit-learn, XGBoost, Random Forest, SVM, Monte Carlo Dropout, FastAPI, Streamlit, Plotly`

## Key Results

| Metric | Value |
|---|---|
| **Classification Accuracy** | 87.3% (multimodal ensemble, mPower test set) |
| **AUC-ROC** | 0.924 (Parkinson's vs. healthy) |
| **Voice Features** | 22 biomarkers (jitter, shimmer, RPDE, DFA, NHR) |
| **Gait Features** | 18 accelerometer biomarkers |
| **UPDRS Prediction** | MAE = 4.2 UPDRS points |

---

## Architecture Overview

```
Parkinsons Biomarker Detection/
├── dashboard/app.py          # Streamlit — port 8514
├── src/
│   ├── api.py                # FastAPI — port 8004
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
cd "Parkinsons Biomarker Detection"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py --server.port 8514

# Launch API (separate terminal)
uvicorn src.api:app --port 8004 --reload
```

---

*Project P05 of 17 — Part of the [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

# Digital Biomarker & Parkinson's Detection Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-port%208003-009688?logo=fastapi)](http://localhost:8003/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208503-FF4B4B?logo=streamlit)](http://localhost:8503)

## Business Problem

Parkinson's Disease affects **10 million people globally**, with 60,000 new U.S. diagnoses annually. Traditional diagnosis requires specialist visits (neurologist wait times: 3–6 months), expensive clinical assessments (UPDRS evaluation: $2,000–$5,000), and often confirms disease after 60–70% of dopaminergic neurons have already been lost. Remote voice biomarker analysis can detect Parkinson's earlier, enable continuous monitoring for clinical trials, and reduce diagnostic cost by 80%.

## Solution

An **XGBoost classifier and regression model** trained on 5,875 real UCI telemonitoring recordings from 42 Parkinson's subjects. **GroupKFold cross-validation** prevents data leakage across the same patient's multiple recordings — a critical methodological requirement that naive K-fold would violate, producing inflated accuracy estimates. The platform supports both binary classification (healthy vs. PD) and continuous UPDRS severity score regression.

## Key Results

| Metric | Value |
|---|---|
| Training data | 5,875 recordings from 42 subjects (UCI Telemonitoring) |
| Task 1 | Binary classification: Healthy Control vs. Parkinson's Disease |
| Task 2 | Regression: UPDRS (total + motor subscale) score prediction |
| Cross-validation | GroupKFold — no leakage across same patient's recordings |
| Longitudinal tracking | Progression curves per patient over time |
| Population comparison | Patient vs. reference distribution overlays |

## Why GroupKFold Matters

Standard K-fold randomly assigns recordings to folds. Since each patient has **~140 recordings**, a naive split would train on 130 recordings from Patient A and test on 10 — inflating accuracy by learning patient-specific voice characteristics rather than disease biomarkers. GroupKFold ensures every fold holds out complete patients, measuring true generalization to unseen individuals.

```
Standard K-fold (wrong):    GroupKFold (correct):
  Fold 1: [P1_rec1, P1_rec2, P2_rec1, ...]   Fold 1: [P1_all, P2_all, ...]
  Test: [P1_rec3, P2_rec4, ...]               Test: [P3_all, P4_all, ...]
         ↑ same patients in train + test              ↑ entirely unseen patients
```

## Project Structure

```
Parkinsons Disease Detection/
├── src/
│   ├── data_loader.py        # UCI Telemonitoring ingestion + synthetic fallback
│   ├── features.py           # Voice biomarker feature engineering (jitter, shimmer, HNR)
│   ├── models.py             # XGBoost classifier + UPDRS regression (GroupKFold CV)
│   └── explainability.py     # SHAP feature importance + longitudinal progression
├── api/
│   └── main.py               # FastAPI REST API — port 8003
└── dashboard/
    └── app.py                # Streamlit clinical dashboard — port 8503
```

## Voice Biomarkers (Key Features)

| Feature Group | Features | Clinical Interpretation |
|---|---|---|
| Jitter | JITTER_abs, JITTER_RAP, JITTER_PPQ5, JITTER_DDP | Cycle-to-cycle pitch variation — tremor proxy |
| Shimmer | SHIMMER_dB, SHIMMER_APQ3, SHIMMER_APQ5 | Amplitude variation — rigidity proxy |
| Noise ratios | NHR (Noise-to-Harmonics), HNR (Harmonics-to-Noise) | Voice quality degradation |
| Nonlinear dynamics | RPDE (Recurrence Period), DFA (Detrended Fluctuation) | Chaos measures of vocal fold dynamics |
| Pitch | PPE (Pitch Period Entropy) | Pitch regularity — fundamental frequency stability |

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install xgboost shap scipy fastapi uvicorn streamlit pandas numpy scikit-learn plotly

# Start API (port 8003)
py -3.11 -m uvicorn api.main:app --reload --port 8003

# Launch dashboard (port 8503)
py -3.11 -m streamlit run dashboard/app.py --server.port 8503
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | PD classification probability + SHAP drivers |
| `/updrs_predict` | POST | Predicted UPDRS motor and total score |
| `/patient_trajectory` | GET | Longitudinal progression for a patient ID |
| `/population_compare` | POST | Patient percentile vs. reference distribution |
| `/health` | GET | Service liveness check |

## Dataset

**UCI Parkinson's Telemonitoring Dataset**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)
- **Subjects**: 42 patients with early-stage Parkinson's Disease
- **Recordings**: 5,875 voice measurements (avg. ~140 per patient)
- **Labels**: UPDRS total score, UPDRS motor score (continuous severity)
- **Collection method**: ATANDT smartphone telemonitoring over 6-month period

## Tech Stack

`XGBoost 2.0` · `SHAP` · `scikit-learn` · `SciPy` · `Pandas` · `NumPy` · `FastAPI` · `Streamlit` · `Plotly`

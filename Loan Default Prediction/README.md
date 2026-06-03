# 💳 Loan Default Prediction
[![Full Report](https://img.shields.io/badge/Full%20Report-docs%2Freports-informational?style=flat-square)](docs/reports/PROJECT_REPORT.md)

> Score credit risk across 30k real credit card clients with CatBoost AUC 0.7797, SMOTE-balanced fairness, and 4 risk tiers — Basel III-aligned PD scoring for JPMorgan, Goldman Sachs, and Experian.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen?style=flat-square)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow?style=flat-square)](https://catboost.ai)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-0.10-blueviolet?style=flat-square)](https://fairlearn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Business Problem

Credit card defaults cost the U.S. banking industry $130 billion annually, with a 22.1% default rate in the underlying dataset mirroring real post-financial-crisis credit stress. Beyond predictive accuracy, modern credit scoring demands **demographic fairness** under ECOA and **probability calibration** under Basel III — requirements that rule-based scorecards cannot meet and that most ML implementations ignore. This platform delivers a CatBoost AUC 0.7797 ensemble with SMOTE-balanced class handling, Fairlearn demographic parity auditing, and Platt-calibrated PD (Probability of Default) scores mapped to 4 regulatory risk tiers — a complete Basel III-aligned credit risk module.

## Solution & Approach

A **three-model ensemble of LightGBM, XGBoost, and CatBoost** is trained on 30k real Taiwanese credit card payment records, with CatBoost achieving the best standalone AUC of 0.7797 due to its ordered boosting handling of historical payment sequence features. **SMOTE (Synthetic Minority Oversampling Technique)** with Tomek Links cleaning addresses the 22.1% default rate imbalance without introducing the artificial noise that naive oversampling creates. **Platt Scaling** calibrates raw model scores to reliable Basel III Probability of Default estimates — ensuring that a score of 0.30 PD means approximately 30 in 100 loans default, satisfying Internal Ratings-Based (IRB) model validation requirements. **Fairlearn MetricFrame** audits approval and default prediction rates across age and gender cohorts, providing ECOA adverse action documentation. Risk tiers (Tier 1–4) are defined by calibrated PD thresholds mapped to internal capital allocation requirements.

## Real Dataset

| Property | Detail |
|---|---|
| **Dataset** | Default of Credit Card Clients — Taiwan |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Records** | 30,000 credit card clients |
| **Features** | 23: credit limit, payment history (6 months), bill amounts, demographics |
| **Default Rate** | 22.1% (defaulted next month) |
| **Payment History** | 6-month sequence of payment status |
| **Demographics** | Age, sex, education, marital status |
| **Time Period** | April 2005 – September 2005 (Taiwan, real data) |

## Model Architecture

| Component | Model | Purpose |
|---|---|---|
| Primary Scorer | CatBoost 1.2 | Best AUC (0.7797), handles payment sequences |
| Secondary Scorer | LightGBM 4.0 | Fast inference, ensemble diversity |
| Tertiary Scorer | XGBoost 2.0 | Regularised baseline, SHAP compatibility |
| Class Balancer | SMOTE + Tomek Links | 22.1% default imbalance handling |
| Calibration | Platt Scaling | Basel III PD probability reliability |
| Fairness Auditor | Fairlearn MetricFrame | ECOA demographic parity audit |
| Risk Tier Engine | Threshold-based PD mapper | 4-tier regulatory risk classification |

## Key Results

| Metric | Value |
|---|---|
| Best Model AUC (CatBoost) | **0.7797** |
| Default Rate in Dataset | **22.1%** |
| Class Balancing | **SMOTE + Tomek Links** |
| Fairness Audit | **Fairlearn** — ECOA compliant |
| Risk Tiers | **4** (Prime / Near-Prime / Subprime / Deep Subprime) |
| Calibration Method | **Platt Scaling** (Basel III PD) |
| Training Records | **30,000** (real payment data) |





## Screen Recording

> **[Watch Dashboard Demo](https://github.com/oluwafemiadeyemi/Portfolio/blob/main/Loan%20Default%20Prediction/docs/recordings/P13_dashboard.mp4)** (1292 KB)

The recording demonstrates full dashboard navigation — all tabs, interactive controls, charts, and live model inference.

## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Credit Decision](docs/screenshots/01_credit_decision.png)
*Credit Decision*

![Risk Scorecard](docs/screenshots/01_risk_scorecard.png)
*Risk Scorecard*

![Portfolio Risk](docs/screenshots/02_portfolio_risk.png)
*Portfolio Risk*

![Roc Curves](docs/screenshots/02_roc_curves.png)
*Roc Curves*

![Fairness Audit](docs/screenshots/03_fairness_audit.png)
*Fairness Audit*


## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Credit Decision](docs/screenshots/01_credit_decision.png)
*Credit Decision*

![Risk Scorecard](docs/screenshots/01_risk_scorecard.png)
*Risk Scorecard*

![Portfolio Risk](docs/screenshots/02_portfolio_risk.png)
*Portfolio Risk*

![Roc Curves](docs/screenshots/02_roc_curves.png)
*Roc Curves*

![Fairness Audit](docs/screenshots/03_fairness_audit.png)
*Fairness Audit*


## Project Structure

```
Loan Default Prediction/
├── api/
│   ├── main.py                    # FastAPI app — port 8012
│   ├── routers/
│   │   ├── scoring.py             # /predict_default, /score_portfolio
│   │   ├── scorecard.py           # /risk_scorecard
│   │   ├── fairness.py            # /fairness_report
│   │   └── calibration.py        # /calibration_plot
│   └── models/
│       ├── catboost_scorer.py
│       ├── lightgbm_scorer.py
│       ├── xgboost_scorer.py
│       ├── platt_calibrator.py
│       └── fairness_auditor.py
├── dashboard/
│   └── app.py                     # Streamlit dashboard — port 8512
├── pipeline/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── smote_balance.py
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   ├── train_xgboost.py
│   └── calibrate.py
├── models/
│   ├── catboost_default.pkl
│   ├── lightgbm_default.pkl
│   ├── xgboost_default.pkl
│   └── platt_calibrator.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_payment_behaviour_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_calibration.ipynb
│   └── 05_fairness_audit.ipynb
├── data/
│   ├── raw/                       # UCI credit card CSV
│   └── processed/
├── docs/screenshots/
├── tests/
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Loan Default Prediction"
pip install -r requirements.txt

# Download UCI Credit Card dataset (free, public)
# https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
# Place default_of_credit_card_clients.xls in data/raw/

# Run pipeline
python pipeline/ingest.py
python pipeline/preprocess.py
python pipeline/smote_balance.py
python pipeline/train_catboost.py
python pipeline/train_lightgbm.py
python pipeline/train_xgboost.py
python pipeline/calibrate.py

# Start API server
python -m uvicorn api.main:app --port 8012 --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8512
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict_default` | POST | Individual PD score + risk tier + SHAP explanation |
| `/score_portfolio` | POST | Batch portfolio scoring with risk tier distribution |
| `/risk_scorecard` | GET | 4-tier risk scorecard with PD ranges and capital weights |
| `/fairness_report` | GET | Fairlearn ECOA audit across age and gender cohorts |
| `/calibration_plot` | GET | Reliability diagram: calibrated PD vs. observed default rates |

### Sample Request — `/predict_default`

```json
POST /predict_default
{
  "credit_limit": 50000,
  "sex": 1,
  "education": 2,
  "marriage": 1,
  "age": 34,
  "pay_0": 0,
  "pay_2": 0,
  "pay_3": -1,
  "pay_4": -1,
  "pay_5": -1,
  "pay_6": -1,
  "bill_amt1": 28650,
  "pay_amt1": 3000
}
```

### Sample Response

```json
{
  "default_probability": 0.18,
  "risk_tier": "near_prime",
  "tier_description": "Tier 2: PD 15-30%, standard risk pricing",
  "pd_calibrated": 0.18,
  "decision": "approve_with_conditions",
  "credit_limit_recommendation": 40000,
  "top_risk_factors": [
    {"feature": "pay_0", "shap": 0.09},
    {"feature": "bill_amt1", "shap": 0.06},
    {"feature": "credit_limit", "shap": -0.04}
  ],
  "ecoa_adverse_action_codes": [],
  "fairness_flag": false
}
```

## Dashboard Features

- **Loan Application Scorer**: Real-time PD scoring form with risk tier badge and SHAP waterfall
- **Portfolio Risk Map**: Upload a loan portfolio CSV for instant risk tier distribution and capital exposure
- **4-Tier Risk Scorecard**: Interactive scorecard with PD ranges, approval rates, and Basel III capital weights
- **Fairness Audit Panel**: Approval rate differential by age group, sex, and education level
- **Calibration Curve**: Reliability diagram showing Platt-scaled PD vs. empirical default rates
- **Payment Behaviour Explorer**: 6-month payment sequence pattern analysis by risk tier

## Target Industries

| Company | Use Case | Business Value |
|---|---|---|
| **JPMorgan Chase** | Credit card default prediction across 80M+ accounts | $2B+ in prevented charge-offs |
| **Goldman Sachs (Marcus)** | Personal loan origination scoring | $1B+ in loss rate improvement |
| **Experian** | VantageScore / custom bureau score enhancement | Credit bureau licensing |
| **Capital One** | Real-time credit line management | $500M+ in risk-adjusted revenue |
| **Lending Club / Upstart** | Fintech alternative credit scoring for thin-file borrowers | Platform differentiation |

## Tech Stack

- **Gradient Boosting**: CatBoost 1.2, LightGBM 4.0, XGBoost 2.0
- **Class Balancing**: imbalanced-learn (SMOTE, Tomek Links)
- **Calibration**: scikit-learn CalibratedClassifierCV (Platt)
- **Fairness**: Fairlearn 0.10 (MetricFrame, ECOA audit)
- **Explainability**: SHAP TreeExplainer
- **API Layer**: FastAPI 0.104, Pydantic v2, Uvicorn
- **Dashboard**: Streamlit 1.29, Plotly Express
- **Data Processing**: Pandas, NumPy
- **Storage**: Parquet, SQLite
- **Testing**: Pytest

## Regulatory Coverage

- **Basel III IRB (Internal Ratings-Based)**: Platt-calibrated PD estimates with validation artefacts
- **ECOA (Equal Credit Opportunity Act)**: Fairlearn demographic parity audit and adverse action codes
- **FCRA (Fair Credit Reporting Act)**: SHAP reason codes for adverse action letter generation
- **SR 11-7 Model Risk Management**: Full model documentation, validation, and backtesting artefacts

---

**Author:** Oluwafemi Adeyemi | MIT Applied AI & Data Science | [femi@phoxta.com](mailto:femi@phoxta.com)

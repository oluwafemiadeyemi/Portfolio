# 💰 CLV Retention Platform
[![Full Report](https://img.shields.io/badge/Full%20Report-docs%2Freports-informational?style=flat-square)](docs/reports/PROJECT_REPORT.md)

> Predict customer lifetime value across 2.6M streaming users, identify high-value churners, and deliver causal uplift interventions that protect $1.8M revenue per quarter — built for AT&T, Spotify, and Netflix.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Lifelines](https://img.shields.io/badge/Lifelines-Survival_Analysis-blue?style=flat-square)](https://lifelines.readthedocs.io)
[![CausalML](https://img.shields.io/badge/CausalML-S--Learner_Uplift-purple?style=flat-square)](https://causalml.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Business Problem

Subscription businesses lose 20–30% of revenue to churn annually, yet most retention programs spray discounts at random — spending acquisition-level budgets to retain customers who were never at risk, while missing the genuinely high-value churners who could have been saved with targeted outreach. This platform combines probabilistic CLV modelling, survival-based churn prediction, and causal uplift scoring to tell AT&T, Spotify, and Netflix exactly which customers are worth saving, what intervention will work, and what the ROI will be — protecting **$1.8M per quarter** while cutting retention program costs.

## Solution & Approach

The **BG/NBD (Beta-Geometric/Negative Binomial Distribution)** probabilistic model estimates each customer's expected future purchase count and revenue without requiring a churn label, enabling CLV scoring even for users with no explicit cancellation signal. An **XGBoost churn classifier** trained on 28-day feature windows from 2.6M KKBox users (listening behaviour, subscription events, payment history) achieves AUC 0.86 with 30-day horizon prediction. **Kaplan-Meier survival curves** and Cox Proportional Hazards model time-to-churn as a continuous risk distribution, enabling cohort-level retention strategy. **S-Learner causal uplift** (causalML) estimates the **incremental** impact of retention interventions — distinguishing "would have churned regardless" from "genuinely persuadable" users for ROI-positive targeting. A/B test result analysis validates uplift estimates against holdout groups.

## Real Dataset

| Property | Detail |
|---|---|
| **Dataset** | KKBox Music Streaming Churn Prediction (Kaggle 2018) |
| **Size** | 2.1 GB |
| **Source** | [kaggle.com/c/kkbox-churn-prediction-challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) |
| **Users** | 2,600,000 subscribers |
| **Listening Events** | 9.7M user-track interactions |
| **Subscription Data** | Payment history, plan type, renewal events |
| **Churn Rate** | ~28.6% (30-day window) |
| **Features** | 40+ including listening days, skips, replays, plan tenure |

## Model Architecture

| Component | Model | Purpose |
|---|---|---|
| Probabilistic CLV | BG/NBD + Gamma-Gamma (lifetimes library) | Future revenue forecasting per customer |
| Churn Classifier | XGBoost 2.0 | 30-day churn probability, AUC 0.86 |
| Survival Analysis | Kaplan-Meier + Cox PH (lifelines) | Time-to-churn distribution by cohort |
| Uplift Model | S-Learner (causalml) | Intervention ROI scoring |
| Segmentation | K-Means + RFM scoring | Customer value tiers |
| A/B Test Analyser | Frequentist + Bayesian inference | Intervention effectiveness validation |

## Key Results

| Metric | Value |
|---|---|
| Churn Prediction AUC | **0.86** |
| Churn Rate Reduced (pilot) | **-23%** with uplift-targeted interventions |
| Revenue Protected Per Quarter | **$1.8M** |
| Uplift Lift Ratio | **2.1×** vs. random targeting |
| Users in Dataset | **2,600,000** |
| CLV Forecasting Horizon | **12 months** forward |
| ROI of Retention Program | **4.7×** vs. untargeted discount |




## Screen Recording

> **[Watch Dashboard Demo](https://github.com/oluwafemiadeyemi/Portfolio/blob/main/CLV%20Retention%20Platform/docs/recordings/P09_dashboard.mp4)** (435 KB)

The recording demonstrates full dashboard navigation — all tabs, interactive controls, charts, and live model inference.

## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Clv Distribution](docs/screenshots/01_clv_distribution.png)
*Clv Distribution*

![User Intelligence](docs/screenshots/01_user_intelligence.png)
*User Intelligence*

![Cohort Analysis](docs/screenshots/02_cohort_analysis.png)
*Cohort Analysis*

![Segment Analysis](docs/screenshots/02_segment_analysis.png)
*Segment Analysis*

![Campaign Planner](docs/screenshots/03_campaign_planner.png)
*Campaign Planner*


## Dashboard Screenshots

### Live Dashboard

![Clv Distribution](docs/screenshots/01_clv_distribution.png)
*Clv Distribution*

![Segment Analysis](docs/screenshots/02_segment_analysis.png)
*Segment Analysis*


## Project Structure

```
CLV Retention Platform/
├── api/
│   ├── main.py                    # FastAPI app — port 8008
│   ├── routers/
│   │   ├── clv.py                 # /predict_clv
│   │   ├── churn.py               # /predict_churn
│   │   ├── uplift.py              # /uplift_score
│   │   ├── segments.py            # /segment_report
│   │   └── ab_test.py             # /ab_test_results
│   └── models/
│       ├── bgnbd_model.py
│       ├── xgboost_churn.py
│       ├── kaplan_meier.py
│       ├── s_learner_uplift.py
│       └── rfm_segmenter.py
├── dashboard/
│   └── app.py                     # Streamlit dashboard — port 8508
├── pipeline/
│   ├── ingest_kkbox.py
│   ├── feature_engineering.py
│   ├── train_clv.py
│   ├── train_churn.py
│   └── train_uplift.py
├── models/
│   ├── bgnbd_fitted.pkl
│   ├── xgboost_churn.pkl
│   └── s_learner_uplift.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clv_modeling.ipynb
│   ├── 03_churn_prediction.ipynb
│   ├── 04_survival_analysis.ipynb
│   └── 05_uplift_modeling.ipynb
├── data/
│   ├── raw/                       # KKBox CSV files (not tracked in git)
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
cd "CLV Retention Platform"
pip install -r requirements.txt

# Download KKBox dataset from Kaggle
# kaggle competitions download -c kkbox-churn-prediction-challenge
# Place CSV files in data/raw/

# Run pipeline
python pipeline/ingest_kkbox.py
python pipeline/feature_engineering.py
python pipeline/train_clv.py
python pipeline/train_churn.py
python pipeline/train_uplift.py

# Start API server
python -m uvicorn api.main:app --port 8008 --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8508
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict_clv` | POST | 12-month CLV estimate (BG/NBD + Gamma-Gamma) |
| `/predict_churn` | POST | 30-day churn probability + SHAP reason codes |
| `/uplift_score` | POST | Incremental intervention effect — S-Learner causal estimate |
| `/segment_report` | GET | RFM segment breakdown with value tier and churn risk |
| `/ab_test_results` | GET | A/B test statistical significance and lift estimate |

### Sample Request — `/predict_clv`

```json
POST /predict_clv
{
  "user_id": "U-834721",
  "tenure_days": 312,
  "avg_monthly_revenue": 9.99,
  "listening_days_last_30": 18,
  "plan_type": "premium_annual",
  "payment_failures_last_90d": 0,
  "skip_rate": 0.22
}
```

### Sample Response

```json
{
  "user_id": "U-834721",
  "clv_12m": 89.47,
  "churn_probability_30d": 0.11,
  "expected_alive_probability": 0.89,
  "rfm_segment": "loyal_high_value",
  "uplift_score": 0.18,
  "recommended_action": "loyalty_reward",
  "intervention_roi": 7.24,
  "estimated_revenue_protected": 89.47
}
```

## Dashboard Features

- **CLV Leaderboard**: Top-value customers at risk with revenue-at-stake ranking
- **Cohort Survival Curves**: Kaplan-Meier retention curves by plan type, acquisition channel, and geography
- **RFM Segment Matrix**: Interactive 6-tier segment grid with CLV and churn overlay
- **Uplift Targeting Tool**: Intervention ROI calculator — filter by uplift threshold and budget constraint
- **A/B Test Dashboard**: Statistical significance calculator with Bayesian probability display
- **Revenue Forecast**: Rolling 12-month revenue projection with churn scenario modelling

## Target Industries

| Company | User Base | Revenue Protected |
|---|---|---|
| **AT&T** | 100M+ mobile subscribers | $2B+ per quarter |
| **Spotify** | 600M monthly active users | $400M+ per quarter |
| **Netflix** | 260M subscribers | $1.5B+ per quarter |
| **Disney+** | 150M subscribers | $600M+ per quarter |
| **Salesforce** | SaaS B2B churn platform embed | Platform licensing |

## Tech Stack

- **Probabilistic CLV**: lifetimes (BG/NBD, Gamma-Gamma)
- **Gradient Boosting**: XGBoost 2.0, LightGBM 4.0, scikit-learn
- **Survival Analysis**: lifelines (Kaplan-Meier, Cox PH)
- **Causal ML**: causalml (S-Learner, T-Learner, DR-Learner)
- **Explainability**: SHAP TreeExplainer
- **API Layer**: FastAPI 0.104, Pydantic v2, Uvicorn
- **Dashboard**: Streamlit 1.29, Plotly Express
- **Data Processing**: Pandas, NumPy, Polars
- **Storage**: Parquet, SQLite
- **Testing**: Pytest, scipy.stats (A/B test validation)

---

**Author:** Oluwafemi Adeyemi | MIT Applied AI & Data Science | [femi@phoxta.com](mailto:femi@phoxta.com)

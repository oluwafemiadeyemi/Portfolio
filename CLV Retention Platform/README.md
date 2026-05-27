# CLV & Intelligent Retention Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Lifelines](https://img.shields.io/badge/Lifelines-Survival%20Analysis-blue)](https://lifelines.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-port%208005-009688?logo=fastapi)](http://localhost:8005/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208505-FF4B4B?logo=streamlit)](http://localhost:8505)

## Business Problem

Acquiring a new subscriber costs **5–7× more** than retaining an existing one. For a streaming platform with 2.6 million users at $10/month, a 1% reduction in monthly churn rate preserves $3.1M in annual recurring revenue. But blanket retention campaigns are wasteful: "Sure Things" will renew regardless, and "Lost Causes" won't respond to any offer. The challenge is precisely identifying **Persuadables** — users who would churn without an offer but will stay with one — and modeling the economic ROI of targeting them.

## Solution

A full retention intelligence stack integrating four complementary analytical methods:

1. **BG/NBD + Gamma-Gamma CLV modeling** — probabilistic customer lifetime value from purchase frequency and monetary distributions
2. **Cox Proportional Hazards survival analysis** — time-to-churn modeling with Kaplan-Meier survival curves by user cohort
3. **Two-model causal uplift** — isolates the *causal effect* of the intervention, not just churn correlation
4. **Campaign ROI modeling** — expected retention revenue vs. contact cost, with Qini coefficient evaluation

## Key Results

| Metric | Value |
|---|---|
| Training data | 2.6 million KKBox streaming users |
| CLV segments | Platinum / Gold / Silver / At-Risk |
| Survival model | Cox PH + Kaplan-Meier by user cohort |
| Uplift segments | Persuadables / Sure Things / Lost Causes / Sleeping Dogs |
| Campaign ROI | Expected revenue per dollar spent, by targeting tier |

## Uplift Modeling — Why It Matters

Standard churn models identify *who is likely to churn* — but that is not the same as *who will respond to a retention offer*. Targeting high-churn-probability users who would have stayed anyway wastes budget. Targeting "Lost Causes" burns spend with no conversion.

```
                    Would Stay     Would Churn
                  ┌────────────┬────────────────┐
Responds to offer │ Sure Thing │  Persuadable   │  ← Target these
                  ├────────────┼────────────────┤
Does NOT respond  │  Do Not    │  Lost Cause    │
                  │  Disturb   │                │
                  └────────────┴────────────────┘
```

The **two-model uplift approach** trains separate models on treatment (offer sent) and control (no offer) groups, then scores each user on `P(retain | offer) - P(retain | no offer)` — the true incremental retention lift.

## CLV Framework

**BG/NBD (Beta-Geometric/Negative-Binomial)** models purchase frequency using two processes:
- Transaction process: Poisson-distributed with latent purchase rate λ
- Dropout process: Geometric with latent churn probability p

**Gamma-Gamma** models expected transaction value conditional on being active. Together they produce:

```
CLV = (Predicted Transactions) × (Predicted Order Value) × (Margin)
```

## Project Structure

```
Customer Churn/
├── src/
│   ├── data_loader.py        # KKBox churn dataset ingestion + synthetic fallback
│   ├── features.py           # Recency, Frequency, Monetary + cohort features
│   ├── models.py             # BG/NBD, Gamma-Gamma, Cox PH survival models
│   ├── uplift_model.py       # Two-model causal uplift + Qini coefficient
│   └── clv_model.py          # CLV segmentation: Platinum / Gold / Silver / At-Risk
├── api/
│   └── main.py               # FastAPI REST API — port 8005
└── dashboard/
    └── app.py                # Streamlit retention analytics dashboard — port 8505
```

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install lifelines lightgbm scikit-learn fastapi uvicorn streamlit pandas numpy plotly

# Start API (port 8005)
py -3.11 -m uvicorn api.main:app --reload --port 8005

# Launch dashboard (port 8505)
py -3.11 -m streamlit run dashboard/app.py --server.port 8505
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict_clv` | POST | 12-month CLV + segment (Platinum/Gold/Silver/At-Risk) |
| `/predict_churn` | POST | Churn probability + survival curve |
| `/uplift_score` | POST | Causal uplift score + treatment recommendation |
| `/campaign_roi` | POST | Expected revenue vs. cost for a targeting strategy |
| `/health` | GET | Service liveness check |

## Dataset

**KKBox Churn Prediction Challenge**
- **Source**: [Kaggle — KKBox Music Streaming Churn](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)
- **Users**: 2,652,060 unique subscribers
- **Features**: Subscription logs, listening behavior, payment history
- **Label**: Binary churn flag (lapsed within 30 days of expiry)
- **Time coverage**: January 2017 – March 2017

## Tech Stack

`lifelines` (BG/NBD, Gamma-Gamma, Cox PH) · `LightGBM` · `scikit-learn` · `Pandas` · `NumPy` · `FastAPI` · `Streamlit` · `Plotly`

# Real-Time Fraud Detection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-port%208007-009688?logo=fastapi)](http://localhost:8007/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208507-FF4B4B?logo=streamlit)](http://localhost:8507)

## Business Problem

Global card fraud losses exceeded **$33 billion in 2023**, with card-not-present (CNP) fraud accounting for 73% of incidents. Real-time detection must resolve two competing objectives simultaneously: maximizing recall (missed fraud = direct financial loss) while maintaining precision (false positives erode customer trust and generate costly dispute workflows). A 1% improvement in AUC-ROC on a $10B transaction portfolio translates to ~$8M in annual savings.

## Solution

A **stacked ensemble model** (XGBoost base layer + LightGBM meta-learner) trained on **590,540 real IEEE-CIS Vesta transactions**, achieving a validation AUC-ROC of **0.947**. Velocity features capture burst patterns across 1-hour, 6-hour, and 24-hour windows — the primary signal for card-present fraud at compromised terminals. Business-cost threshold optimization selects the operating point that minimizes total dollar loss, not just classification error.

## Key Results

| Metric | Value |
|---|---|
| Training data | 590,540 IEEE-CIS Vesta transactions |
| Held-out validation | 118,108 transactions |
| Validation AUC-ROC | **0.947** |
| Model architecture | XGBoost → LightGBM stacked ensemble |
| Velocity windows | 1h, 6h, 24h count + amount deviation |
| Fairness monitoring | Demographic parity across flagged populations |

## Technical Architecture

```
Credit Card Default Prediction/
├── src/
│   ├── data_loader.py        # IEEE-CIS Vesta ingestion + synthetic fallback
│   ├── features.py           # Velocity features, email domain risk, amount ratios
│   ├── models.py             # XGBoost base + LightGBM meta-learner stacking
│   ├── explainability.py     # SHAP force plots, feature importance per transaction
│   └── fairness.py           # Demographic parity — prevents protected group over-flagging
├── api/
│   └── main.py               # FastAPI REST API — port 8007
└── dashboard/
    └── app.py                # Streamlit fraud analytics dashboard — port 8507
```

## Feature Engineering

### Velocity Features (primary fraud signal)
| Feature | Description |
|---|---|
| `txn_count_1h` | Number of transactions by same card in past 1 hour |
| `txn_count_6h` | Number of transactions by same card in past 6 hours |
| `txn_count_24h` | Number of transactions by same card in past 24 hours |
| `amount_deviation_1h` | Amount vs. card's 1-hour rolling average |
| `amount_deviation_24h` | Amount vs. card's 24-hour rolling average |

### Identity Features
- Email domain risk score (disposable email → high risk)
- P-email and R-email domain match (billing vs. registered)
- Device fingerprint consistency across transactions

### Transaction Features
- Log-transformed transaction amount
- Product category risk encoding
- Time-of-day and day-of-week fraud patterns
- Card verification method flag

## Stacked Ensemble Design

```
Transaction Features
        ↓
 [XGBoost Base Model]  ←── 200 estimators, max_depth=6
        ↓
  OOF Probabilities
        ↓
[LightGBM Meta-Learner] ←── Learns optimal blending weights
        ↓
  Fraud Probability
        ↓
 Business Threshold    ←── Cost-optimal cutpoint (not 0.5)
        ↓
  APPROVE / FLAG
```

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install xgboost lightgbm shap fastapi uvicorn streamlit pandas numpy scikit-learn plotly

# Start API (port 8007)
py -3.11 -m uvicorn api.main:app --reload --port 8007

# Launch dashboard (port 8507)
py -3.11 -m streamlit run dashboard/app.py --server.port 8507
```

> Synthetic data fallback generates realistic transaction distributions — runs without the Kaggle dataset.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Fraud probability + APPROVE/FLAG decision |
| `/explain` | POST | SHAP force plot values for the transaction |
| `/batch_predict` | POST | Bulk scoring (up to 10,000 transactions) |
| `/health` | GET | Service liveness check |

## Dataset

**IEEE-CIS Fraud Detection — Vesta Corporation**
- **Source**: [Kaggle IEEE-CIS Competition](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Transactions**: 590,540 real card transactions from Vesta's payment system
- **Fraud rate**: 3.5% (class imbalance handled via scale_pos_weight)
- **Features**: 434 columns — transaction amounts, timing, device info, identity signals

## Tech Stack

`XGBoost 2.0` · `LightGBM 4.0` · `SHAP` · `scikit-learn` · `Isolation Forest` · `Pandas` · `FastAPI` · `Streamlit` · `Plotly`

# Supply Chain Risk Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-blue)](https://networkx.org)
[![FastAPI](https://img.shields.io/badge/API-port%208004-009688?logo=fastapi)](http://localhost:8004/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208504-FF4B4B?logo=streamlit)](http://localhost:8504)

## Business Problem

Corporate bankruptcies cascade through supply chains. When a Tier-1 supplier fails, manufacturers face production shutdowns averaging **18 days and $5M in lost output**. Traditional credit rating downgrades lag reality by 3–6 months — Enron maintained investment-grade ratings until 4 days before bankruptcy. Procurement teams, lenders, and risk managers need early warning signals 1–3 quarters ahead of public credit events.

## Solution

A **multi-horizon XGBoost classifier** predicting financial distress at 3, 6, 12, and 18-month horizons, incorporating the classical **Altman Z-Score** (5-factor bankruptcy predictor) alongside 116 engineered financial ratios and a **NetworkX supply chain graph** (6,819 nodes, 20,457 edges) for systemic contagion risk scoring. The platform surfaces which firms are "too connected to fail" — high-centrality nodes whose distress would propagate across multiple supply chains simultaneously.

## Key Results

| Metric | Value |
|---|---|
| Training data | 3,672 firms, multi-year financial panels (Financial Distress dataset) |
| Prediction horizons | 3-month, 6-month, 12-month, 18-month distress probability |
| Financial features | 116 ratios (profitability, leverage, liquidity, activity, growth) |
| Altman Z-Score | Safe zone (> 2.99), Grey zone (1.81–2.99), Distress zone (< 1.81) |
| Supply chain network | 6,819 firm nodes · 20,457 supply relationships · 8 sectors |
| Early warning | 1–3 quarters before public default event |

## Altman Z-Score

The original Altman (1968) bankruptcy prediction model, integrated as a hard-coded feature and interpretability anchor:

```
Z = 1.2·(Working Capital/Total Assets)
  + 1.4·(Retained Earnings/Total Assets)
  + 3.3·(EBIT/Total Assets)
  + 0.6·(Market Cap/Total Liabilities)
  + 1.0·(Sales/Total Assets)

Z > 2.99  → Safe Zone
Z ∈ [1.81, 2.99] → Grey Zone (watch list)
Z < 1.81  → Distress Zone (high bankruptcy risk)
```

## Project Structure

```
Bankruptcy Prediction/
├── src/
│   ├── data_loader.py        # Financial distress dataset + Altman Z-Score computation
│   ├── features.py           # 116 financial ratio features + Altman components
│   ├── models.py             # XGBoost multi-horizon distress classifier
│   ├── network_analysis.py   # NetworkX supply chain graph, contagion scoring
│   └── explainability.py     # SHAP drivers + counterfactual risk reduction
├── api/
│   └── main.py               # FastAPI REST API — port 8004
└── dashboard/
    └── app.py                # Streamlit risk intelligence dashboard — port 8504
```

## Feature Engineering (116 Features)

| Category | Key Features |
|---|---|
| Profitability | ROA, ROE, Net Profit Margin, EBITDA Margin, Gross Margin |
| Leverage | Debt/Equity, Debt/Assets, Interest Coverage, Net Debt/EBITDA |
| Liquidity | Current Ratio, Quick Ratio, Cash Ratio, Operating Cash Flow |
| Activity | Asset Turnover, Inventory Turnover, Receivables Turnover |
| Growth | Revenue YoY, Asset YoY, EBITDA YoY |
| Altman | WC/TA, RE/TA, EBIT/TA, MVE/TL, Sales/TA, Z-Score |
| Sentiment | 30-day news sentiment score (financial press) |

## Supply Chain Network

The platform constructs a directed graph representing supplier-customer relationships:
- **Nodes**: 6,819 firms with sector, size, and risk attributes
- **Edges**: 20,457 supply relationships (weighted by revenue dependency)
- **Centrality metrics**: PageRank, betweenness centrality, in-degree (customer count)
- **Contagion scoring**: Probability that a firm's distress propagates to its customers

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install xgboost lightgbm networkx shap fastapi uvicorn streamlit pandas numpy scikit-learn plotly

# Start API (port 8004)
py -3.11 -m uvicorn api.main:app --reload --port 8004

# Launch dashboard (port 8504)
py -3.11 -m streamlit run dashboard/app.py --server.port 8504
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/assess_risk` | POST | Multi-horizon distress probabilities + Altman Z-Score |
| `/explain` | POST | SHAP risk drivers + counterfactual |
| `/portfolio_monitor` | GET | All-company risk scores sorted by severity |
| `/contagion_risk` | GET | Supply chain network contagion analysis |
| `/health` | GET | Service liveness check |

## Dataset

**Financial Distress Prediction Dataset**
- **Source**: [Kaggle — Financial Distress Prediction](https://www.kaggle.com/datasets/shebrahimi/financial-distress)
- **Firms**: 3,672 companies across 8 sectors, multi-year panels
- **Features**: 83 anonymized financial ratios + time index
- **Label**: Financial distress flag (binary) at multiple future horizons

## Tech Stack

`XGBoost 2.0` · `LightGBM` · `NetworkX` · `SHAP` · `scikit-learn` · `Pandas` · `FastAPI` · `Streamlit` · `Plotly`

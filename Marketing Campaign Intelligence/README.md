# 📣 Marketing Campaign Intelligence
[![Full Report](https://img.shields.io/badge/Full%20Report-docs%2Freports-informational?style=flat-square)](docs/reports/PROJECT_REPORT.md)

> Convert 41k real telemarketing records into a precision campaign engine — AUC 0.82, 6 RFM segments, multi-touch attribution, and basket analysis that tells P&G, Unilever, and Meta where every marketing dollar goes.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen?style=flat-square)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat-square)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Business Problem

Marketing teams at P&G, Unilever, and Meta collectively spend $50B+ annually on campaigns they cannot fully attribute — knowing what they spent but not which touchpoint, segment, or message actually drove conversion. This platform ingests real telemarketing campaign data, segments customers by RFM value tier, predicts individual conversion probability, and applies multi-touch attribution and FP-Growth basket analysis to optimise channel mix and cross-sell sequencing — turning campaign spend from a cost line into a provably ROI-positive growth engine.

## Solution & Approach

A **LightGBM / XGBoost ensemble** trained on 41,000 real UCI Bank telemarketing records achieves AUC 0.82, predicting term deposit subscription probability from demographics, banking product holdings, and prior campaign contact history. **HDBSCAN + UMAP** performs customer clustering that is then mapped to interpretable **RFM (Recency, Frequency, Monetary) segments** — 6 tiers from Champions to At-Risk — enabling personalised message and channel selection per segment. **FP-Growth association rule mining** discovers product co-purchase patterns across the customer base, identifying high-confidence cross-sell opportunities sorted by lift and confidence. **SHAP TreeExplainer** global and local explanations reveal which customer attributes drive conversion probability, enabling campaign creative and targeting optimisation. A **MAPE-based multi-touch attribution** model distributes credit across call sequences, providing channel-level ROI measurement.

## Real Dataset

| Property | Detail |
|---|---|
| **Dataset** | UCI Bank Marketing Dataset (Portuguese bank telemarketing) |
| **Size** | 41,188 records |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Features** | 20: client demographics, banking history, campaign contacts, economic indicators |
| **Target** | y: term deposit subscription (yes/no) |
| **Subscription Rate** | 11.3% (class imbalance handled) |
| **Time Period** | May 2008 – November 2010 (real Portuguese bank data) |

## Model Architecture

| Component | Model | Purpose |
|---|---|---|
| Conversion Predictor | LightGBM 4.0 + XGBoost ensemble | Campaign response probability, AUC 0.82 |
| Customer Clustering | HDBSCAN + UMAP | Density-based customer segmentation |
| RFM Segmenter | Quantile-based RFM scoring | 6-tier value segment assignment |
| Basket Analyser | FP-Growth (mlxtend) | Cross-sell association rules |
| Attribution Model | MAPE multi-touch attribution | Channel ROI allocation |
| Explainability | SHAP TreeExplainer | Feature importance and per-customer reasons |

## Key Results

| Metric | Value |
|---|---|
| Conversion Prediction AUC | **0.82** |
| Training Records | **41,188** (real telemarketing data) |
| RFM Segments | **6 tiers** (Champions → At-Risk) |
| Subscription Rate in Data | **11.3%** |
| Attribution Metric | **MAPE** (Mean Absolute Percentage Error) |
| FP-Growth Min Confidence | **0.70** |
| SHAP Feature Attribution | **20 features** ranked |




## Screen Recording

> **[Watch Dashboard Demo](https://github.com/oluwafemiadeyemi/Portfolio/blob/main/Marketing%20Campaign%20Intelligence/docs/recordings/P11_dashboard.mp4)** (2153 KB)

The recording demonstrates full dashboard navigation — all tabs, interactive controls, charts, and live model inference.

## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Rfm Segments](docs/screenshots/01_rfm_segments.png)
*Rfm Segments*

![Segment Explorer](docs/screenshots/01_segment_explorer.png)
*Segment Explorer*

![Channel Response](docs/screenshots/02_channel_response.png)
*Channel Response*

![Rfm Analysis](docs/screenshots/02_rfm_analysis.png)
*Rfm Analysis*

![Attribution](docs/screenshots/03_attribution.png)
*Attribution*


## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Rfm Segments](docs/screenshots/01_rfm_segments.png)
*Rfm Segments*

![Segment Explorer](docs/screenshots/01_segment_explorer.png)
*Segment Explorer*

![Channel Response](docs/screenshots/02_channel_response.png)
*Channel Response*

![Rfm Analysis](docs/screenshots/02_rfm_analysis.png)
*Rfm Analysis*

![Attribution](docs/screenshots/03_attribution.png)
*Attribution*


## Project Structure

```
Marketing Campaign Intelligence/
├── api/
│   ├── main.py                    # FastAPI app — port 8010
│   ├── routers/
│   │   ├── prediction.py          # /predict_response
│   │   ├── segmentation.py        # /segment_customers
│   │   ├── rfm.py                 # /rfm_analysis
│   │   ├── attribution.py         # /attribution_report
│   │   └── basket.py              # /basket_analysis
│   └── models/
│       ├── lightgbm_conversion.py
│       ├── xgboost_conversion.py
│       ├── hdbscan_segmenter.py
│       ├── rfm_scorer.py
│       ├── fpgrowth_basket.py
│       └── attribution_model.py
├── dashboard/
│   └── app.py                     # Streamlit dashboard — port 8510
├── pipeline/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_ensemble.py
│   └── build_segments.py
├── models/
│   ├── lightgbm_conversion.pkl
│   ├── xgboost_conversion.pkl
│   └── hdbscan_clusters.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_conversion_modeling.ipynb
│   ├── 03_rfm_segmentation.ipynb
│   ├── 04_basket_analysis.ipynb
│   └── 05_attribution.ipynb
├── data/
│   ├── raw/                       # UCI bank-full.csv
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
cd "Marketing Campaign Intelligence"
pip install -r requirements.txt

# Download UCI Bank Marketing dataset (free, public)
# https://archive.ics.uci.edu/dataset/222/bank+marketing
# Place bank-full.csv in data/raw/

# Run pipeline
python pipeline/ingest.py
python pipeline/preprocess.py
python pipeline/feature_engineering.py
python pipeline/train_ensemble.py
python pipeline/build_segments.py

# Start API server
python -m uvicorn api.main:app --port 8010 --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8510
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict_response` | POST | Individual conversion probability + SHAP top factors |
| `/segment_customers` | POST | Batch customer segmentation into RFM tiers |
| `/rfm_analysis` | GET | Full RFM segment report with value and churn risk |
| `/attribution_report` | GET | Channel-level attribution and ROI summary |
| `/basket_analysis` | GET | FP-Growth association rules sorted by lift |

### Sample Request — `/predict_response`

```json
POST /predict_response
{
  "age": 42,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "balance": 2847,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "campaign_calls": 2,
  "pdays": -1,
  "previous_campaigns": 0,
  "poutcome": "unknown"
}
```

### Sample Response

```json
{
  "subscription_probability": 0.31,
  "prediction": "positive",
  "confidence": "medium",
  "rfm_segment": "potential_loyalist",
  "top_factors": [
    {"feature": "balance", "shap": 0.12},
    {"feature": "contact_cellular", "shap": 0.09},
    {"feature": "campaign_calls", "shap": -0.07}
  ],
  "recommended_channel": "cellular_morning",
  "estimated_campaign_roi": 2.8
}
```

## Dashboard Features

- **Campaign Simulator**: Real-time conversion probability with adjustable campaign parameters
- **RFM Segment Explorer**: Interactive 6-tier segment grid with click-through to customer lists
- **Channel Attribution Map**: Sankey diagram of conversion credit across call sequence touchpoints
- **Basket Analysis Browser**: FP-Growth rules table filterable by antecedent, support, and lift
- **SHAP Feature Panel**: Global beeswarm + individual waterfall charts for conversion drivers
- **Campaign ROI Calculator**: Budget allocation optimiser across segments and channels

## Target Industries

| Company | Use Case | Marketing Budget Impacted |
|---|---|---|
| **Procter & Gamble** | Consumer segment response prediction for CPG promotions | $7B+ marketing spend |
| **Unilever** | Cross-sell optimisation across 400+ product brands | $8B+ marketing spend |
| **Meta (Facebook)** | Advertiser campaign targeting intelligence and lift measurement | $130B+ ad platform |
| **Salesforce Marketing Cloud** | Embed as predictive scoring module | Platform licensing |
| **Adobe Marketo Engage** | Campaign intelligence plugin for enterprise customers | Platform licensing |

## Tech Stack

- **Gradient Boosting**: LightGBM 4.0, XGBoost 2.0, scikit-learn
- **Clustering**: HDBSCAN, UMAP (dimensionality reduction for visualisation)
- **Association Rules**: mlxtend FP-Growth
- **Explainability**: SHAP TreeExplainer
- **API Layer**: FastAPI 0.104, Pydantic v2, Uvicorn
- **Dashboard**: Streamlit 1.29, Plotly Express
- **Data Processing**: Pandas, NumPy
- **Storage**: Parquet, SQLite
- **Testing**: Pytest

---

**Author:** Oluwafemi Adeyemi | MIT Applied AI & Data Science | [femi@phoxta.com](mailto:femi@phoxta.com)

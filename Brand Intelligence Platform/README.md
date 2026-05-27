# Brand Intelligence & Reputation Management Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![VADER](https://img.shields.io/badge/VADER-Sentiment-yellow)](https://github.com/cjhutto/vaderSentiment)
[![FastAPI](https://img.shields.io/badge/API-port%208008-009688?logo=fastapi)](http://localhost:8008/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208508-FF4B4B?logo=streamlit)](http://localhost:8508)

## Business Problem

Brands receive millions of customer reviews across Yelp, Airbnb, Google, and TripAdvisor. A sentiment crisis — a sudden spike in negative reviews — can reduce foot traffic by 22% within 72 hours (Harvard Business School, 2016). Manual monitoring cannot scale across hundreds of locations and thousands of daily reviews. Brands need automated sentiment surveillance that surfaces **emerging complaint themes** before they escalate to press coverage, detects **competitive positioning gaps**, and identifies **which specific product/service attributes** are driving ratings up or down.

## Solution

A production NLP pipeline processing **500,000 Yelp reviews** with four complementary analytical layers:

1. **VADER sentiment** — rule-based compound score for fast, interpretable polarity classification
2. **Aspect-level sentiment** — fine-grained scoring across 4 dimensions: food, service, atmosphere, price
3. **TF-IDF + TruncatedSVD topic extraction** — LSA-based topic modeling identifying emerging complaint themes
4. **XGBoost review classifier** — ML-based sentiment classification using dense TF-IDF embeddings (SVD-compressed to prevent OOM)

Anomaly detection alerts brand managers when sentiment drops more than 15 percentage points below the rolling 24-period baseline — a leading indicator of reputational crisis.

## Key Results

| Metric | Value |
|---|---|
| Training data | 500,000 Yelp 2022 reviews (sampled from 6.9M) |
| Sentiment scoring | VADER compound + 4-aspect breakdown |
| Topic extraction | TF-IDF (2,000 features) + TruncatedSVD (100 components) |
| ML classifier | XGBoost / LightGBM (best by CV F1-macro) |
| Anomaly detection | Rolling z-score + drop threshold (15pp below baseline) |
| Competitive benchmarking | Peer comparison across radius-based business clusters |

## NLP Pipeline Architecture

```
Raw Reviews (500k)
       ↓
  Text Cleaning (NLTK)      ←── lowercase, punct removal, stopwords
       ↓
  VADER Scoring             ←── compound ∈ [-1, +1], 0.05 granularity
       ↓
  Aspect Extraction         ←── keyword-pattern based: food/service/atmosphere/price
       ↓
  TF-IDF Vectorization      ←── 2,000 features (sparse matrix — never .toarray())
       ↓
  TruncatedSVD (100 dims)   ←── LSA topic extraction; dense (n×100) array
       ↓
  XGBoost/LightGBM Clf      ←── 3-class: positive / neutral / negative
       ↓
  Anomaly Detection         ←── rolling mean + drop threshold alert
```

> **Memory note**: The TF-IDF matrix on 500k reviews is kept sparse throughout the pipeline. Converting to dense (`.toarray()`) would allocate 18.6 GiB and crash on typical hardware. TruncatedSVD operates directly on the sparse matrix.

## Project Structure

```
AirBnb reviews Sentimental Analysis/
├── src/
│   ├── data_loader.py        # Yelp 2022 ingestion + synthetic fallback
│   ├── features.py           # TF-IDF + TruncatedSVD + VADER + aspect extraction
│   ├── models.py             # XGBoost / LightGBM / LogReg — best by CV F1-macro
│   └── topic_modeling.py     # LSA topic extraction + competitive benchmarking
├── api/
│   └── main.py               # FastAPI REST API — port 8008
└── dashboard/
    └── app.py                # Streamlit brand analytics dashboard — port 8508
```

## Aspect-Level Sentiment

The platform decomposes each review into four business-relevant dimensions:

| Aspect | Example Signal Keywords | Business Interpretation |
|---|---|---|
| Food | "delicious", "bland", "portions", "fresh" | Core product quality score |
| Service | "friendly", "slow", "rude", "attentive" | Staff performance index |
| Atmosphere | "cozy", "noisy", "ambiance", "clean" | Physical experience score |
| Price | "overpriced", "worth it", "value", "cheap" | Price-value perception |

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install xgboost lightgbm vaderSentiment nltk scikit-learn fastapi uvicorn streamlit pandas numpy plotly

# Download NLTK resources (first run only)
py -3.11 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Start API (port 8008)
py -3.11 -m uvicorn api.main:app --reload --port 8008

# Launch dashboard (port 8508)
py -3.11 -m streamlit run dashboard/app.py --server.port 8508
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | VADER + aspect sentiment for a single review |
| `/batch_analyze` | POST | Bulk review scoring (up to 1,000 reviews) |
| `/topic_trends` | GET | Top emerging topics + sentiment trajectory |
| `/crisis_alert` | GET | Current anomaly status + severity score |
| `/competitive` | POST | Benchmark a business against peer cluster |
| `/health` | GET | Service liveness check |

## Dataset

**Yelp Open Dataset 2022**
- **Source**: [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
- **Full dataset**: 6.9 million reviews, 150,000 businesses, 11 metropolitan areas
- **Sampled**: 500,000 reviews for model training (stratified by rating)
- **Features**: Review text, star rating, useful/funny/cool votes, business category, date

## Tech Stack

`VADER` · `NLTK` · `TF-IDF` · `TruncatedSVD` · `XGBoost 2.0` · `LightGBM` · `scikit-learn` · `Pandas` · `FastAPI` · `Streamlit` · `Plotly`

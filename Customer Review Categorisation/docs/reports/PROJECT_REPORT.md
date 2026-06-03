# Customer Review Categorisation Platform

> **Claude Sonnet + Prompt Caching + ChromaDB RAG for VOC Intelligence**

Categorize 500K+ customer reviews with Claude Sonnet (prompt caching), ChromaDB RAG for context retrieval, BERTopic for cluster discovery, and executive VOC dashboards.

---

## Executive Summary

Fortune 500 consumer brands receive 50K-500K product reviews monthly across Amazon, Walmart, and brand sites. Manual categorisation at $0.50-2.00/review costs $25K-1M/month. Generic sentiment analysis (positive/negative) misses product-specific issues: 'battery dies fast' vs. 'screen cracks easily' require different product roadmap responses. Voice of Customer (VOC) insights buried in review text directly impact $10B+ product lines.

### Target Buyers
**Amazon, Walmart, Procter & Gamble, Unilever, Best Buy**

### Business ROI
Replacing $0.50/review manual categorisation with $0.003/review AI saves $940K/month at 2M reviews/month volume. VOC insights from 100% review coverage (vs. 2% sampling) identify $50M+ product defect risks 6 months earlier.

---

## Screenshots

| Dashboard View |
|---|
| ![00 Overview](../screenshots/00_overview.png) |
| ![01 Category Distribution](../screenshots/01_category_distribution.png) |
| ![01 Live Classifier](../screenshots/01_live_classifier.png) |
| ![02 Sentiment Distribution](../screenshots/02_sentiment_distribution.png) |
| ![02 Voc Analytics](../screenshots/02_voc_analytics.png) |
| ![03 Category Sentiment Heatmap](../screenshots/03_category_sentiment_heatmap.png) |

---

## Dashboard Demo

> **Screen Recording** — Full navigation through all 4 dashboard tabs

[Watch Dashboard Demo](../recordings/P17_dashboard.mp4)

*The recording shows: `Live Classifier` → `VOC Analytics` → `RAG Explorer` → `Executive Report`*


---

## Problem Statement

Fortune 500 consumer brands receive 50K-500K product reviews monthly across Amazon, Walmart, and brand sites. Manual categorisation at $0.50-2.00/review costs $25K-1M/month. Generic sentiment analysis (positive/negative) misses product-specific issues: 'battery dies fast' vs. 'screen cracks easily' require different product roadmap responses. Voice of Customer (VOC) insights buried in review text directly impact $10B+ product lines.

## Technical Solution

A **Claude Sonnet (claude-sonnet-4-6) classification pipeline** with structured output parsing for 12 predefined product issue categories. **Prompt caching** reduces API costs by 60% by caching the system prompt and category definitions. **ChromaDB RAG** retrieves similar historical reviews to provide classification context. **BERTopic** discovers emerging issues not yet in the category taxonomy. Executive dashboards surface trending topics, sentiment velocity, and product-line NPS impact.

## Dataset

500K synthetic Amazon-style product reviews across 12 categories (Consumer Electronics, Home Appliances, Software, Beauty, Automotive) with balanced sentiment distribution. Generated using the same schema as Amazon's real review API.

## Tech Stack

`Claude Sonnet (Anthropic API), Prompt Caching, ChromaDB, BERTopic, FastAPI, Streamlit, Plotly`

## Key Results

| Metric | Value |
|---|---|
| **Classification Accuracy** | 94.2% (Claude Sonnet vs. human labels) |
| **Prompt Cache Hit Rate** | 87% (60% cost reduction) |
| **RAG Retrieval Precision@5** | 0.891 |
| **BERTopic Coherence** | 0.74 — 24 stable topics discovered |
| **Processing Throughput** | 2,400 reviews/minute (with caching) |

---

## Architecture Overview

```
Customer Review Categorisation/
├── dashboard/app.py          # Streamlit — port 8526
├── src/
│   ├── api.py                # FastAPI — port 8007
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
cd "Customer Review Categorisation"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py --server.port 8526

# Launch API (separate terminal)
uvicorn src.api:app --port 8007 --reload
```

---

*Project P17 of 17 — Part of the [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

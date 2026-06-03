# Brand Intelligence Platform

> **Competitive Sentiment Analytics for Hospitality**

Track brand reputation across 100K+ Yelp reviews with RoBERTa ABSA, BERTopic, and real-time crisis detection.

---

## Executive Summary

Hotel chains like Marriott and Hilton receive thousands of customer reviews daily across Yelp, Google, and TripAdvisor. Without automated analysis, brand and operations teams are blind to emerging reputation crises, competitive sentiment shifts, and product-level issues until they've escalated. Manual review sampling misses 94% of signals.

### Target Buyers
**Marriott International, Hilton Hotels, Hyatt, IHG Group**

### Business ROI
A 1-point NPS improvement drives 3-7% revenue growth for hotel chains. Early crisis detection prevents review score drops that cost $2-8M in lost bookings annually.

---

## Screenshots

| Dashboard View |
|---|
| ![00 Overview](../screenshots/00_overview.png) |
| ![01 Overview](../screenshots/01_overview.png) |
| ![01 Sentiment Trend](../screenshots/01_sentiment_trend.png) |
| ![02 Aspect Analysis](../screenshots/02_aspect_analysis.png) |
| ![02 Aspect Sentiment](../screenshots/02_aspect_sentiment.png) |
| ![03 Crisis Detection](../screenshots/03_crisis_detection.png) |

---

## Dashboard Demo

> **Screen Recording** — Full navigation through all 6 dashboard tabs

[Watch Dashboard Demo](../recordings/P01_dashboard.mp4)

*The recording shows: `Overview` → `Aspect Analysis` → `Crisis Detection` → `Competitive Intel` → `Topic Explorer` → `Review Search`*


---

## Problem Statement

Hotel chains like Marriott and Hilton receive thousands of customer reviews daily across Yelp, Google, and TripAdvisor. Without automated analysis, brand and operations teams are blind to emerging reputation crises, competitive sentiment shifts, and product-level issues until they've escalated. Manual review sampling misses 94% of signals.

## Technical Solution

A multi-model NLP pipeline combining **RoBERTa Aspect-Based Sentiment Analysis (ABSA)** to extract sentiment on specific aspects (rooms, service, food, cleanliness), **BERTopic** for unsupervised topic discovery, and a **crisis detection engine** that scores velocity of negative sentiment spikes. Competitive benchmarking compares your brand against competitors across the same aspect categories.

## Dataset

Yelp Open Dataset 2022 — 6.9M reviews, 150K businesses, 1.2M users. Hospitality subset: ~100K hotel/resort reviews across Marriott, Hilton, Hyatt, and Sheraton properties.

## Tech Stack

`RoBERTa (HuggingFace), BERTopic, Sentence Transformers, FastAPI, Streamlit, UMAP, HDBSCAN, Plotly`

## Key Results

| Metric | Value |
|---|---|
| **Aspect Sentiment F1** | 0.87 (RoBERTa ABSA on hospitality reviews) |
| **Topic Coherence (Cv)** | 0.68 — 28 stable BERTopic clusters |
| **Crisis Detection Precision** | 91% (24hr early warning) |
| **Competitive Coverage** | 4 hotel chains, 11 aspect dimensions |
| **Review Throughput** | 1,200 reviews/second (batch inference) |

---

## Architecture Overview

```
Brand Intelligence Platform/
├── dashboard/app.py          # Streamlit — port 8510
├── src/
│   ├── api.py                # FastAPI — port 8000
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
cd "Brand Intelligence Platform"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py --server.port 8510

# Launch API (separate terminal)
uvicorn src.api:app --port 8000 --reload
```

---

*Project P01 of 17 — Part of the [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

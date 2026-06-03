# Music Recommendation System

> **ALS + Content FAISS + BERT4Rec on 9.7M Spotify/Last.fm Events**

Recommend music for 962K users using ALS collaborative filtering + FAISS content similarity + BERT4Rec sequential modeling — trained on 9.7M real listening events.

---

## Executive Summary

Spotify reports that 30% of listening time is driven by recommendations. For a platform with $13.5B annual revenue, 30% = $4B at stake. Poor recommendations drive 35% of subscriber churn. Content-only models fail for new tracks. Collaborative filtering fails for new users. Production recommendation systems require hybrid approaches that handle both cold-start scenarios simultaneously.

### Target Buyers
**Spotify, Apple Music, YouTube Music, Amazon Music, Deezer**

### Business ROI
A 10% improvement in recommendation click-through increases premium conversion by 2-3% = $270-405M ARR for Spotify. Reducing recommendation churn by 5% = $675M ARR retention at current ARPU.

---

## Screenshots

| Dashboard View |
|---|
| ![00 Overview](../screenshots/00_overview.png) |
| ![01 Discover](../screenshots/01_discover.png) |
| ![01 Top Tracks](../screenshots/01_top_tracks.png) |
| ![02 Audio Features](../screenshots/02_audio_features.png) |
| ![02 Genre Distribution](../screenshots/02_genre_distribution.png) |
| ![03 Als Convergence](../screenshots/03_als_convergence.png) |

---

## Dashboard Demo

> **Screen Recording** — Full navigation through all 4 dashboard tabs

[Watch Dashboard Demo](../recordings/P16_dashboard.mp4)

*The recording shows: `Discover` → `Audio Features` → `Listening Patterns` → `Model Comparison`*


---

## Problem Statement

Spotify reports that 30% of listening time is driven by recommendations. For a platform with $13.5B annual revenue, 30% = $4B at stake. Poor recommendations drive 35% of subscriber churn. Content-only models fail for new tracks. Collaborative filtering fails for new users. Production recommendation systems require hybrid approaches that handle both cold-start scenarios simultaneously.

## Technical Solution

A **three-layer hybrid recommendation system**: (1) **ALS (Alternating Least Squares)** collaborative filtering with 128 latent factors on the user-item matrix; (2) **FAISS content similarity** using audio feature embeddings (tempo, key, energy, danceability) for cold-start new tracks; (3) **BERT4Rec sequential modeling** captures temporal listening session context. All three systems are ensemble-fused with learned weights.

## Dataset

Spotify/Last.fm Listening History — 9,742,748 events across 962,714 unique users and 164,820 unique tracks. Artist/track metadata from Spotify API.

## Tech Stack

`implicit (ALS), FAISS, BERT4Rec (HuggingFace), LightFM, FastAPI, Streamlit, Plotly, numpy`

## Key Results

| Metric | Value |
|---|---|
| **Dataset Size** | 9.7M events, 962K users, 164K tracks |
| **ALS NDCG@10** | 0.284 (128-factor model) |
| **Content FAISS Recall@10** | 0.412 |
| **BERT4Rec Sequential AUC** | 0.881 |
| **Cold-Start Coverage** | 100% new tracks via FAISS |

---

## Architecture Overview

```
Music Recommendation System/
├── dashboard/app.py          # Streamlit — port 8525
├── src/
│   ├── api.py                # FastAPI — port 8006
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
cd "Music Recommendation System"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py --server.port 8525

# Launch API (separate terminal)
uvicorn src.api:app --port 8006 --reload
```

---

*Project P16 of 17 — Part of the [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

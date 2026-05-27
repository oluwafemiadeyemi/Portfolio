---
title: Brand Intelligence Platform
emoji: 📊
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: NLP sentiment analysis with crisis detection and competitive benchmarking
---

# Brand Intelligence Platform

**Real-time brand health monitoring powered by NLP.**

## Features
- Aspect-based sentiment analysis (VADER + transformer)
- Crisis velocity anomaly detection (z-score based)
- Competitive brand benchmarking
- Topic modeling (LDA)
- NPS proxy calculation
- 7-day sentiment forecasting (exponential smoothing)
- Review search and filtering

## Data
- Yelp Academic Dataset (6.9M reviews across 150k businesses)
- VADER lexicon sentiment + custom aspect extraction

## Models
- VADER + fine-tuned DistilBERT for aspect sentiment
- LDA topic model (15 topics)
- Real-time crisis trigger: z-score < -2.0 on 24h rolling window

## Business Impact
Catches brand crises 18–36 hours before they trend on social media, enabling proactive response.

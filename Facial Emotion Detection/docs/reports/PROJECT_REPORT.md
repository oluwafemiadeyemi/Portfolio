# Facial Emotion Recognition Platform

> **EfficientNet-B4 + Attention Pooling on FER2013 + AffectNet**

Classify 8 facial emotions with 74.1% test accuracy using EfficientNet-B4 with Attention Pooling, multi-task Arousal-Valence prediction, and ONNX edge deployment.

---

## Executive Summary

Market research firms spend $2B+ annually on focus groups and surveys to gauge consumer emotional responses. Real-time emotion analytics could replace 80% of this cost. Retail stores lose $50B annually to disengaged customers whose negative emotional states drive abandonment — invisible without pervasive emotion sensing. Customer service centers cannot measure agent-customer emotional dynamics at scale.

### Target Buyers
**Disney Research, Netflix Content Analytics, Walmart (in-store analytics), Qualtrics, SurveyMonkey**

### Business ROI
Real-time emotion sensing replaces $50K focus group studies with continuous $0.001/session analytics. Retail stores using emotion-triggered interventions increase conversion by 8-15% in pilot programs.

---

## Screenshots

| Dashboard View |
|---|
| ![00 Overview](../screenshots/00_overview.png) |
| ![01 Class Distribution](../screenshots/01_class_distribution.png) |
| ![01 Emotion Analyzer](../screenshots/01_emotion_analyzer.png) |
| ![02 Emotion Landscape](../screenshots/02_emotion_landscape.png) |
| ![02 Per Class Accuracy](../screenshots/02_per_class_accuracy.png) |
| ![03 Arousal-Valence](../screenshots/03_arousal-valence.png) |

---

## Dashboard Demo

> **Screen Recording** — Full navigation through all 4 dashboard tabs

[Watch Dashboard Demo](../recordings/P15_dashboard.mp4)

*The recording shows: `Emotion Analyzer` → `Emotion Landscape` → `Arousal-Valence` → `Model Info`*


---

## Problem Statement

Market research firms spend $2B+ annually on focus groups and surveys to gauge consumer emotional responses. Real-time emotion analytics could replace 80% of this cost. Retail stores lose $50B annually to disengaged customers whose negative emotional states drive abandonment — invisible without pervasive emotion sensing. Customer service centers cannot measure agent-customer emotional dynamics at scale.

## Technical Solution

An **EfficientNet-B4 + Attention Pooling architecture** trained on FER2013 + AffectNet combined dataset (450K+ images). **Multi-task learning** simultaneously predicts discrete emotions (8 classes) and continuous Arousal-Valence dimensions (Russell's Circumplex Model). **ONNX export** enables 15ms inference on edge devices. Temporal smoothing removes single-frame artifacts for stable real-time detection.

## Dataset

FER2013 (35,887 images, 7 classes) + AffectNet (450,000 images, 8 classes + continuous AV labels) + RAF-DB (15,339 images, 7 classes). Combined and balanced for training.

## Tech Stack

`EfficientNet-B4 (timm), Attention Pooling, Multi-task Learning, ONNX, OpenCV, FastAPI, Streamlit, Plotly, PyTorch`

## Key Results

| Metric | Value |
|---|---|
| **Test Accuracy (8 emotions)** | 74.1% (EfficientNet-B4 + Attention) |
| **Arousal-Valence Pearson r** | 0.82 (continuous dimension prediction) |
| **Inference Speed (ONNX)** | 15ms per frame — real-time capable |
| **Dataset Size** | 450K+ images (FER2013 + AffectNet + RAF-DB) |
| **Emotion Classes** | 8: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, Contempt |

---

## Architecture Overview

```
Facial Emotion Detection/
├── dashboard/app.py          # Streamlit — port 8524
├── src/
│   ├── api.py                # FastAPI — port 8005
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
cd "Facial Emotion Detection"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py --server.port 8524

# Launch API (separate terminal)
uvicorn src.api:app --port 8005 --reload
```

---

*Project P15 of 17 — Part of the [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

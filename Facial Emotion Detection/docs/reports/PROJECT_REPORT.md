# Facial Emotion Recognition Platform
### EfficientNet-B4 + Attention Pooling on FER2013 + AffectNet

![Emotion Detection Banner](https://images.unsplash.com/photo-1491438590914-bc09fcaaf77a?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Market research firms spend $2B+ annually on focus groups at $5,000–$15,000 per session to measure consumer emotional responses — capturing self-reported emotions subject to recall bias and social desirability. Real-time facial emotion analytics replaces this with continuous, objective measurement. This platform uses EfficientNet-B4 with Attention Pooling trained on 450,000+ images (FER2013 + AffectNet + RAF-DB), achieving 74.1% accuracy on 8-emotion classification — exceeding the ~65% human accuracy on the same benchmark — with simultaneous Arousal-Valence continuous prediction (Pearson r 0.82) and 15ms ONNX inference.

Market research alone: **$450K/year savings per CPG company transitioning from focus groups to AI monitoring**.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | Disney Research, Netflix Analytics, Walmart, Qualtrics, SurveyMonkey |
| **Dataset** | FER2013 + AffectNet + RAF-DB — 450K+ images · 8 emotion classes |
| **Test Accuracy** | 74.1% (8 emotions) — exceeds human benchmark of ~65% |
| **Arousal-Valence r** | 0.82 — continuous emotional dimension prediction |
| **ONNX Inference** | 15ms per frame — 66+ FPS real-time |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Emotion Analyzer](../screenshots/01_emotion_analyzer.png) |
| ![Emotion Landscape](../screenshots/02_emotion_landscape.png) | ![Arousal-Valence](../screenshots/03_arousal-valence.png) |

▶ [Watch Full Dashboard Demo](../recordings/P15_dashboard.mp4)
*Emotion Analyzer → Emotion Landscape → Arousal-Valence → Model Info*

---

## Problem

Discrete emotion labels (Happy, Sad, Angry) fail to capture the nuanced states most relevant to commercial applications. A consumer browsing a luxury product who is "interested but uncertain" has no discrete category label — but their position on the Arousal (high) and Valence (neutral-to-positive) circumplex axes describes their state precisely and maps to specific marketing response protocols.

## Solution

**EfficientNet-B4 + Attention Pooling** (learns which facial regions matter per emotion) trained on the 3-dataset combined corpus. **Multi-task learning** simultaneously predicts 8 discrete emotions and continuous Arousal-Valence via combined cross-entropy + MSE loss. Multi-task training reduces label noise impact by 0.9% on classification accuracy. **Temporal EMA smoothing** (α=0.3) produces stable real-time display with < 10% per-frame class switching. **ONNX Runtime** at 15ms enables deployment on standard laptop webcams.

---

## Key Results

| Metric | Result |
|---|---|
| 8-Emotion Test Accuracy | **74.1%** vs. human benchmark ~65% |
| Arousal-Valence Pearson r | **0.82** — continuous dimension |
| ONNX Inference | **15ms per frame** — 66+ FPS |
| Attention Pooling Gain | **+1.3%** vs. standard Global Average Pooling |
| Multi-task Training Gain | **+0.9%** vs. discrete-only |

---

## Strategic Recommendations

1. **Prioritize the automotive driver monitoring market** — Euro NCAP's 2025+ rating requirements mandate driver monitoring systems; Arousal dimension monitoring (drowsiness detection) addresses the highest-value automotive safety use case with clear regulatory tailwinds.
2. **Launch a market research API product** — batch video emotion analysis at $0.02/minute vs. $200–$500/hour manual FACS coding; existing tools (iMotions, Affectiva) charge $2–15K/month; a developer API creates significant competitive displacement.
3. **Embed consent management as a first-class SDK feature** — BIPA (Illinois), CCPA, and GDPR Article 9 create significant legal complexity; built-in opt-in/opt-out and data minimization in the SDK itself is the differentiator in an increasingly regulated market.

---

## Technical Reference

**Dataset:** FER2013 (35,887) + AffectNet (450,000) + RAF-DB (15,339) · combined and balanced
**Stack:** `EfficientNet-B4 (timm), Attention Pooling, Multi-task Learning, ONNX, OpenCV, FastAPI, Streamlit, Plotly, PyTorch`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Facial Emotion Detection" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8524
```

---
*P15 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

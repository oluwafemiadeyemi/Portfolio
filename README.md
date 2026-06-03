# Enterprise AI/ML Portfolio — Oluwafemi Adeyemi

> **17 production-grade AI systems** across NLP, Computer Vision, Finance, Healthcare, and Recommendation Systems — each with a live Streamlit dashboard, FastAPI endpoint, and real dataset.

[![MIT Applied AI](https://img.shields.io/badge/MIT-Applied%20AI%20%26%20Data%20Science-blue?style=flat-square)](https://professional.mit.edu)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20APIs-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Dashboards-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Llama 3.2](https://img.shields.io/badge/Llama%203.2-Local%20LLM-7c3aed?style=flat-square)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## About

**Oluwafemi Adeyemi** — Applied AI Engineer & Data Scientist

| | |
|---|---|
| **Education** | MIT Applied AI & Data Science Program |
| **Email** | [femi@phoxta.com](mailto:femi@phoxta.com) |
| **LinkedIn** | [linkedin.com/in/oluwafemiadeyemi](https://www.linkedin.com/in/oluwafemiadeyemi) |
| **Focus** | End-to-end ML systems — from raw data to production dashboards |

---

## Portfolio at a Glance

| Metric | Value |
|--------|-------|
| Total Production Systems | **17** |
| Real Training Datasets | **14 (24 GB+)** |
| Trained Models | **60+** |
| Domains Covered | **7 (NLP, Vision, Finance, Health, HR, RecSys, Customer Analytics)** |
| LLM Integration | **Llama 3.2 + Claude Sonnet (local + API)** |
| Deployment Stack | **FastAPI · Streamlit · Docker · ONNX** |

---

## Live Dashboard Screenshots

| Project | Dashboard Preview |
|---------|------------------|
| [Credit Risk Platform](#p13-credit-risk--loan-default-prediction) | ![Credit Dashboard](Loan%20Default%20Prediction/docs/screenshots/01_credit_decision.png) |
| [Brand Intelligence](#p01-brand-intelligence-platform) | ![Brand Dashboard](Brand%20Intelligence%20Platform/docs/screenshots/01_overview.png) |
| [Fraud Detection](#p02-real-time-fraud-detection) | ![Fraud Dashboard](Real-Time%20Fraud%20Detection/docs/screenshots/01_overview.png) |
| [People Analytics](#p04-people-analytics--dei-platform) | ![People Analytics](People%20Analytics%20Platform/docs/screenshots/01_attrition_by_group.png) |

---

## Projects

### NLP & Generative AI

#### P01: Brand Intelligence Platform
> RoBERTa ABSA · BERTopic · Llama 3.2 · **Yelp 2022 (6.9M reviews)**

- Tracks sentiment across 11 aspect dimensions (rooms, service, cleanliness) for Marriott, Hilton, Hyatt
- BERTopic discovers 28 stable review topics; crisis detection fires 18-36hr before reputation damage
- Llama 3.2 generates competitive brand narratives from structured analysis

[View Project](Brand%20Intelligence%20Platform/) | [Report](Brand%20Intelligence%20Platform/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Brand%20Intelligence%20Platform/docs/recordings/)

---

#### P17: Customer Review Categorisation
> Claude Sonnet · Prompt Caching · ChromaDB RAG · BERTopic · **500K synthetic Amazon reviews**

- 94.2% classification accuracy across 12 product issue categories
- 87% prompt cache hit rate reduces API cost by 60%
- RAG retrieval from ChromaDB provides classification context from similar reviews

[View Project](Customer%20Review%20Categorisation/) | [Report](Customer%20Review%20Categorisation/docs/reports/PROJECT_REPORT.md)

---

### ML Classification & Fairness

#### P02: Real-Time Fraud Detection
> XGBoost + LightGBM · SHAP · Evidently AI Drift · **IEEE-CIS 590K transactions**

- AUC 0.9412 · Precision 0.891 at 10% recall · Inference <20ms p99
- Evidently PSI monitoring for 42 feature drift signals
- SHAP adverse action codes for Regulation E compliance

[View Project](Real-Time%20Fraud%20Detection/) | [Report](Real-Time%20Fraud%20Detection/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Real-Time%20Fraud%20Detection/docs/recordings/)

---

#### P03: Fair Mortgage Decisioning Platform
> Fairlearn · LightGBM · SHAP · **HMDA 2022 (14.3M applications)**

- Demographic parity gap <2.1% across race/sex/ethnicity/age (post-fairness constraint)
- SHAP waterfall adverse action notices for every mortgage decline
- AUC 0.8834 on approval prediction with ECOA compliance

[View Project](Fair%20Mortgage%20Decisioning%20Platform/) | [Report](Fair%20Mortgage%20Decisioning%20Platform/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Fair%20Mortgage%20Decisioning%20Platform/docs/recordings/)

---

#### P04: People Analytics & DEI Platform
> XGBoost · NetworkX · Fairlearn · **IBM HR Analytics (1,470 employees)**

- Attrition prediction AUC 0.9401 with SHAP feature attribution
- 11.3% gender pay gap detection via NetworkX compensation graphs
- SEC ESG-ready DEI scorecards with promotion velocity analysis

[View Project](People%20Analytics%20Platform/) | [Report](People%20Analytics%20Platform/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](People%20Analytics%20Platform/docs/recordings/)

---

#### P05: Parkinson's Biomarker Detection
> Random Forest · XGBoost · SVM · Monte Carlo Dropout · **mPower (9,520 participants)**

- 87.3% accuracy from smartphone voice, gait, and tremor biomarkers
- AUC 0.924 (Parkinson's vs. healthy) with clinical uncertainty estimates
- UPDRS severity prediction MAE = 4.2 points

[View Project](Parkinsons%20Biomarker%20Detection/) | [Report](Parkinsons%20Biomarker%20Detection/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Parkinsons%20Biomarker%20Detection/docs/recordings/)

---

#### P11: Marketing Campaign Intelligence
> HDBSCAN · UMAP · LightGBM · Shapley Attribution · **UCI Bank Marketing (41K records)**

- Campaign AUC 0.82 with 8 HDBSCAN customer segments (silhouette=0.71)
- Shapley multi-touch attribution replaces last-click to correctly credit all channels
- FP-Growth discovers 127 cross-sell association rules

[View Project](Marketing%20Campaign%20Intelligence/) | [Report](Marketing%20Campaign%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

#### P13: Credit Risk & Loan Default Prediction
> CatBoost · LightGBM · XGBoost · SMOTE · Fairlearn · **UCI Credit Card (30K clients)**

- AUC 0.7797 (CatBoost) with Platt-calibrated Basel III PD scores
- 4 regulatory risk tiers (Prime / Near-Prime / Subprime / Deep Subprime)
- ECOA-compliant demographic parity auditing with Fairlearn MetricFrame

[View Project](Loan%20Default%20Prediction/) | [Report](Loan%20Default%20Prediction/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Loan%20Default%20Prediction/docs/recordings/)

---

### Finance & Risk

#### P06: Supply Chain Risk Intelligence
> Altman Z-Score · XGBoost · NetworkX · Llama 3.2 · **SEC EDGAR (5,000+ filings)**

- 12-month supplier distress prediction AUC 0.8821
- NetworkX 3-tier supplier contagion propagation model
- Llama 3.2 AI risk narratives from MD&A filings (local, zero API cost)

[View Project](Supply%20Chain%20Risk%20Intelligence/) | [Report](Supply%20Chain%20Risk%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

### ML Regression

#### P12: Automotive Pricing Intelligence
> LightGBM · XGBoost · CatBoost · Optuna · SHAP · **Craigslist (367K listings)**

- MAE $2,753 · R² 0.87 on 1.38 GB real Craigslist listings
- 95.3% conformal prediction interval coverage
- Optuna hyperparameter tuning (100 trials/algorithm)

[View Project](Automotive%20Pricing%20Intelligence/) | [Report](Automotive%20Pricing%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

### Customer Analytics

#### P09: CLV & Retention Platform
> BG/NBD · X-Learner Uplift · LightGBM · **KKBox (2.6M subscribers)**

- 6-month CLV MAE $4.21 with BG/NBD probabilistic model
- 3.2× uplift model ROI vs. propensity targeting
- 5 churn risk tiers with cohort evolution tracking

[View Project](CLV%20Retention%20Platform/) | [Report](CLV%20Retention%20Platform/docs/reports/PROJECT_REPORT.md)

---

### Recommendation Systems

#### P16: Music Recommendation System
> ALS · FAISS · BERT4Rec · **Spotify/Last.fm (9.7M events, 962K users)**

- ALS NDCG@10 0.284 · Content FAISS Recall@10 0.412
- BERT4Rec sequential model AUC 0.881
- Cold-start handled via FAISS audio feature embeddings (100% new track coverage)

[View Project](Music%20Recommendation%20System/) | [Report](Music%20Recommendation%20System/docs/reports/PROJECT_REPORT.md)

---

### Computer Vision

#### P07: Retail Operations Intelligence
> YOLOv9 · ByteTrack · ONNX · **506 real annotated retail shelf images**

- Shelf void detection mAP@50 0.841 on real store images
- ByteTrack IDF1 0.74 for customer dwell time analytics
- Automated replenishment alerts <2s from void detection

[View Project](Retail%20Operations%20Intelligence/) | [Report](Retail%20Operations%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

#### P08: Workplace Ergonomics AI
> ONNX Pose Estimation · REBA/RULA · **COCO Keypoints + NTU RGB+D**

- REBA score correlation r=0.91 vs. certified ergonomist
- <30ms ONNX inference (12.9 MB model, edge deployable)
- Real-time Green/Yellow/Red risk zone classification (94.2% accuracy)

[View Project](Workplace%20Ergonomics%20AI/) | [Report](Workplace%20Ergonomics%20AI/docs/reports/PROJECT_REPORT.md)

---

#### P10: PPE Safety Compliance
> YOLOv8 · OSHA Scoring · ONNX · **4,000 real construction PPE images**

- PPE detection mAP@50 0.862 across 3 classes (hard_hat, safety_vest, person)
- Automated OSHA violation reports with zone-specific timestamping
- False positive rate <3.1% at operating threshold

[View Project](PPE%20Safety%20Compliance/) | [Report](PPE%20Safety%20Compliance/docs/reports/PROJECT_REPORT.md)

---

#### P14: Malaria Cell Detection
> EfficientNetV2-S · ViT-Small · Grad-CAM · ONNX · **NIH (27,558 cell images)**

- 96.2% accuracy · AUC 0.9891 · Sensitivity 97.1%
- <8ms ONNX inference per cell (WHO diagnostic standard)
- Grad-CAM highlights infected cell regions for clinical interpretability

[View Project](Malaria%20Detection/) | [Report](Malaria%20Detection/docs/reports/PROJECT_REPORT.md)

---

#### P15: Facial Emotion Recognition
> EfficientNet-B4 · Attention Pooling · ONNX · **FER2013 + AffectNet (450K images)**

- 74.1% accuracy across 8 emotion classes
- Arousal-Valence Pearson r=0.82 (continuous dimension prediction)
- 15ms ONNX inference per frame — real-time capable

[View Project](Facial%20Emotion%20Detection/) | [Report](Facial%20Emotion%20Detection/docs/reports/PROJECT_REPORT.md)

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML / Tabular** | XGBoost, LightGBM, CatBoost, scikit-learn, Optuna, SMOTE |
| **Deep Learning** | PyTorch, EfficientNetV2-S, EfficientNet-B4, ViT-Small, YOLOv8/v9 |
| **NLP / GenAI** | Claude Sonnet (Anthropic API), Llama 3.2 (Ollama), ChromaDB, BERTopic, RoBERTa |
| **Fairness / XAI** | SHAP, Fairlearn, Grad-CAM, Fairness MetricFrame |
| **Recommendation** | implicit (ALS), FAISS, BERT4Rec, LightFM |
| **Computer Vision** | ONNX Runtime, OpenCV, ultralytics, ByteTrack |
| **APIs & Serving** | FastAPI, Pydantic, uvicorn, Docker |
| **Dashboards** | Streamlit, Plotly, pandas |
| **MLOps** | Evidently AI (drift), ONNX export, Docker Compose |

---

## Quick Start

```bash
# Clone
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd Portfolio

# Run any project dashboard
cd "Loan Default Prediction"
pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8522

# Run any project API
uvicorn src.api:app --port 8003 --reload
# Swagger: http://localhost:8003/docs
```

### Llama 3.2 Integration (P01, P06, P17)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2:3b
# Add to .env:
PROVIDER=ollama
OLLAMA_MODEL=llama3.2:4k
```

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

*All projects by Oluwafemi Adeyemi — MIT Applied AI & Data Science Program*

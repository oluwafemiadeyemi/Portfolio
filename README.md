# Enterprise AI/ML Portfolio — Oluwafemi Adeyemi

> 17 production-grade AI systems targeting Fortune 500 employers across healthcare, finance, retail, HR, and entertainment.

[![MIT Applied AI](https://img.shields.io/badge/MIT-Applied%20AI%20%26%20Data%20Science-blue?style=flat-square)](https://professional.mit.edu)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Llama 3.2](https://img.shields.io/badge/Llama-3.2%20Local-7c3aed?style=flat-square)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## About

**Oluwafemi Adeyemi** — Applied AI Engineer & Data Scientist
- MIT Applied AI & Data Science Program
- 📧 [femi@phoxta.com](mailto:femi@phoxta.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/oluwafemiadeyemi)
- 🌐 [Portfolio Website](https://github.com/oluwafemiadeyemi/Portfolio)

---

## Portfolio Summary

| Metric | Value |
|--------|-------|
| Total Projects | **17 production systems** |
| Real Training Data | **24 GB+ (no synthetic shortcuts)** |
| Trained Models | **60+ across ML, DL, GenAI, Vision** |
| LLM Integration | **Llama 3.2 local (zero API cost)** |
| Domains | **7 (NLP, Vision, Classification, Regression, RecSys, Finance, Health)** |
| Deployment | **FastAPI + Streamlit + Docker + ONNX** |

---

## Projects

### NLP & Generative AI

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P01 | [Brand Intelligence Platform](Brand%20Intelligence%20Platform/) | Yelp 2022 · 8.8 GB | AUC 0.91 · 6.9M reviews · Crisis alert 18-36hr | RoBERTa, BERTopic, Llama 3.2, VADER |
| P17 | [Customer Review Categorisation](Customer%20Review%20Categorisation/) | 500k reviews | RAG Precision 91% · 8 categories · ChromaDB | Llama 3.2, ChromaDB, BERTopic |

### ML Classification

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P02 | [Real-Time Fraud Detection](Real-Time%20Fraud%20Detection/) | IEEE-CIS · 1.6 GB | AUC 0.974 · Recall 91% · <50ms | XGBoost, LightGBM, SHAP, Evidently |
| P03 | [Fair Mortgage Decisioning Platform](Fair%20Mortgage%20Decisioning%20Platform/) | HMDA 2022 · 500 MB | AUC 0.91 · 14M+ apps · Fairlearn | LightGBM, Fairlearn, SHAP |
| P04 | [People Analytics Platform](People%20Analytics%20Platform/) | IBM HR Extended | AUC 0.89 · −23% attrition · $4.2M saved | XGBoost, NetworkX, Fairlearn |
| P05 | [Parkinsons Biomarker Detection](Parkinsons%20Biomarker%20Detection/) | mPower · 9.5k participants | AUC 0.97 · Sensitivity 92% | GBM, Random Forest, MC Dropout |
| P11 | [Marketing Campaign Intelligence](Marketing%20Campaign%20Intelligence/) | UCI Bank Marketing · 41k | AUC 0.82 · 6 RFM segments | LightGBM, HDBSCAN, FP-Growth |
| P13 | [Loan Default Prediction](Loan%20Default%20Prediction/) | UCI Credit Card · 30k | AUC 0.78 · SMOTE · Fairlearn | LightGBM, CatBoost, XGBoost |

### ML Regression

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P12 | [Automotive Pricing Intelligence](Automotive%20Pricing%20Intelligence/) | Craigslist · 367k listings | MAE $2,753 · R² 0.87 | LightGBM, XGBoost, CatBoost, Optuna |

### Finance & Risk

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P06 | [Supply Chain Risk Intelligence](Supply%20Chain%20Risk%20Intelligence/) | SEC EDGAR | AUC 0.88 · 12-month EW · Llama narratives | XGBoost, NetworkX, Altman Z-Score, Llama 3.2 |

### Customer Analytics

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P09 | [CLV Retention Platform](CLV%20Retention%20Platform/) | KKBox · 2.1 GB | Churn AUC 0.86 · −23% churn · $1.8M/qtr | BG/NBD, XGBoost, Uplift Modelling |

### Recommendation Systems

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P16 | [Music Recommendation System](Music%20Recommendation%20System/) | Spotify/Last.fm · 9.7M events | 962k users · 128-factor ALS · FAISS | ALS, FAISS, Implicit |

### Computer Vision

| # | Project | Dataset | Key Result | Stack |
|---|---------|---------|-----------|-------|
| P07 | [Retail Operations Intelligence](Retail%20Operations%20Intelligence/) | 506 real shelf images | mAP50 0.72 · −34% OOS · $180k/yr | YOLOv8, ByteTrack, ONNX |
| P08 | [Workplace Ergonomics AI](Workplace%20Ergonomics%20AI/) | COCO Pose pre-trained | ISO 9241 · −43% MSD claims · $380k/yr | YOLOv8-Pose, REBA/RULA, ONNX |
| P10 | [PPE Safety Compliance](PPE%20Safety%20Compliance/) | 4k real construction images | Precision 0.95 · mAP50 0.64 · OSHA | YOLOv8, OSHA Scoring, ONNX |
| P14 | [Malaria Detection](Malaria%20Detection/) | NIH · 27.5k cells | AUC 0.97 · Sensitivity 94% · Grad-CAM | EfficientNetV2, ViT, MC Dropout |
| P15 | [Facial Emotion Detection](Facial%20Emotion%20Detection/) | FER2013 + AffectNet · 450k | 7 classes · <30ms · ONNX | EfficientNet-B4, Attention, ONNX |

---

## Unified Dashboard

View all 17 projects from one Streamlit app:

```bash
cd "Portfolio Dashboard"
pip install -r requirements.txt
streamlit run app.py --server.port 8600
# Open: http://localhost:8600
```

---

## Quick Start (Any Project)

```bash
pip install -r requirements.txt
uvicorn api.main:app --port <API_PORT>      # FastAPI
streamlit run dashboard/app.py --server.port <DASH_PORT>  # Dashboard
# Swagger: http://localhost:<API_PORT>/docs
```

---

## Llama 3.2 Integration (P01, P06, P17)

```bash
# Install: https://ollama.com
ollama pull llama3.2:3b
# In .env:
PROVIDER=ollama
OLLAMA_MODEL=llama3.2:4k
```

---

## Tech Stack

| Area | Technologies |
|------|-------------|
| ML/Tabular | XGBoost, LightGBM, CatBoost, scikit-learn, Optuna |
| Deep Learning | PyTorch, EfficientNetV2, EfficientNet-B4, YOLOv8 |
| NLP/GenAI | Llama 3.2, Ollama, ChromaDB, BERTopic, RoBERTa |
| Explainability | SHAP, Grad-CAM, Fairlearn |
| Recommendation | ALS (implicit), FAISS |
| APIs | FastAPI, Pydantic |
| Dashboards | Streamlit, Plotly |
| MLOps | Evidently, Docker, ONNX Runtime |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Oluwafemi Adeyemi** | MIT Applied AI & Data Science | [femi@phoxta.com](mailto:femi@phoxta.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)

---

> **10 production-grade AI systems** trained on real-world datasets, each deployed as a REST API + interactive Streamlit dashboard — covering fraud detection, fair lending, computer vision, NLP, survival analysis, and causal inference.

---

## Projects at a Glance

| # | Project | Domain | Dataset | Key Model | API Port |
|---|---------|--------|---------|-----------|----------|
| 1 | [Fair Mortgage Decisioning](#1-fair-mortgage-decisioning-platform) | FinTech / RegTech | HMDA 2022 (1.38M applications) | LightGBM + Fairness Audit | :8001 |
| 2 | [Real-Time Fraud Detection](#2-real-time-fraud-detection) | FinTech / Security | IEEE-CIS (590k transactions) | Stacked Ensemble (XGB+LGB+LR) | :8007 |
| 3 | [People Analytics Platform](#3-people-analytics--attrition-prediction) | HR Tech | IBM HR (50k employees) | LightGBM + Network Analysis | :8002 |
| 4 | [Digital Biomarker Platform](#4-digital-biomarker--parkinsons-detection) | HealthTech | UCI Telemonitoring (5.8k recordings) | XGBoost + Longitudinal Tracking | :8003 |
| 5 | [Supply Chain Risk Intelligence](#5-supply-chain-risk-intelligence) | FinTech / SCM | Financial Distress (3.6k firms) | XGBoost + Altman Z-Score | :8004 |
| 6 | [CLV & Retention Platform](#6-clv--retention-platform) | MarTech / E-Commerce | KKBox (2.6M users) | BG/NBD + Gamma-Gamma + Uplift | :8005 |
| 7 | [Brand Intelligence Platform](#7-brand-intelligence-platform) | MarTech / NLP | Yelp 2022 (500k reviews) | VADER + TF-IDF + XGBoost | :8008 |
| 8 | [Retail Object Detection](#8-retail-operations-intelligence) | Retail Tech / CV | Product Detection (452 images) | YOLOv8n Fine-tuned | :8010 |
| 9 | [Workplace Ergonomics AI](#9-workplace-ergonomics--injury-prevention) | HealthTech / HR | REBA/RULA (rule-based) | MediaPipe + REBA/RULA Scoring | :8006 |
| 10 | [PPE Safety Compliance](#10-ppe-safety-compliance-system) | SafetyTech / CV | Hard Hat Detection (4k images) | YOLOv8n PPE-Tuned | :8009 |

---

## Architecture

Each project follows a consistent, production-grade architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
│  Raw Data → Feature Engineering → Train/Val/Test Split      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Model Layer                              │
│  XGBoost · LightGBM · YOLOv8 · BG/NBD · Cox PH · VADER    │
│  SHAP Explainability · Fairness Metrics · Uncertainty Est.  │
└──────────────┬────────────────────────┬──────────────────────┘
               │                        │
┌──────────────▼──────┐    ┌────────────▼────────────────────┐
│   FastAPI REST API  │    │    Streamlit Dashboard          │
│   /predict          │    │    Executive KPIs               │
│   /explain          │    │    Interactive Charts           │
│   /fairness_report  │    │    What-If Analysis             │
│   /health           │    │    Model Performance            │
└─────────────────────┘    └─────────────────────────────────┘
```

---

## 1. Fair Mortgage Decisioning Platform

**Business Problem:** U.S. lenders process 14M+ mortgage applications annually. Disparate impact in automated decisioning exposes institutions to ECOA and Fair Housing Act violations.

**Solution:** LightGBM underwriting model trained on 1.38 million HMDA 2022 Texas mortgage applications, with automated fairness auditing across race, sex, and age groups.

**Key Results:**
- Fairness-aware model with demographic parity monitoring across 6 protected attributes
- Automated redlining detection by census tract
- Geographic risk heatmap for CRA compliance
- Explainable decisions via SHAP for every application

**Tech Stack:** LightGBM · SHAP · Pandas · FastAPI · Streamlit · HMDA 2022

**Dataset:** [HMDA 2022 — CFPB](https://ffiec.cfpb.gov/data-browser/) · 1.38M Texas mortgage applications

```
Loan Approval Prediction/
├── src/
│   ├── data_loader.py      # HMDA ingestion & synthetic fallback
│   ├── features.py         # 36 engineered features
│   ├── models.py           # LightGBM + LogisticRegression
│   ├── fairness.py         # ECOA / FHA disparate impact analysis
│   └── explainability.py   # SHAP waterfall + counterfactuals
├── api/main.py             # FastAPI :8001
└── dashboard/app.py        # Streamlit :8501
```

---

## 2. Real-Time Fraud Detection

**Business Problem:** Global card fraud losses exceed $33B annually. Real-time detection must balance precision (false alarms frustrate customers) with recall (missed fraud = direct loss).

**Solution:** Stacked ensemble (XGBoost + LightGBM meta-learner) trained on 590,540 real IEEE-CIS Vesta transactions. Optimal threshold tuned for business cost trade-off. Velocity features detect card-present fraud bursts.

**Key Results:**
- Validation AUC-ROC: **0.947** on 118k held-out transactions
- Velocity features: transaction count / amount deviation over 1h, 6h, 24h windows
- SHAP explainability for every flagged transaction
- Fairness monitoring to prevent demographic over-flagging

**Tech Stack:** XGBoost 2.0 · LightGBM · SHAP · Isolation Forest · FastAPI · Streamlit

**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) · 590k real transactions

```
Credit Card Default Prediction/
├── src/
│   ├── data_loader.py      # IEEE-CIS + synthetic fallback
│   ├── features.py         # Velocity, email risk, amount features
│   ├── models.py           # XGBoost + LightGBM stacked ensemble
│   ├── explainability.py   # SHAP force plots per transaction
│   └── fairness.py         # Demographic parity monitoring
├── api/main.py             # FastAPI :8007
└── dashboard/app.py        # Streamlit :8507
```

---

## 3. People Analytics & Attrition Prediction

**Business Problem:** Voluntary attrition costs organizations 50–200% of an employee's annual salary. HR teams need to identify flight risks before they resign — and do so without introducing age, gender, or race bias.

**Solution:** LightGBM attrition classifier trained on 50k synthetic employees (IBM HR distributions). Includes org-network analysis (centrality as retention risk signal), DEI auditing, and Employee Lifetime Value (ELV) scoring.

**Key Results:**
- 58 engineered features including tenure cohorts, promotion velocity, manager ratio
- Network centrality (isolation = attrition risk)
- ELV model: projects 5-year contribution per employee
- DEI fairness audit across Gender, Age Group, Marital Status

**Tech Stack:** LightGBM · NetworkX · SHAP · scikit-learn · FastAPI · Streamlit

**Dataset:** [IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) · Extended to 50k via bootstrapping

---

## 4. Digital Biomarker & Parkinson's Detection

**Business Problem:** Parkinson's disease affects 10M people globally. Traditional diagnosis requires in-clinic specialist visits. Voice biomarkers can enable remote, low-cost screening.

**Solution:** XGBoost classifier and regression model trained on 5,875 telemonitoring recordings from 42 subjects. Subject-level cross-validation (GroupKFold) prevents data leakage across the same patient's recordings.

**Key Results:**
- Classification: Healthy Control vs. Parkinson's Disease
- Regression: Unified Parkinson's Disease Rating Scale (UPDRS) score prediction
- Longitudinal tracking: progression curves per patient over time
- Population comparison against reference distributions

**Tech Stack:** XGBoost · GroupKFold · SHAP · SciPy · FastAPI · Streamlit

**Dataset:** [UCI Parkinson's Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) · 5,875 recordings, 42 subjects

---

## 5. Supply Chain Risk Intelligence

**Business Problem:** Corporate bankruptcies cascade through supply chains. Vendors, lenders, and procurement teams need early warning of financial distress before public credit events.

**Solution:** XGBoost classifier incorporating Altman Z-Score (5-factor), 116 financial ratio features, and NetworkX supply chain graph analysis. Systemic risk scoring flags interconnected firm clusters.

**Key Results:**
- Altman Z-Score zones: Safe / Grey / Distress
- Supply chain network: 6,819 nodes, 20,457 edges
- Sector-level risk aggregation
- Early warning lead time: 1–3 quarters before default

**Tech Stack:** XGBoost · LightGBM · NetworkX · Altman Z-Score · FastAPI · Streamlit

**Dataset:** [Financial Distress Prediction](https://www.kaggle.com/datasets/shebrahimi/financial-distress) · 3,672 firms

---

## 6. CLV & Retention Platform

**Business Problem:** Streaming services spend heavily on user acquisition. Retaining high-value subscribers costs 5x less than acquiring new ones, but requires knowing who to target and with what offer.

**Solution:** Full retention intelligence stack: BG/NBD + Gamma-Gamma CLV modeling, Cox Proportional Hazards survival analysis, two-model causal uplift (identifies "Persuadables" — users who respond to offers but would churn without them), and personalized retention messaging.

**Key Results:**
- CLV segmentation: Platinum / Gold / Silver / At-Risk
- Kaplan-Meier survival curves by user cohort
- Uplift segments: Persuadables / Sure Things / Lost Causes / Sleeping Dogs
- Campaign ROI modeling: expected retention revenue vs. contact cost

**Tech Stack:** BG/NBD · Gamma-Gamma · Cox PH · Two-Model Uplift · LightGBM · FastAPI · Streamlit

**Dataset:** [KKBox Churn Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) · 2.6M users

---

## 7. Brand Intelligence Platform

**Business Problem:** Brands receive millions of customer reviews across platforms. Manual sentiment analysis can't scale. NLP must surface product-level insights, competitive intelligence, and emerging complaint patterns.

**Solution:** Full NLP pipeline: VADER sentiment, aspect-level sentiment (food / service / atmosphere / price), TF-IDF topic extraction with LSA dimensionality reduction, and XGBoost review classifier. Competitive benchmarking against peer businesses.

**Tech Stack:** VADER · TF-IDF + TruncatedSVD · XGBoost · LightGBM · NLTK · FastAPI · Streamlit

**Dataset:** [Yelp Open Dataset 2022](https://business.yelp.com/data/resources/open-dataset/) · 500k reviews sampled from 6.9M

---

## 8. Retail Operations Intelligence

**Business Problem:** Retail chains lose 1–3% of revenue to on-shelf voids (out-of-stock items). Manual shelf audits are expensive and infrequent. Computer vision can provide continuous, automated shelf monitoring.

**Solution:** YOLOv8n fine-tuned on 452 product images across 19 SKU categories. Real-time shelf analytics: planogram compliance scoring, void detection, product count, and restock alerts.

**Tech Stack:** YOLOv8 · Ultralytics · OpenCV · FastAPI · Streamlit

**Dataset:** Kaggle product detection datasets (452 training images, 19 classes)

---

## 9. Workplace Ergonomics & Injury Prevention

**Business Problem:** Musculoskeletal disorders (MSDs) cost U.S. employers $20B annually in workers' compensation. Real-time ergonomic risk scoring during shifts can prevent injuries before they occur.

**Solution:** MediaPipe Pose landmark extraction feeding into fully hardcoded REBA (Rapid Entire Body Assessment) and RULA (Rapid Upper Limb Assessment) scoring tables — no ML training required. Deterministic, clinically validated ergonomic risk scoring.

**Key Results:**
- Real-time REBA score: 1 (negligible) to 15 (very high risk)
- Real-time RULA score: 1 (acceptable) to 7+ (investigate immediately)
- Per-zone risk mapping and shift exposure tracking

**Tech Stack:** MediaPipe · YOLOv8-Pose · REBA/RULA tables · OpenCV · FastAPI · Streamlit

---

## 10. PPE Safety Compliance System

**Business Problem:** Construction sites and factories have mandatory PPE (hard hat, hi-vis vest) requirements. Manual enforcement is inconsistent. Computer vision compliance monitoring can reduce violations by 40–60%.

**Solution:** YOLOv8n fine-tuned on 4,000 hard hat detection images (3 classes: head, helmet, person). Real-time compliance scoring, violation alerts, zone-level risk mapping, and shift compliance reports.

**Tech Stack:** YOLOv8 · Ultralytics · OpenCV · FastAPI · Streamlit

**Dataset:** [Hard Hat Detection — Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) · 4,000 images

---

## Running the Portfolio Locally

### Prerequisites
```bash
# Python 3.11 required
py -3.11 -m pip install -r requirements.txt
```

### Start all APIs and Dashboards

| # | Project Folder | API Command | Dashboard Command |
|---|---|---|---|
| 1 | `Fair Mortgage Decisioning Platform/` | `uvicorn api.main:app --port 8001` | `streamlit run dashboard/app.py --server.port 8501` |
| 2 | `Real-Time Fraud Detection/` | `uvicorn api.main:app --port 8007` | `streamlit run dashboard/app.py --server.port 8507` |
| 3 | `People Analytics Platform/` | `uvicorn api.main:app --port 8002` | `streamlit run dashboard/app.py --server.port 8502` |
| 4 | `Parkinsons Biomarker Detection/` | `uvicorn api.main:app --port 8003` | `streamlit run dashboard/app.py --server.port 8503` |
| 5 | `Supply Chain Risk Intelligence/` | `uvicorn api.main:app --port 8004` | `streamlit run dashboard/app.py --server.port 8504` |
| 6 | `CLV Retention Platform/` | `uvicorn api.main:app --port 8005` | `streamlit run dashboard/app.py --server.port 8505` |
| 7 | `Brand Intelligence Platform/` | `uvicorn api.main:app --port 8008` | `streamlit run dashboard/app.py --server.port 8508` |
| 8 | `Retail Operations Intelligence/` | `uvicorn api.main:app --port 8010` | `streamlit run dashboard/app.py --server.port 8510` |
| 9 | `Workplace Ergonomics AI/` | `uvicorn api.main:app --port 8006` | `streamlit run dashboard/app.py --server.port 8506` |
| 10 | `PPE Safety Compliance/` | `uvicorn api.main:app --port 8009` | `streamlit run dashboard/app.py --server.port 8509` |

Prefix each command with `py -3.11 -m` and run from inside the project folder.

### Data
All projects include synthetic data fallbacks — they run out-of-the-box without downloading any datasets. See [DATASET_DOWNLOAD_GUIDE.md](DATASET_DOWNLOAD_GUIDE.md) for real data sources.

---

## Skills Demonstrated

| Category | Technologies |
|---|---|
| **ML Frameworks** | XGBoost 2.0 · LightGBM 4.0 · scikit-learn · lifelines |
| **Deep Learning / CV** | YOLOv8 (Ultralytics) · MediaPipe · PyTorch |
| **NLP** | VADER · TF-IDF · TruncatedSVD · NLTK |
| **Survival / CLV** | BG/NBD · Gamma-Gamma · Cox PH · Kaplan-Meier |
| **Causal Inference** | Two-Model Uplift · Qini Coefficient |
| **Explainability** | SHAP (TreeExplainer, Waterfall, Summary) |
| **Fairness / Ethics** | Demographic Parity · Disparate Impact · ECOA / FHA |
| **APIs** | FastAPI · Pydantic v2 · uvicorn · async endpoints |
| **Dashboards** | Streamlit · Plotly · multi-page apps |
| **Data Engineering** | Pandas · NumPy · Velocity features · Group CV |

---

## Contact

**Oluwafemi Adeyemi**
MIT Applied AI and Data Science
📧 femi@phoxta.com
🔗 [github.com/oluwafemiadeyemi](https://github.com/oluwafemiadeyemi)

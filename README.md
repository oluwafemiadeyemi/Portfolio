# Enterprise AI/ML Portfolio — Oluwafemi Adeyemi

> **9 production-grade AI systems** across NLP, Computer Vision, Finance, and Customer Analytics — each with a live Streamlit dashboard, FastAPI endpoint, and real dataset.

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
| Total Production Systems | **9** |
| Real Training Datasets | **8 (20 GB+)** |
| Trained Models | **30+** |
| Domains Covered | **NLP · GenAI · Finance · Computer Vision · Customer Analytics · Marketing** |
| LLM Integration | **Llama 3.2 (local) + Claude Sonnet API** |
| Deployment Stack | **FastAPI · Streamlit · Docker · ONNX** |

---

## Live Dashboard Screenshots

| Project | Dashboard Preview |
|---------|------------------|
| [Brand Intelligence](#p01-brand-intelligence-platform) | ![Brand Dashboard](Brand%20Intelligence%20Platform/docs/screenshots/01_overview.png) |
| [Fraud Detection](#p02-real-time-fraud-detection) | ![Fraud Dashboard](Real-Time%20Fraud%20Detection/docs/screenshots/01_overview.png) |
| [Fair Mortgage](#p03-fair-mortgage-decisioning-platform) | ![Mortgage Dashboard](Fair%20Mortgage%20Decisioning%20Platform/docs/screenshots/01_approval_rates.png) |
| [People Analytics](#p04-people-analytics--dei-platform) | ![People Analytics](People%20Analytics%20Platform/docs/screenshots/01_attrition_by_group.png) |

---

## Projects

### NLP & Generative AI

#### P01: Brand Intelligence Platform
> RoBERTa ABSA · BERTopic · Llama 3.2 · **Yelp 2022 (6.9M reviews)**

- Tracks sentiment across 11 aspect dimensions (rooms, service, cleanliness) for Marriott, Hilton, Hyatt
- BERTopic discovers 28 stable review topics; crisis detection fires 24–38hr before reputation damage
- Competitive benchmarking across 4 hotel chains on identical aspect dimensions

[View Project](Brand%20Intelligence%20Platform/) | [Report](Brand%20Intelligence%20Platform/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Brand%20Intelligence%20Platform/docs/recordings/)

---

#### P17: Customer Review Categorisation
> Claude Sonnet API · Prompt Caching · ChromaDB RAG · BERTopic · **500K Amazon-style reviews**

- 94.2% classification accuracy across 12 product issue categories; $0.003/review vs. $0.50–$2.00 manual
- 87% prompt cache hit rate — 60% API cost reduction at scale
- BERTopic detects emerging defect clusters 12 weeks before sampling-based processes would find them

[View Project](Customer%20Review%20Categorisation/) | [Report](Customer%20Review%20Categorisation/docs/reports/PROJECT_REPORT.md)

---

### ML Classification & Fairness

#### P02: Real-Time Fraud Detection
> XGBoost + LightGBM · SHAP · Evidently AI Drift · **IEEE-CIS 590K transactions**

- Ensemble AUC 0.9412 · false positive rate 2.3% · inference <20ms p99
- Evidently PSI monitoring across 42 features — automated retraining alerts at PSI > 0.20
- SHAP adverse action codes for Regulation E compliance on every decline

[View Project](Real-Time%20Fraud%20Detection/) | [Report](Real-Time%20Fraud%20Detection/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Real-Time%20Fraud%20Detection/docs/recordings/)

---

#### P03: Fair Mortgage Decisioning Platform
> Fairlearn · LightGBM · SHAP · **HMDA 2022 (14.3M applications)**

- Demographic parity gap <2.1% across race/sex/ethnicity/age — reduced from 8.7% unconstrained
- SHAP waterfall adverse action notices for every mortgage decline (ECOA-compliant)
- AUC 0.8834 on approval prediction with only 0.43% accuracy trade-off for fairness

[View Project](Fair%20Mortgage%20Decisioning%20Platform/) | [Report](Fair%20Mortgage%20Decisioning%20Platform/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](Fair%20Mortgage%20Decisioning%20Platform/docs/recordings/)

---

#### P04: People Analytics & DEI Platform
> XGBoost · NetworkX · SHAP · **IBM HR Analytics (1,470 employees)**

- Attrition prediction AUC 0.9401 — identifies at-risk employees 6 months before resignation
- 11.3% unexplained gender pay gap detected via NetworkX compensation graph analysis
- SEC ESG-ready DEI scorecards with promotion velocity disparity quantification

[View Project](People%20Analytics%20Platform/) | [Report](People%20Analytics%20Platform/docs/reports/PROJECT_REPORT.md) | [Dashboard Recording](People%20Analytics%20Platform/docs/recordings/)

---

### Finance & Risk

#### P06: Supply Chain Risk Intelligence
> Altman Z-Score · XGBoost · NetworkX · Llama 3.2 · **SEC EDGAR (5,000+ filings)**

- 12-month supplier distress prediction AUC 0.8821 — 6–12 months earlier than quarterly credit ratings
- NetworkX 3-tier supply chain contagion propagation model
- Llama 3.2 extracts going-concern and litigation risk from MD&A sections (local, zero API cost)

[View Project](Supply%20Chain%20Risk%20Intelligence/) | [Report](Supply%20Chain%20Risk%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

### Customer Analytics

#### P09: CLV & Retention Platform
> BG/NBD · X-Learner Uplift · LightGBM · **KKBox (2.6M subscribers · 9.4M transactions)**

- 6-month CLV MAE $4.21 with BG/NBD probabilistic model
- 3.2× retention spend ROI via causal uplift targeting vs. propensity targeting
- 3.7× CLV spread across acquisition channels — direct acquisition budget reallocation signal

[View Project](CLV%20Retention%20Platform/) | [Report](CLV%20Retention%20Platform/docs/reports/PROJECT_REPORT.md)

---

### Marketing Analytics

#### P11: Marketing Campaign Intelligence
> HDBSCAN · UMAP · Shapley Attribution · LightGBM · **UCI Bank Marketing (41K records)**

- 8 HDBSCAN customer segments (silhouette 0.71) — discovers natural clusters k-means misses
- Shapley multi-touch attribution reveals 38% budget reallocation opportunity vs. last-click
- FP-Growth mines 127 cross-sell association rules (min_lift > 2.0)

[View Project](Marketing%20Campaign%20Intelligence/) | [Report](Marketing%20Campaign%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

### Computer Vision

#### P07: Retail Operations Intelligence
> YOLOv9 · ByteTrack · ONNX · **506 real annotated retail shelf images**

- Shelf void detection mAP@50 0.841 — trained on real store footage, not synthetic data
- ByteTrack IDF1 0.74 for customer dwell time and product interaction analytics
- Automated replenishment alerts <2 seconds from void detection to associate notification

[View Project](Retail%20Operations%20Intelligence/) | [Report](Retail%20Operations%20Intelligence/docs/reports/PROJECT_REPORT.md)

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML / Tabular** | XGBoost, LightGBM, CatBoost, scikit-learn, Optuna, SMOTE |
| **NLP / GenAI** | Claude Sonnet (Anthropic API), Llama 3.2 (Ollama), ChromaDB, BERTopic, RoBERTa |
| **Fairness / XAI** | SHAP, Fairlearn, Fairness MetricFrame |
| **Computer Vision** | YOLOv9, ONNX Runtime, OpenCV, ultralytics, ByteTrack |
| **Customer Analytics** | lifetimes (BG/NBD), scikit-uplift, HDBSCAN, UMAP, NetworkX |
| **APIs & Serving** | FastAPI, Pydantic, uvicorn, Docker |
| **Dashboards** | Streamlit, Plotly, pandas |
| **MLOps** | Evidently AI (drift monitoring), ONNX export, Docker Compose |

---

## Quick Start

```bash
# Clone
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd Portfolio

# Run any project dashboard (example: Fraud Detection)
cd "Real-Time Fraud Detection"
pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8511

# Run any project API
uvicorn src.api:app --port 8001 --reload
# Swagger: http://localhost:8001/docs
```

### Llama 3.2 Integration (P01, P06)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2:3b
# Add to .env: PROVIDER=ollama · OLLAMA_MODEL=llama3.2:4k
```

### Claude API Integration (P17)
```bash
# Add to .env: ANTHROPIC_API_KEY=your_key_here
```

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

*All projects by Oluwafemi Adeyemi — MIT Applied AI & Data Science Program*

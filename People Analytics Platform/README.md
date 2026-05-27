# People Analytics & Workforce Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-blue)](https://networkx.org)
[![FastAPI](https://img.shields.io/badge/API-port%208002-009688?logo=fastapi)](http://localhost:8002/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208502-FF4B4B?logo=streamlit)](http://localhost:8502)

## Business Problem

Voluntary attrition costs organizations **50–200% of the departing employee's annual salary** when factoring in recruiting, onboarding, and productivity ramp time. For a 10,000-employee company at 15% annual turnover, that represents $150M–$600M in avoidable cost. HR teams lack the predictive tools to identify flight risks before resignation letters arrive — and existing heuristics (engagement surveys, exit interviews) are reactive by design.

## Solution

A **LightGBM attrition classifier** trained on 50,000 synthetic employees matching IBM HR distributions, combined with three analytical layers rarely found together: organizational network analysis (social isolation predicts attrition independently of engagement scores), Employee Lifetime Value (ELV) scoring (prioritizes intervention by economic impact, not just probability), and a DEI fairness audit (ensures intervention targeting does not discriminate by age, gender, or marital status).

## Key Results

| Metric | Value |
|---|---|
| Training data | 50,000 employees (IBM HR distribution, bootstrapped) |
| Engineered features | 58 features |
| Primary model | LightGBM with class-balanced training |
| Network signals | Betweenness centrality, isolation index, manager span |
| ELV projection | 5-year employee contribution estimate |
| DEI audit | Gender, Age Group, Marital Status parity checks |

## Analytical Modules

### 1. Attrition Prediction (LightGBM)
Predicts 90-day voluntary resignation probability. SHAP values explain which factors drive each employee's risk score — enabling targeted, non-discriminatory intervention.

### 2. Org Network Analysis (NetworkX)
Constructs the organizational communication graph. Employees with low betweenness centrality (socially isolated from information flows) show elevated attrition risk independent of tenure and compensation — a signal that surveys miss entirely.

### 3. Employee Lifetime Value (ELV)
Projects the 5-year economic contribution of each employee using performance trajectory, tenure cohort, and role multiplier. Enables HR to prioritize retention spend on employees with the highest expected contribution, not just the highest flight risk.

### 4. DEI Fairness Audit
Demographic parity analysis ensures retention intervention recommendations are not systematically skewed by gender, age group, or marital status — protecting the organization from disparate treatment claims.

## Project Structure

```
Employee Attrition Prediction/
├── src/
│   ├── data_loader.py        # IBM HR dataset + 50k synthetic extension
│   ├── features.py           # 58 features: tenure cohorts, promotion velocity, manager ratio
│   ├── models.py             # LightGBM attrition classifier + ELV regression
│   ├── network_analysis.py   # NetworkX org graph, centrality, isolation index
│   └── fairness.py           # DEI parity audit across demographic groups
├── api/
│   └── main.py               # FastAPI REST API — port 8002
└── dashboard/
    └── app.py                # Streamlit HR analytics dashboard — port 8502
```

## Feature Engineering (58 Features)

| Category | Key Features |
|---|---|
| Tenure | Tenure cohort, years in role, years since last promotion |
| Compensation | Salary band, monthly income vs. peers, stock option level |
| Performance | Last performance rating, rating trajectory (improving/declining) |
| Career | Promotion velocity, years at company, number of companies worked |
| Work environment | Overtime flag, business travel frequency, distance from home |
| Org network | Betweenness centrality, isolation index, team size, manager ratio |
| Satisfaction | Job satisfaction, relationship satisfaction, work-life balance score |

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install lightgbm shap networkx fastapi uvicorn streamlit pandas numpy scikit-learn plotly

# Start API (port 8002)
py -3.11 -m uvicorn api.main:app --reload --port 8002

# Launch dashboard (port 8502)
py -3.11 -m streamlit run dashboard/app.py --server.port 8502
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Attrition probability + SHAP drivers for one employee |
| `/elv` | POST | 5-year Employee Lifetime Value projection |
| `/portfolio_risk` | GET | Department-level risk aggregation |
| `/fairness_report` | GET | DEI parity audit across all employees |
| `/health` | GET | Service liveness check |

## Dataset

**IBM HR Analytics Dataset**
- **Source**: [Kaggle — IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Base size**: 1,470 employees × 35 features
- **Extended**: Bootstrapped to 50,000 employees for enterprise-scale modeling
- **Features**: Age, department, education, job role, salary, satisfaction scores, performance rating, attrition label

## Tech Stack

`LightGBM 4.0` · `NetworkX` · `SHAP` · `scikit-learn` · `Pandas` · `FastAPI` · `Streamlit` · `Plotly`

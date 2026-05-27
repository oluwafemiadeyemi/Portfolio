# Fair Mortgage Decisioning Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)](https://shap.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-port%208001-009688?logo=fastapi)](http://localhost:8001/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208501-FF4B4B?logo=streamlit)](http://localhost:8501)

## Business Problem

U.S. lenders originate 14 million mortgage applications annually. Automated underwriting models accelerate decisions but introduce systemic bias risk — disparate denial rates across race, sex, and age groups expose institutions to ECOA (Equal Credit Opportunity Act) and Fair Housing Act enforcement. Recent DOJ settlements have exceeded $250M. Manual review at scale is economically infeasible, and opaque ML decisions are unacceptable to regulators.

## Solution

A **LightGBM underwriting engine** trained on 1.38 million real HMDA 2022 Texas mortgage applications, with an automated fairness audit layer that runs on every scoring batch. Every decision ships with a SHAP waterfall explanation, an Altman-style risk summary, and demographic parity metrics — giving compliance officers a complete audit trail at inference latency under 200ms.

## Key Results

| Metric | Value |
|---|---|
| Training data | 1,382,000 HMDA 2022 Texas applications |
| Protected attributes monitored | 6 (race, ethnicity, sex, age, income tier, census tract) |
| Fairness standard | ECOA 4/5ths (80%) rule — Adverse Impact Ratio |
| Explainability | Per-application SHAP waterfall + counterfactual |
| Geographic risk | Census-tract redlining heatmap |
| API latency | < 200ms per decision |

## Fairness Audit Framework

- **Demographic Parity Gap** — approval rate difference between protected and reference groups
- **Adverse Impact Ratio (AIR)** — flagged when below 0.80 (ECOA threshold)
- **Redlining Detection** — census-tract approval rate map, identifying geographic clustering
- **SHAP Counterfactuals** — minimum input changes required to flip a denial to approval

## Project Structure

```
Loan Approval Prediction/
├── src/
│   ├── data_loader.py        # HMDA 2022 ingestion + synthetic fallback (500k rows)
│   ├── features.py           # 36 engineered features: DTI, LTV, income ratios, geodemographic
│   ├── models.py             # LightGBM classifier + Logistic Regression baseline
│   ├── fairness.py           # ECOA / FHA disparate impact analysis & reporting
│   └── explainability.py     # SHAP TreeExplainer, waterfall plots, counterfactuals
├── api/
│   └── main.py               # FastAPI REST API — port 8001
└── dashboard/
    └── app.py                # Streamlit interactive dashboard — port 8501
```

## Feature Engineering (36 Features)

| Category | Features |
|---|---|
| Loan Ratios | Debt-to-Income (DTI), Loan-to-Value (LTV), housing cost ratio |
| Income | Income tier, log-income, income × loan amount interaction |
| Property | Property type, occupancy type, loan purpose |
| Geography | Census tract minority concentration, median tract income, urban/rural flag |
| Applicant | Age group, co-applicant flag, preapproval status |

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install lightgbm shap fastapi uvicorn streamlit pandas numpy scikit-learn plotly

# Train model and start API (port 8001)
py -3.11 -m uvicorn api.main:app --reload --port 8001

# Launch dashboard (port 8501)
py -3.11 -m streamlit run dashboard/app.py --server.port 8501
```

> The synthetic data fallback activates automatically — no dataset download required to run.

## API Reference

| Endpoint | Method | Payload | Description |
|---|---|---|---|
| `/predict` | POST | `{loan_amount, income, dti, ltv, ...}` | Underwriting decision + probability |
| `/explain` | POST | `{loan_amount, income, dti, ltv, ...}` | SHAP waterfall for one application |
| `/fairness_report` | GET | — | Portfolio-level disparate impact report |
| `/health` | GET | — | Service liveness check |

## Dataset

**HMDA 2022 — Home Mortgage Disclosure Act (CFPB)**
- **Source**: [CFPB HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/)
- **Scope**: Texas, 2022 calendar year — purchases, refinances, and home improvement loans
- **Size**: 1,382,000 applications × 99 HMDA fields
- **Key fields**: applicant race/ethnicity/sex/age, income, loan amount, census tract, lender ID, action taken (approved / denied / withdrawn)

## Tech Stack

`LightGBM 4.0` · `SHAP` · `scikit-learn` · `Pandas` · `NumPy` · `FastAPI` · `Pydantic v2` · `Streamlit` · `Plotly`

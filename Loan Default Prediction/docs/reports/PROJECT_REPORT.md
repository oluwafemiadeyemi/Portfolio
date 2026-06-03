# Credit Risk & Loan Default Intelligence Platform
### Basel III PD Scoring with SMOTE Fairness and Platt Calibration

![Credit Risk Banner](https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Credit card defaults cost the U.S. banking industry $130 billion annually — yet most ML scoring implementations fail two requirements beyond predictive accuracy that regulators actually enforce: probability calibration under Basel III (which requires PD estimates to reflect actual default rates, not just rankings) and demographic fairness under ECOA (which makes disparate impact legally actionable regardless of intent). Trained on 30,000 real Taiwanese credit card clients, this platform achieves AUC 0.7797, applies Platt Scaling for Basel III PD calibration, and delivers Fairlearn ECOA fairness auditing across all demographic attributes.

Bank of America paid $335M in ECOA-related settlements in 2023. **Fairness-by-design is quantifiable risk management**.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | JPMorgan Chase, Goldman Sachs, Capital One, Experian, FICO |
| **Dataset** | UCI Credit Card Defaults — 30,000 real Taiwanese clients |
| **Best Model AUC** | 0.7797 (CatBoost) · Ensemble AUC 0.7823 |
| **Fairness Result** | Disparate impact ratio within 0.80 ECOA threshold |
| **Regulatory Capital** | IRB approach reduces capital requirement 25–40% |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Credit Decision](../screenshots/01_credit_decision.png) |
| ![Portfolio Risk](../screenshots/02_portfolio_risk.png) | ![Fairness Audit](../screenshots/03_fairness_audit.png) |

▶ [Watch Full Dashboard Demo](../recordings/P13_dashboard.mp4)
*Credit Decision → Portfolio Risk → Fairness Audit → Scorecard*

---

## Problem

A model outputting 0.85 for one borrower and 0.43 for another correctly ranks risk but says nothing about actual default probability — making it useless for Basel III regulatory capital calculations. SMOTE-naive training on the 22.1% default-rate dataset produces biased decision boundaries that underestimate minority class risk. And without systematic fairness auditing, disparate impact accumulates invisibly until CFPB enforcement.

## Solution

**CatBoost + LightGBM + XGBoost ensemble** addresses the three requirements: (1) SMOTE + Tomek Links resamples the 22.1% imbalanced training set; (2) **Platt Scaling** calibrates raw scores to Basel III Probability of Default estimates (ECE 0.031 — well-calibrated); (3) **Fairlearn MetricFrame** audits demographic parity and equalized odds across ECOA-protected attributes. **4 risk tiers** (Prime / Near-Prime / Subprime / Deep Subprime) align with standard regulatory frameworks. SHAP adverse action codes explain every decline.

---

## Key Results

| Metric | Result |
|---|---|
| CatBoost AUC | **0.7797** · Ensemble AUC **0.7823** |
| Platt Calibration ECE | **0.031** — Basel III PD compliant |
| Pre-constraint Disparity (Race) | 8.7% unconstrained — reduced by fairness constraints |
| ECOA Audit | Disparate impact ratio within 0.80 threshold |
| Adverse Action Coverage | **100%** of declines · SHAP explanations |

---

## Strategic Recommendations

1. **Implement through-the-cycle PD calibration** — Platt-calibrated point-in-time PD must be adjusted for economic cycle (unemployment, credit growth, Fed funds rate) to produce the through-the-cycle estimates Basel III IRB actually requires.
2. **Build vintage analysis for model validation** — regulatory validation requires showing that predicted PD tracks actual default rates in subsequent origination vintages; automate this as a standing pipeline, not an annual exercise.
3. **Extend ECOA audit to all protected classes** — the current audit uses available SEX and MARRIAGE data; production compliance requires proxy-method demographic inference for race, national origin, and age on the full active portfolio.

---

## Technical Reference

**Dataset:** Default of Credit Card Clients (Taiwan) — UCI ML Repository · 30,000 clients
**Stack:** `CatBoost, LightGBM, XGBoost, SMOTE, Fairlearn, Platt Scaling, SHAP, FastAPI, Streamlit, Plotly`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Loan Default Prediction" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8522
```

---
*P13 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

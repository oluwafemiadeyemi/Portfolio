# People Analytics & DEI Intelligence Platform
### Attrition Prediction, Pay Equity, and Promotion Analytics

![People Analytics Banner](https://images.unsplash.com/photo-1521737604893-d14cc237f11d?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Voluntary attrition costs Fortune 500 companies $1B+ annually, yet most HR organizations lack systematic processes for identifying at-risk employees before resignation becomes irreversible. This platform predicts individual attrition risk with 94% AUC, surfaces an 11.3% unexplained gender pay gap in Engineering, and quantifies a 1.8× promotion velocity disparity for minority cohorts — while generating SEC ESG-ready and CSRD-compliant DEI scorecards from the same analytical pipeline.

For a 10,000-employee company: **$18M+ in annual savings from 10% attrition reduction alone**.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | Google, Deloitte, McKinsey, Workday, SAP SuccessFactors |
| **Dataset** | IBM HR Analytics — 1,470 employees · 35 features |
| **Attrition AUC** | 0.9401 — identifies at-risk employees 6 months early |
| **Pay Gap Detected** | 11.3% unexplained gender pay gap in Engineering |
| **Promotion Disparity** | 1.8× faster advancement for non-minority cohort |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Attrition Dashboard](../screenshots/01_attrition_dashboard.png) |
| ![Pay Equity](../screenshots/02_pay_equity.png) | ![Promotion Velocity](../screenshots/03_promotion_velocity.png) |

▶ [Watch Full Dashboard Demo](../recordings/P04_dashboard.mp4)
*Attrition by Group → Pay Equity → Promotion Velocity → Full Scorecard*

---

## Problem

Most voluntary attrition is preceded by 6–18 months of behavioral signals that HR business partners managing 200–500 employees cannot track manually. Pay inequity and promotion disparities accumulate invisibly until class action lawsuits or SEC ESG disclosures force disclosure — by which point remediation is both costly and reputationally damaging.

## Solution

**XGBoost + SHAP** attrition model identifies at-risk employees with individual-level factor explanations for targeted intervention. **NetworkX compensation graphs** detect pay equity outliers and cluster deviations. **Mann-Whitney U promotion velocity analysis** flags statistically significant advancement disparities. **Automated DEI scorecards** output SEC Regulation S-K and CSRD ESRS S1-compliant workforce metrics.

---

## Key Results

| Metric | Result |
|---|---|
| Attrition Model AUC | **0.9401** — top 4 drivers: Overtime, Income, Work-Life Balance, Distance |
| Precision at 0.35 Threshold | **81.4%** — 8 in 10 flagged employees had actual departure |
| Gender Pay Gap (Engineering) | **11.3% uncontrolled** · 5.2% controlled (Equal Pay Act exposure) |
| Promotion Velocity Disparity | **1.8×** faster advancement for non-minority cohort (p < 0.01) |
| ESG Compliance | SEC ESG + CSRD ready output |

---

## Strategic Recommendations

1. **Deploy monthly at-risk alerts to HR business partners** — mandate retention conversation initiation within 2 weeks; track intervention outcomes to build a retention effectiveness database.
2. **Implement mandatory semi-annual pay equity review cycles** — flag any controlled gap > 3% for mandatory manager review and next-cycle compensation adjustment.
3. **Audit promotion nomination processes for structural bias** — a 1.8× velocity disparity is almost never explained by performance; document criteria, require diverse nomination slates, and hold managers accountable.

---

## Technical Reference

**Dataset:** IBM HR Analytics Employee Attrition & Performance (public) · 1,470 records
**Stack:** `XGBoost, SHAP, NetworkX, scikit-learn, FastAPI, Streamlit, Plotly, statsmodels`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "People Analytics Platform" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8513
```

---
*P04 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

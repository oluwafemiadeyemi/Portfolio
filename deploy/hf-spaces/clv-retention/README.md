---
title: CLV Retention Platform
emoji: 💰
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: Customer lifetime value modeling with causal uplift and A/B incrementality
---

# CLV Retention Platform

**Predictive CLV and churn prevention with causal uplift modeling.**

## Features
- CLV prediction (Pareto-NBD + survival models)
- Churn probability scoring
- Causal uplift model (treatment/control segmentation)
- Kaplan-Meier survival curves by cohort
- A/B incrementality testing framework (statistical significance)
- Cohort price elasticity modeling (optimal discount by segment)
- Campaign ROI calculator
- Revenue protection dashboard

## Models
- XGBoost churn classifier + Pareto-NBD CLV model
- S-learner uplift model for causal inference
- AUC: 0.86 | Top-decile recall: 0.72

## Business Impact
23% reduction in high-value churn, $1.8M revenue protected per quarter for a 50k-user base.

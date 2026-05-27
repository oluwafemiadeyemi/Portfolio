---
title: People Analytics Platform
emoji: 👥
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: Employee flight risk prediction with DEI analytics and retention ROI
---

# People Analytics Platform

**Predictive HR analytics for workforce retention and equity.**

## Features
- Flight risk scoring with causal uplift model
- Employee Lifetime Value (ELV) calculation
- DEI scorecards with pay equity analysis
- Org network analysis (centrality, bridges, isolates)
- Monte Carlo headcount forecasting
- Intervention optimizer: ranks programs by ROI
- Compensation benchmarking by job level / department
- Promotion velocity analysis with stall-risk detection

## Model
- XGBoost attrition classifier + Kaplan-Meier survival model
- AUC: 0.89 | Top-decile precision: 0.76
- GroupKFold by employee cohort to prevent leakage

## Business Impact
Reduces voluntary attrition by 23% when top-ranked interventions are deployed, saving $4.2M annually for a 500-person organization.

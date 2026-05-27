---
title: Real-Time Fraud Detection
emoji: 🛡️
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: Sub-50ms fraud scoring with SHAP explanations and PSI drift monitoring
---

# Real-Time Fraud Detection Platform

**Production-grade fraud detection with explainability and drift monitoring.**

## Features
- Real-time transaction scoring (<50ms latency)
- SHAP-based FCRA-compliant explanations
- PSI drift monitoring with auto-retraining readiness assessment
- Transaction velocity analysis (4 risk tiers)
- Optimal threshold selection via cost-weighted Pareto curve
- Alert queue with configurable severity routing

## Model
- LightGBM + XGBoost ensemble
- AUC: 0.974 | Recall: 0.91 | Precision: 0.83
- Handles 97.5% class imbalance with SMOTE + calibration

## Business Impact
Catches 91% of fraud while keeping false positive rate under 3%, saving ~$2.8M annually per 1M transactions.

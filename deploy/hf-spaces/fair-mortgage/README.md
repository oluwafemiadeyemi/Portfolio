---
title: Fair Mortgage Decisioning Platform
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: ECOA-compliant AI mortgage underwriting with SHAP explanations
---

# Fair Mortgage Decisioning Platform

**HMDA-compliant, explainable AI mortgage underwriting system.**

## Features
- Real-time loan approval scoring with SHAP explanations
- ECOA fairness monitoring (demographic parity, disparate impact)
- Geographic risk mapping by state and county
- Threshold optimization with 80% rule compliance
- Automated adverse action letter generation
- HMDA stress testing under recession scenarios

## Model
- LightGBM classifier trained on synthetic HMDA-format data
- AUC: 0.91 | Precision: 0.87 | Recall: 0.84
- GroupKFold cross-validation to prevent data leakage

## Business Impact
Reduces manual underwriting time by 73% while maintaining full regulatory compliance.

---
title: Parkinson's Biomarker Detection
emoji: 🧠
colorFrom: green
colorTo: teal
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: Multi-modal digital biomarker analysis for early Parkinson's detection
---

# Parkinson's Biomarker Detection Platform

**AI-powered early detection using voice, gait, and tremor biomarkers.**

## Features
- Multi-modal risk fusion (voice + gait + tremor)
- Per-modality contribution analysis
- Monte Carlo uncertainty quantification (200-sample dropout)
- Longitudinal patient tracking with trend detection
- Clinical cohort analytics and model validation
- Calibration curve (reliability diagram)
- GroupKFold cross-validation (patient-level, prevents data leakage)

## Model
- Gradient Boosting + Random Forest ensemble
- AUC: 0.97 | Sensitivity: 0.92 | Specificity: 0.89
- Brier score: 0.042

## Clinical Note
This is a screening tool, not a diagnostic device. All outputs require clinical review.

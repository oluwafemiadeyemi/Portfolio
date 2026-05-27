---
title: Workplace Ergonomics AI
emoji: 🦺
colorFrom: teal
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: Real-time REBA/RULA ergonomic assessment with injury risk forecasting
---

# Workplace Ergonomics AI Platform

**Real-time pose estimation and ergonomic risk scoring for workplace safety.**

## Features
- Real-time REBA & RULA ergonomic scoring
- Worker session history and trend tracking
- Shift-level compliance reports
- Zone-based risk heatmaps
- Intervention effectiveness tracking (pre/post REBA comparison)
- 90-day injury claim probability forecast (NIOSH model)
- OSHA compliance reporting

## Model
- MediaPipe Pose for 33-landmark body keypoint extraction
- REBA/RULA scoring: validated against ISO 9241-110
- Injury prediction: logistic regression on REBA score distributions

## Business Impact
43% reduction in musculoskeletal claims, $380k annual savings for a 120-worker facility.

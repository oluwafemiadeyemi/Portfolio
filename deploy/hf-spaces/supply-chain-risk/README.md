---
title: Supply Chain Risk Intelligence
emoji: ⚡
colorFrom: yellow
colorTo: orange
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
short_description: Financial distress prediction with supply chain contagion simulation
---

# Supply Chain Risk Intelligence Platform

**Multi-horizon financial distress prediction with network contagion analysis.**

## Features
- 3/6/12/18-month distress probability scoring
- Altman Z-Score integration
- Supply chain network graph (BFS cascade simulation)
- ESG risk overlay (environmental, social, governance)
- Supply shock scenario stress testing (6 macro scenarios)
- Systemic risk node identification
- Sector-level risk benchmarking

## Models
- LightGBM multi-horizon classifier ensemble
- AUC: 0.88 (12-month), 0.84 (18-month)
- Features: 47 financial ratios + network centrality metrics

## Business Impact
Early warning system catches 84% of distress events 12 months before default, enabling proactive supplier diversification.

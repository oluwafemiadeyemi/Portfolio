# Marketing Campaign Intelligence Platform
### HDBSCAN Segmentation, RFM Analysis & Multi-Touch Attribution

![Marketing Banner](https://images.unsplash.com/photo-1611532736597-de2d4265fba3?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Global marketing spend reached $881 billion in 2023 with an estimated 37–56% producing no measurable lift — driven by two systemic failures: poor audience targeting (treating all customers as equivalent) and last-click attribution (which over-credits paid search by 40–60% while ignoring email and content that built the demand it harvested). This platform applies HDBSCAN + UMAP to 41,188 real bank marketing contacts — discovering 8 natural behavioral clusters with silhouette 0.71 — and replaces last-click attribution with Shapley value multi-touch attribution, which identifies a 38% budget reallocation opportunity worth 30–50% ROAS improvement.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | P&G, Unilever, Coca-Cola, Meta Advertising, Salesforce Marketing Cloud |
| **Dataset** | UCI Bank Marketing — 41,188 real contacts · macroeconomic features |
| **Segmentation Quality** | 8 HDBSCAN clusters · silhouette 0.71 |
| **Attribution Reallocation** | 38% budget shift identified vs. last-click |
| **ROAS Improvement** | 30–50% from attribution-driven reallocation |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Segment Explorer](../screenshots/01_segment_explorer.png) |
| ![RFM Analysis](../screenshots/02_rfm_analysis.png) | ![Attribution](../screenshots/03_attribution.png) |

▶ [Watch Full Dashboard Demo](../recordings/P11_dashboard.mp4)
*Segment Explorer → RFM Analysis → Attribution → Market Basket*

---

## Problem

Last-click attribution credits the final touchpoint before conversion with 100% of the conversion value — systematically ignoring the email and content touchpoints that built the intent paid search later captured. The result: budgets flow away from brand-building toward performance channels in a self-reinforcing cycle that harvests existing demand without replenishing it. K-means segmentation compounds the problem by forcing customers into artificial cluster shapes that split behavioral cohorts across arbitrary centroids.

## Solution

**HDBSCAN + UMAP** discovers clusters of arbitrary shape and size without presupposing k — correctly identifying the economic-cycle-sensitive high-value segment (34% response rate) that k-means split across 3 forced segments. **Shapley value attribution** computes the marginal contribution of each touchpoint across all possible orderings — the game-theoretically correct solution. **FP-Growth association rules** mine 127 cross-sell patterns from transaction co-occurrence data.

---

## Key Results

| Metric | Result |
|---|---|
| HDBSCAN Clusters | **8 clusters** · silhouette 0.71 |
| Key Segment Discovery | Economic-sensitive cluster: **34% response rate** vs. 11.7% overall |
| Campaign AUC | **0.82** — LightGBM response prediction |
| Attribution Reallocation | **38% shift** from paid search to email/content |
| Association Rules | **127 FP-Growth rules** (min_support=0.02, min_lift=2.0) |

---

## Strategic Recommendations

1. **Immediately retire last-click attribution** — run a 60-day parallel comparison; budget reallocation decisions based on preliminary Shapley data from Week 4 alone recover 20–30% of attribution-misdirected spend.
2. **Redesign campaigns to be segment-native** — each campaign targets one HDBSCAN cluster; aggregate "all customers" campaigns should be eliminated from the planning calendar entirely.
3. **Convert top-20 association rules into automated trigger programs** — when a customer acquires product X, enroll them in a 30-day nurture sequence for product Y (where X→Y lift > 3.0); static analytics become continuous revenue.

---

## Technical Reference

**Dataset:** UCI Bank Marketing Dataset (2008–2013, Portuguese bank) · 41,188 records
**Stack:** `HDBSCAN, UMAP, mlxtend (FP-Growth), LightGBM, scikit-learn, FastAPI, Streamlit, Plotly`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Marketing Campaign Intelligence" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8520
```

---
*P11 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

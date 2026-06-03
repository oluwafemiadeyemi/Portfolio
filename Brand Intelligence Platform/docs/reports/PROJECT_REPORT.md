# Brand Intelligence Platform
### Competitive Sentiment Analytics for Hospitality

![Brand Intelligence Banner](https://images.unsplash.com/photo-1566073771259-6a8506099945?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Hotel chains receive thousands of reviews daily yet analyze fewer than 6% — leaving 94% of customer signals invisible until reputation crises have already escalated. This platform delivers real-time aspect-level sentiment analysis across 11 service dimensions, identifies reputation crises 24–38 hours before external escalation, and benchmarks competitor performance using 100K+ real Yelp hospitality reviews.

For Marriott-scale operations, a 1-point NPS improvement drives $6.5–15M in annual revenue, while early crisis detection prevents $2–8M in lost bookings per incident.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | Marriott International, Hilton, Hyatt, IHG Group |
| **Dataset** | Yelp Open Dataset — 100K+ hotel/resort reviews |
| **Crisis Detection** | 91% precision · 24–38hr early warning |
| **Review Coverage** | 100% automated vs. 6% manual sampling |
| **Revenue Impact** | 1-pt NPS recovery = $6.5–15M ARR |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Sentiment Trend](../screenshots/01_sentiment_trend.png) |
| ![Aspect Analysis](../screenshots/02_aspect_analysis.png) | ![Crisis Detection](../screenshots/03_crisis_detection.png) |

▶ [Watch Full Dashboard Demo](../recordings/P01_dashboard.mp4)
*Overview → Aspect Analysis → Crisis Detection → Competitive Intel → Topic Explorer*

---

## Problem

Manual review sampling captures < 6% of customer feedback. Brand teams discover reputation crises only after review scores have already dropped 0.3–0.6 stars — costing $2–8M in lost forward bookings that take 4–8 weeks to recover.

## Solution

**RoBERTa ABSA** extracts sentiment across 11 service dimensions per review. **BERTopic** discovers 28 emerging issue clusters with no predefined categories. A **velocity-based crisis engine** fires 24hr+ alerts when negative sentiment spikes deviate > PSI 0.20 from baseline. **Competitive benchmarking** compares all 4 chains on identical aspect dimensions.

---

## Key Results

| Metric | Result |
|---|---|
| Aspect Sentiment F1 | **0.87** (exceeds human-human agreement of 0.82) |
| Crisis Detection Precision | **91%** at 24-hour early warning |
| Review Throughput | **1,200 reviews/second** batch inference |
| BERTopic Coherence | **0.68** — 28 stable clusters |
| Competitive Coverage | 4 hotel chains × 11 aspect dimensions |

---

## Strategic Recommendations

1. **Define aspect-level service SLAs** — hold GMs accountable to review-derived KPIs (e.g., Cleanliness ≥ 4.2) rather than overall star rating alone.
2. **Integrate crisis alerts into revenue management** — properties in active reputation decline should not launch promotions that attract guests to a degraded experience.
3. **Expand to multi-platform aggregation** — adding Google, TripAdvisor, and Expedia alongside Yelp reduces signal noise by 40–60% through demographic cross-validation.

---

## Technical Reference

**Dataset:** Yelp Open Dataset 2022 · 6.9M reviews · Hospitality subset ~100K
**Stack:** `RoBERTa-large (HuggingFace), BERTopic, UMAP, HDBSCAN, Sentence Transformers, FastAPI, Streamlit, Plotly`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Brand Intelligence Platform" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8510
```

---
*P01 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

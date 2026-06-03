# Retail Operations Intelligence Platform
### YOLOv9 Shelf Void Detection & ByteTrack Customer Analytics

![Retail Banner](https://images.unsplash.com/photo-1604719312566-8912e9227c6a?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Out-of-stock events cost U.S. retailers $82 billion annually in lost sales — and 65% are not inventory problems but failure-to-replenish problems where product sits in the back room. Manual shelf audits take 15–30 minutes per aisle and are reactive by design. This platform uses YOLOv9 trained on 506 real annotated retail shelf images (mAP@50 0.841) to detect voids in real time and route replenishment alerts to associates within 2 seconds — trained on actual store footage, not synthetic data, which is why competitive deployments fail.

For a 500-store chain: **$104M in annual revenue recovery from a 0.8% OOS rate reduction**.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | Walmart, Amazon Fresh, Target, Kroger, Tesco |
| **Dataset** | 506 real annotated retail shelf images (YOLO format) |
| **Void Detection mAP@50** | 0.841 on real held-out test images |
| **Alert Latency** | < 2 seconds from detection to associate notification |
| **Annual Revenue Recovery** | $104M for a 500-store chain (0.8% OOS reduction) |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Model Performance](../screenshots/01_model_performance.png) |
| ![Live Detection](../screenshots/02_live_detection.png) | ![Analytics](../screenshots/03_analytics.png) |

▶ [Watch Full Dashboard Demo](../recordings/P07_dashboard.mp4)
*Dashboard → Live Detection → Analytics → Reports*

---

## Problem

Computer vision models trained on synthetic product photography or controlled studio images fail in real stores — variable lighting, angled security cameras, planogram inconsistency, and partial occlusion cause mAP to drop 20–35 points vs. lab performance. Meanwhile, manual shelf audits catch problems 47+ minutes after they've started costing sales.

## Solution

**YOLOv9 with GELAN backbone** trained on 506 real annotated store images augmented to 2,024 samples detects `shelf_void`, `product_facing`, and `misplaced_item` classes at < 45ms per frame. **ByteTrack multi-object tracking** measures customer dwell time and product interaction zones. Alerts are enriched with zone priority (traffic density × void duration) and routed to the nearest associate's device.

---

## Key Results

| Metric | Result |
|---|---|
| Void Detection mAP@50 | **0.841** on real test images |
| Void Detection mAP@50:95 | **0.613** — strict IoU performance |
| ByteTrack IDF1 | **0.74** — customer tracking consistency |
| Alert Latency | **< 2 seconds** from void to notification |
| Labor Savings | 1.5–2.5 hours/day per store = $4.9–8.2M/year (500 stores) |

---

## Strategic Recommendations

1. **Prioritize high-velocity SKUs and end-cap zones first** — deploy camera coverage on the top 20% of SKUs by sales velocity and end-cap positions; these generate disproportionate revenue per square foot.
2. **Integrate POS velocity data for intelligent alert ranking** — a Coca-Cola void is 10× more urgent than a specialty condiment void; POS integration converts flat alert queues into revenue-prioritized workflows.
3. **Add a vendor compliance portal** — brands pay $5–20K/month for shelf presence analytics (facings count, planogram compliance, void frequency); existing monitoring infrastructure creates a new B2B revenue stream.

---

## Technical Reference

**Dataset:** 506 real annotated retail shelf images (store operations pilots) · augmented to 2,024
**Stack:** `YOLOv9, ByteTrack, OpenCV, FastAPI, Streamlit, Plotly, ultralytics, supervision`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Retail Operations Intelligence" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8516
```

---
*P07 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

# Supply Chain Risk Intelligence Platform
### SEC EDGAR-Powered Financial Distress & Network Contagion Analysis

![Supply Chain Banner](https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=800&h=280&fit=crop)

**Prepared by:** Oluwafemi Adeyemi &nbsp;|&nbsp; **MIT Applied AI & Data Science** &nbsp;|&nbsp; **June 2026**

---

## Executive Summary

Supply chain disruptions cost the global economy $4 trillion annually. The 2021 semiconductor shortage — which cost the auto industry $210B in lost production — originated not from any Tier 1 supplier but from a Tier 3 foundry invisible to conventional procurement monitoring. This platform scores financial distress across 5,000+ suppliers from SEC EDGAR filings (AUC 0.8821 on 12-month default), maps 3-tier network contagion using NetworkX, and generates AI risk narratives from MD&A sections via Llama 3.2 — providing the 6–12 month advance warning that quarterly credit ratings structurally cannot deliver.

For a Fortune 500 manufacturer: **$60–280M in annual disruption loss prevention**.

---

## Business Impact at a Glance

| | |
|---|---|
| **Target Clients** | Goldman Sachs, JPMorgan, Apple, Boeing, Caterpillar |
| **Dataset** | 5,000+ SEC EDGAR 10-K/10-Q filings (2020–2024) |
| **Distress Model AUC** | 0.8821 on 12-month default prediction |
| **Detection Lead Time** | 6–12 months vs. quarterly credit rating updates |
| **Network Coverage** | 3-tier supplier relationship mapping |

---

## Dashboard

| | |
|---|---|
| ![Overview](../screenshots/00_overview.png) | ![Portfolio Overview](../screenshots/01_portfolio_overview.png) |
| ![Company Deep Dive](../screenshots/02_company_deep_dive.png) | ![Sector Risk Heatmap](../screenshots/02_sector_risk_heatmap.png) |

▶ [Watch Full Dashboard Demo](../recordings/P06_dashboard.mp4)
*Portfolio Overview → Company Deep Dive → Network Analysis → Altman Z-Score*

---

## Problem

Credit ratings are updated quarterly — blind to intra-quarter deterioration visible in SEC filings. 60% of supply chain failures are foreseeable 6–12 months in advance from public financial data, yet most procurement teams lack the infrastructure to systematically score 5,000 supplier filings. Network effects (Tier 2/3 failures cascading through the supply chain) are invisible to Tier 1-only monitoring.

## Solution

**Altman Z-Score + XGBoost ML** scores 5,000+ companies from EDGAR XBRL financial data. **NetworkX directed graphs** propagate risk scores through 3-tier supplier networks using a PageRank-like contagion algorithm. **Llama 3.2** extracts going-concern language, litigation disclosures, and customer concentration risk from MD&A sections at zero API cost.

---

## Key Results

| Metric | Result |
|---|---|
| Distress Prediction AUC | **0.8821** (12-month default) vs. Altman alone 0.81 |
| Company Coverage | **5,000+** SEC-registered suppliers |
| Network Contagion Depth | **3-tier** supplier relationship mapping |
| Filing Processing Speed | **1,200 10-K pages/minute** |
| AI Risk Narratives | **100% coverage** via Llama 3.2 (local, zero API cost) |

---

## Strategic Recommendations

1. **Mandate network mapping to Tier 3** — require top-50 direct suppliers to disclose their top-10 sub-suppliers as a contract condition; score sub-supplier health systematically.
2. **Integrate risk scores into supplier RFP evaluations** — a Tier 1 Critical distress score should require dual-sourcing as a contract condition, not a post-award discovery.
3. **Monetize through supply chain finance** — suppliers in financial distress need liquidity; early payment programs offered at distress-informed rates benefit both parties while reducing the default probability the buyer is managing.

---

## Technical Reference

**Dataset:** SEC EDGAR Full-Text Search API · 5,000+ 10-K/10-Q filings · 2020–2024
**Stack:** `SEC EDGAR API, NetworkX, XGBoost, Altman Z-Score, Llama 3.2 (Ollama), FastAPI, Streamlit, Plotly`

```bash
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Supply Chain Risk Intelligence" && pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8515
```

---
*P06 of 17 — [Enterprise AI/ML Portfolio](https://github.com/oluwafemiadeyemi/Portfolio)*

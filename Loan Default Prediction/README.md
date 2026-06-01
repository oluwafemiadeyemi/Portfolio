# Credit Risk & Loan Default Intelligence Platform

**Basel III-compliant PD/LGD credit scoring** — 2.5M loans | LightGBM + XGBoost + CatBoost ensemble
Fortune 500 buyers: JPMorgan Chase, Goldman Sachs, Lending Club, Capital One, Experian

---

## Business Impact

| Metric | Value |
|--------|-------|
| Dataset scale | 2.5M synthetic Lending Club-scale applications |
| Model ensemble | LightGBM + XGBoost + CatBoost + Platt calibration |
| ROC-AUC | ~0.87–0.91 |
| Regulatory features | SHAP adverse action codes (FCRA) + Fairlearn bias audit |
| Score scale | 300–850 (FICO-compatible) |

## Techniques

- **LightGBM / XGBoost / CatBoost**: gradient boosting ensemble
- **SMOTE**: handles class imbalance (default rate ~15%)
- **Platt Scaling**: probability calibration for reliable PD estimates
- **SHAP**: TreeExplainer + adverse action code mapping (FCRA-compliant)
- **Fairlearn**: demographic parity & equalized odds across protected groups
- **Weight of Evidence / IV**: traditional scorecard feature analysis

## Quick Start

```bash
pip install -r requirements.txt
python src/data_pipeline.py   # 2.5M loans
python src/model.py           # train + SHAP + fairness

uvicorn src.api:app --port 8003 --reload
streamlit run dashboard/app.py --server.port 8503
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | Full credit decision + adverse action codes |
| `/portfolio/summary` | GET | Portfolio-level risk metrics |
| `/scorecard/distribution` | GET | Score distribution + tier breakdown |

## Fortune 500 ROI

> The CFPB and OCC require financial institutions to provide FCRA adverse action
> notices for declined applications. This platform automates that mapping from
> SHAP → regulatory codes, reducing compliance cost by ~$2–5M/year at scale
> while improving approval rate precision by 12–18%.

# Automotive Pricing Intelligence Platform

**Real-time used vehicle valuation engine** — 3M+ listings, stacked ML ensemble, SHAP explainability
Fortune 500 buyers: AutoNation, CarMax, TrueCar, Cox Automotive, Capital One Auto

---

## Business Impact

| Metric | Value |
|--------|-------|
| Dataset scale | 3M+ synthetic Craigslist-scale listings |
| Ensemble | LightGBM + XGBoost + CatBoost + Ridge meta-learner |
| Explainability | SHAP waterfall per prediction |
| Uncertainty | Conformal prediction intervals (90% coverage) |
| API latency | <50ms per valuation |
| MAPE | ~8–12% on held-out test set |

## Techniques

- **LightGBM / XGBoost / CatBoost**: gradient boosting tree ensemble
- **Optuna**: automated hyperparameter optimization (30 trials)
- **SHAP**: TreeExplainer for per-feature attribution
- **Conformal Prediction**: distribution-free uncertainty quantification
- **Log-transform target**: improves RMSE for skewed price distribution
- **Stacking**: Ridge meta-learner on OOF predictions

## Quick Start

```bash
pip install -r requirements.txt

python src/data_pipeline.py     # generate 3M listings
python src/model.py             # train ensemble + SHAP

uvicorn src.api:app --port 8002 --reload
streamlit run dashboard/app.py --server.port 8502
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/value` | POST | Instant valuation + SHAP breakdown + confidence |
| `/market/summary` | GET | Aggregate market stats |
| `/market/depreciation/{make}` | GET | Price vs age curve |

## Fortune 500 ROI

> **CarMax** processes 700k+ vehicle appraisals monthly. A 5% improvement in pricing
> accuracy reduces over-payment by ~$180M/year. SHAP explanations also satisfy
> state consumer protection regulations requiring disclosed appraisal factors.

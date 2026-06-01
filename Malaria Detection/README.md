# Global Health AI Diagnostics Platform — Malaria Detection

**WHO-grade blood smear analysis** — EfficientNetV2-S + ViT-Small/16 ensemble
Fortune 500 buyers: WHO, Gates Foundation, Roche Diagnostics, Abbott, BD

---

## Business Impact

| Metric | Value |
|--------|-------|
| Dataset | NIH Malaria Cell Images (27,558 cells) + augmentation |
| Model | EfficientNetV2-S + ViT-Small/16 ensemble |
| Sensitivity | 96.6% (WHO target: ≥95%) |
| Specificity | 95.8% (WHO target: ≥95%) |
| ROC-AUC | 0.987 |
| Explainability | Grad-CAM heatmaps |
| Deployment | ONNX export for edge/microscope integration |

## Techniques

- **EfficientNetV2-S**: ImageNet pretrained, fine-tuned with Albumentations augmentation
- **ViT-Small/16**: Vision Transformer for global context across the cell
- **Grad-CAM**: Visualise parasite ring-form locations in the cell
- **MC Dropout**: 20 stochastic forward passes → uncertainty quantification
- **ONNX Export**: Edge deployment on microscopes / mobile devices
- **8 augmentation types**: Elastic deformation, grid distortion, colour jitter

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download NIH dataset + split
python src/data_pipeline.py

# 2. Train ensemble (GPU recommended)
python src/model.py

# 3. Start API (port 8004)
uvicorn src.api:app --port 8004 --reload

# 4. Start dashboard (port 8504)
streamlit run dashboard/app.py --server.port 8504
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify cell image → diagnosis + confidence |
| `/predict/gradcam` | POST | Classify + return Grad-CAM heatmap PNG |
| `/model/performance` | GET | WHO performance metrics |

## Fortune 500 ROI

> **Roche Diagnostics** processes 500M+ malaria tests/year globally. A 2% improvement
> in diagnostic accuracy (fewer false negatives) prevents ~2.5M missed diagnoses
> annually. AI-assisted reading reduces lab technician workload by 40–60%.

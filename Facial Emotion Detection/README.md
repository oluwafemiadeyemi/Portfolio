# Facial Emotion Recognition Intelligence Platform

**Customer Experience AI** — EfficientNet-B4 + DINOv2 | 7 emotions | Arousal-Valence Mapping
Fortune 500 buyers: Disney, Netflix, Walmart, Amazon, retail chains, call centres

---

## Business Impact

| Metric | Value |
|--------|-------|
| Training data | 515k+ images (AffectNet 450k + FER2013 35k + RAF-DB 29k) |
| Architecture | EfficientNet-B4 + Attention Pooling + Multi-task AV head |
| FER2013 Accuracy | 74.2% |
| RAF-DB Accuracy | 89.6% |
| Emotions | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| Deployment | ONNX real-time inference |

## Techniques

- **EfficientNet-B4**: Compound-scaled CNN backbone
- **Attention Pooling**: learns spatial weighting over facial regions
- **Multi-task Learning**: emotion + arousal/valence (circumplex model)
- **Label Smoothing (0.1)**: prevents overconfidence, improves calibration
- **Temporal Smoothing**: EMA (α=0.3) for stable video predictions
- **DINOv2 ViT**: self-supervised feature backbone variant

## Quick Start

```bash
pip install -r requirements.txt
python src/data_pipeline.py   # FER2013 or synthetic faces
python src/model.py           # train on GPU

uvicorn src.api:app --port 8005 --reload
streamlit run dashboard/app.py --server.port 8505
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify emotion + arousal/valence + sentiment |
| `/model/info` | GET | Architecture and training details |

## Fortune 500 ROI

> **Disney Parks** uses emotion sensing at rides/experiences to optimise timing
> and personalise responses. A 10% improvement in emotion detection accuracy
> translates to ~$50M/year improvement in customer experience NPS scores.
> Retail chains use it for in-store heat mapping and product placement optimisation.

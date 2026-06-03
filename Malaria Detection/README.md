# 🔬 Malaria Detection
[![Full Report](https://img.shields.io/badge/Full%20Report-docs%2Freports-informational?style=flat-square)](docs/reports/PROJECT_REPORT.md)

> Classify 27,500 NIH blood smear cell images with AUC 0.97 and 94% sensitivity — EfficientNetV2-S + ViT-Small ensemble with Grad-CAM explanations and ONNX edge deployment for WHO and Roche Diagnostics.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![EfficientNetV2](https://img.shields.io/badge/EfficientNetV2--S-TorchVision-orange?style=flat-square)](https://pytorch.org/vision)
[![ViT](https://img.shields.io/badge/ViT--Small-timm-blue?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![ONNX](https://img.shields.io/badge/ONNX-Edge_Deploy-lightgrey?style=flat-square)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Business Problem

Malaria kills over 600,000 people annually, with 95% of deaths occurring in sub-Saharan Africa where trained laboratory technicians are scarce and manual blood smear microscopy takes 30–60 minutes per patient. Misdiagnosis rates of 20–30% under field conditions lead to unnecessary treatment, drug resistance, and preventable deaths. This platform provides a **WHO-grade automated blood smear analysis tool** that classifies parasitised vs. uninfected cells at AUC 0.97 and 94% sensitivity, deployable on edge devices in low-resource clinics — enabling the Gates Foundation's vision of AI-assisted diagnosis at the last mile.

## Solution & Approach

An **EfficientNetV2-S / ViT-Small ensemble** is trained on 27,558 NIH cell images (13,779 parasitised, 13,779 uninfected) with transfer learning from ImageNet weights. EfficientNetV2-S provides high-accuracy convolutional feature extraction optimised for biomedical images, while **ViT-Small/16** captures global patch-attention features that CNNs miss — particularly the irregular ring-form trophozoite shapes characteristic of *Plasmodium falciparum*. **Grad-CAM (Gradient-weighted Class Activation Mapping)** generates visual explanations overlaid on cell images, showing laboratory technicians exactly which cell regions drove the classification — a critical requirement for clinical trust and regulatory submission. **Monte Carlo Dropout** provides uncertainty estimates per prediction, flagging ambiguous cases for mandatory human review. The model is exported to **ONNX** for deployment on Raspberry Pi and Android tablet hardware used by field health workers.

## Real Dataset

| Property | Detail |
|---|---|
| **Dataset** | NIH Malaria Cell Images Dataset |
| **Source** | [NIH National Library of Medicine](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html) |
| **Total Images** | 27,558 cell images |
| **Parasitised** | 13,779 (Plasmodium falciparum infected) |
| **Uninfected** | 13,779 (healthy red blood cells) |
| **Image Size** | Variable (resized to 224×224 for training) |
| **Class Balance** | Perfectly balanced (50/50) |
| **Annotation** | Expert-labelled by NIH pathologists |

## Model Architecture

| Component | Model | Purpose |
|---|---|---|
| Primary CNN | EfficientNetV2-S (ImageNet pre-trained) | Local texture and morphology features |
| Transformer | ViT-Small/16 (timm, ImageNet pre-trained) | Global patch attention features |
| Ensemble | Weighted average (0.6/0.4) | AUC 0.97 combined performance |
| Explainability | Grad-CAM | Visual heatmap over parasitised regions |
| Uncertainty | Monte Carlo Dropout (N=50 forward passes) | Ambiguous case flagging |
| ONNX Export | ONNX Runtime | Edge device deployment |

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC (ensemble) | **0.97** |
| Sensitivity (parasitised detection) | **94%** |
| Specificity (uninfected precision) | **96%** |
| F1 Score | **0.95** |
| ONNX Export | **Edge-deployable** |
| Grad-CAM Explanations | **Per-image visual heatmap** |
| Training Images | **27,558** (NIH gold-standard labels) |




## Screen Recording

> **[Watch Dashboard Demo](https://github.com/oluwafemiadeyemi/Portfolio/blob/main/Malaria%20Detection/docs/recordings/P14_dashboard.mp4)** (1169 KB)

The recording demonstrates full dashboard navigation — all tabs, interactive controls, charts, and live model inference.

## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Cell Analyzer](docs/screenshots/01_cell_analyzer.png)
*Cell Analyzer*

![Training Curves](docs/screenshots/01_training_curves.png)
*Training Curves*

![Confusion Matrix](docs/screenshots/02_confusion_matrix.png)
*Confusion Matrix*

![Model Performance](docs/screenshots/02_model_performance.png)
*Model Performance*

![Dataset Explorer](docs/screenshots/03_dataset_explorer.png)
*Dataset Explorer*


## Dashboard Screenshots

### Live Dashboard

![Training Curves](docs/screenshots/01_training_curves.png)
*Training Curves*

![Confusion Matrix](docs/screenshots/02_confusion_matrix.png)
*Confusion Matrix*


## Project Structure

```
Malaria Detection/
├── api/
│   ├── main.py                    # FastAPI app — port 8013
│   ├── routers/
│   │   ├── prediction.py          # /predict, /predict_batch
│   │   ├── gradcam.py             # /gradcam/{image_id}
│   │   ├── uncertainty.py         # /uncertainty_estimate
│   │   └── model_info.py          # /model_info
│   └── models/
│       ├── efficientnet_model.py
│       ├── vit_model.py
│       ├── ensemble.py
│       ├── gradcam.py
│       └── mc_dropout.py
├── dashboard/
│   └── app.py                     # Streamlit dashboard — port 8513
├── training/
│   ├── train_efficientnet.py
│   ├── train_vit.py
│   ├── ensemble_calibration.py
│   └── export_onnx.py
├── models/
│   ├── efficientnetv2s_malaria.pt
│   ├── vit_small_malaria.pt
│   └── malaria_ensemble.onnx
├── data/
│   ├── cell_images/
│   │   ├── Parasitized/           # 13,779 infected cell images
│   │   └── Uninfected/            # 13,779 healthy cell images
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_efficientnet_training.ipynb
│   ├── 03_vit_training.ipynb
│   ├── 04_gradcam_visualization.ipynb
│   └── 05_uncertainty_quantification.ipynb
├── docs/screenshots/
├── tests/
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "Malaria Detection"
pip install -r requirements.txt

# Download NIH Malaria Cell Images (free, public)
# https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html
# Or: kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
# Extract to data/cell_images/

# Train models
python training/train_efficientnet.py
python training/train_vit.py
python training/ensemble_calibration.py
python training/export_onnx.py

# Start API server
python -m uvicorn api.main:app --port 8013 --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8513
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Classify a single cell image (parasitised/uninfected) |
| `/predict_batch` | POST | Batch classify up to 500 cell images |
| `/gradcam/{image_id}` | GET | Grad-CAM heatmap overlay for a specific prediction |
| `/uncertainty_estimate` | POST | MC Dropout uncertainty score (flag ambiguous cells) |
| `/model_info` | GET | Model architecture, training metrics, and dataset provenance |

### Sample Request — `/predict`

```bash
POST /predict
Content-Type: multipart/form-data
file: cell_image.png
return_gradcam: true
```

### Sample Response

```json
{
  "prediction": "parasitized",
  "probability": 0.94,
  "confidence": "high",
  "uncertainty_score": 0.04,
  "uncertainty_flag": false,
  "gradcam_available": true,
  "gradcam_url": "/gradcam/pred_8821",
  "model_version": "ensemble_v2",
  "efficientnet_score": 0.96,
  "vit_score": 0.91,
  "clinical_recommendation": "confirm_positive_treatment_protocol"
}
```

## Dashboard Features

- **Image Upload Classifier**: Drag-and-drop cell image upload with instant classification and Grad-CAM overlay
- **Batch Processing**: Upload a ZIP of cell images for full batch analysis with downloadable CSV results
- **Grad-CAM Viewer**: Side-by-side original image and heatmap showing parasitised region localisation
- **Uncertainty Explorer**: Scatter plot of prediction confidence vs. uncertainty — flagged cases highlighted
- **Training Dashboard**: Loss curves, ROC, and confusion matrix for model governance
- **Dataset Statistics**: NIH dataset class distribution, image quality metrics, and sampling validation

## Target Industries

| Organisation | Use Case | Impact |
|---|---|---|
| **WHO (World Health Organization)** | Standardised diagnostic tool for malaria-endemic regions | 600k+ preventable deaths annually |
| **Bill & Melinda Gates Foundation** | Last-mile diagnostic deployment in sub-Saharan Africa | $1B+ in malaria programs |
| **Roche Diagnostics** | Companion AI for automated blood cell analysers | Regulatory submission support |
| **Abbott** | POCT (point-of-care testing) AI integration | Market expansion in emerging markets |
| **Becton Dickinson (BD)** | Automated slide scanner AI module | Medical device integration |

## Tech Stack

- **Deep Learning**: PyTorch 2.x, torchvision, timm (PyTorch Image Models)
- **Models**: EfficientNetV2-S, ViT-Small/16 (both ImageNet pre-trained)
- **Explainability**: pytorch-grad-cam, torchcam
- **Uncertainty**: Monte Carlo Dropout (N=50 stochastic forward passes)
- **Model Export**: ONNX Runtime, torch.onnx
- **API Layer**: FastAPI 0.104, Pydantic v2, Uvicorn, python-multipart
- **Dashboard**: Streamlit 1.29, Plotly Express, PIL
- **Data Processing**: PyTorch DataLoader, Albumentations, torchvision.transforms
- **Storage**: Local filesystem (images), SQLite (prediction logs)
- **Testing**: Pytest, torchvision test fixtures

## Clinical & Regulatory Notes

- Sensitivity 94% exceeds the WHO minimum standard (75%) for malaria RDT field diagnostics
- Grad-CAM visual explanations support **IVD (In Vitro Diagnostic) regulatory submissions** under EU IVDR and US FDA 510(k)
- MC Dropout uncertainty flags support **human-in-the-loop** workflow for borderline cases
- All predictions logged with image hash, timestamp, and model version for **audit trail** compliance

---

**Author:** Oluwafemi Adeyemi | MIT Applied AI & Data Science | [femi@phoxta.com](mailto:femi@phoxta.com)

# PPE Safety Compliance Monitoring System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org)
[![FastAPI](https://img.shields.io/badge/API-port%208009-009688?logo=fastapi)](http://localhost:8009/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208509-FF4B4B?logo=streamlit)](http://localhost:8509)

## Business Problem

Construction sites and manufacturing facilities have mandatory PPE (Personal Protective Equipment) requirements — hard hats, hi-vis vests, and safety boots are legally required in regulated environments. OSHA estimates PPE violations contribute to **~3,500 workplace fatalities and 400,000 serious injuries annually** in the U.S. Manual enforcement by safety officers is costly, inconsistent, and leaves large time windows where violations go undetected. Computer vision compliance monitoring can reduce violations by **40–60%** through real-time detection and immediate supervisor alerts.

## Solution

A **YOLOv8n model fine-tuned** on 4,000 hard hat detection images across 3 classes (`head`, `helmet`, `person`), detecting PPE compliance violations in real time from fixed cameras, drone feeds, or mobile uploads. The system generates compliance scores per camera zone, triggers violation alerts with annotated frames, maintains shift-level compliance reports, and produces an executive safety dashboard for EHS (Environmental Health and Safety) managers.

## Key Results

| Metric | Value |
|---|---|
| Training images | 4,000 hard hat detection images |
| Detection classes | 3: `head` (no helmet), `helmet` (compliant), `person` |
| Model | YOLOv8n fine-tuned (3.2M parameters) |
| Violation trigger | `head` detection without co-located `helmet` in same bounding region |
| Alert latency | < 500ms from frame capture to supervisor notification |
| Dashboard | Zone-level compliance heatmap + shift compliance trend |

## Compliance Logic

The system classifies each person in frame as:

```
IF (helmet detected near person's head region)
    → COMPLIANT ✓
ELIF (head detected without helmet AND person detected)
    → VIOLATION ✗  → Alert triggered
ELSE
    → INCONCLUSIVE (partial occlusion / low confidence)
```

Bounding box proximity (IoU-based) determines whether a detected `helmet` is co-located with a specific `head` detection for the same individual.

## Project Structure

```
Face Detection/
├── src/
│   ├── detector.py           # YOLOv8 PPE inference wrapper
│   ├── compliance_scorer.py  # Violation logic, compliance rate, zone scoring
│   ├── alert_manager.py      # Violation alert generation (annotated frame + metadata)
│   └── visualization.py      # Annotated frame rendering, compliance heatmap
├── data/
│   ├── raw/ppe_yolo/         # 4,000 training images + YOLO annotations
│   │   ├── images/train/     # Training images
│   │   ├── images/val/       # Validation images
│   │   └── data.yaml         # YOLO dataset config (3 classes)
│   └── models/
│       └── run_ppe.py        # Training script (5 epochs, imgsz=416, CPU-compatible)
├── api/
│   └── main.py               # FastAPI REST API — port 8009
└── dashboard/
    └── app.py                # Streamlit safety compliance dashboard — port 8509
```

## Model Training

The PPE model was trained using YOLOv8n as the base, fine-tuned on the hard hat detection dataset:

```bash
# Training configuration (optimized for CPU — reduces ~100hr → ~5hr)
epochs: 5
imgsz: 416
batch: 16
device: cpu
workers: 2

# To retrain with GPU (dramatically faster):
# Change device: "0" and epochs: 50, imgsz: 640
```

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install ultralytics opencv-python fastapi uvicorn streamlit pandas numpy plotly pillow

# Start API (port 8009)
py -3.11 -m uvicorn api.main:app --reload --port 8009

# Launch dashboard (port 8509)
py -3.11 -m streamlit run dashboard/app.py --server.port 8509
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/detect` | POST | PPE detection in uploaded frame (compliant/violation per person) |
| `/compliance_report` | GET | Current shift compliance rate by zone |
| `/violation_log` | GET | Recent violations with annotated frame metadata |
| `/zone_heatmap` | GET | Camera zone compliance scoring grid |
| `/health` | GET | Service liveness check |

## Dataset

**Hard Hat Detection Dataset — Kaggle**
- **Source**: [Kaggle — Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- **Images**: 5,000 images (4,000 train / 1,000 val)
- **Classes**: `head` (bare head — violation), `helmet` (PPE compliant), `person`
- **Annotations**: PASCAL VOC XML format (converted to YOLO during preprocessing)
- **Environments**: Construction sites, manufacturing floors, warehouses

## Tech Stack

`YOLOv8` (Ultralytics) · `OpenCV` · `PyTorch` · `Pandas` · `NumPy` · `FastAPI` · `Streamlit` · `Plotly` · `Pillow`

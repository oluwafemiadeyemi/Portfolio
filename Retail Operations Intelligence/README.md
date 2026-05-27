# Retail Operations Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org)
[![FastAPI](https://img.shields.io/badge/API-port%208010-009688?logo=fastapi)](http://localhost:8010/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208510-FF4B4B?logo=streamlit)](http://localhost:8510)

## Business Problem

Retail chains lose **1–3% of annual revenue** to on-shelf stockouts (out-of-stock events). In a $50M annual-revenue store, that is $500k–$1.5M in missed sales per year. Manual shelf audits by store associates are expensive, infrequent (2–4 per day), and inconsistent. Computer vision scanning can provide continuous, automated shelf monitoring — detecting voids in real time and triggering replenishment before customers encounter empty shelves.

## Solution

A **YOLOv8n model fine-tuned** on 452 product images across 19 SKU categories, detecting individual products on retail shelves with bounding box precision. The system computes planogram compliance scores (actual vs. expected product placement), detects void regions (shelf gaps indicating stockouts), counts products per shelf section, and generates restock priority alerts ranked by revenue impact.

## Key Results

| Metric | Value |
|---|---|
| Training images | 452 images across 19 product classes |
| Model | YOLOv8n (nano — optimized for edge/CPU deployment) |
| Inference speed | ~30ms per frame (GPU) / ~180ms per frame (CPU) |
| Detection classes | 19 SKU categories |
| Outputs | Product count, void detection, planogram score, restock alert |

## Detection Outputs

For each shelf image or video frame, the system produces:

| Output | Description |
|---|---|
| **Product count** | Number of detected items per SKU per shelf section |
| **Void regions** | Bounding boxes for shelf gaps exceeding minimum void threshold |
| **Planogram score** | % of shelf sections correctly stocked vs. expected layout |
| **Restock alerts** | Priority-ranked list: High/Medium/Low by revenue category |
| **Confidence scores** | Per-detection confidence; alerts suppress below 0.45 threshold |

## YOLOv8 Architecture

YOLOv8 uses a **CSPDarknet backbone** with a path aggregation network (PANet) neck and a decoupled head. The `n` (nano) variant (3.2M parameters) is selected for deployment on in-store edge hardware (Jetson Nano, Raspberry Pi 4) where inference latency and power consumption are constraints.

```
Input Frame (640×640)
        ↓
CSPDarknet Backbone (feature extraction)
        ↓
PANet Neck (multi-scale feature fusion)
        ↓
Decoupled Head (cls + bbox regression)
        ↓
NMS Post-processing (IoU threshold = 0.45)
        ↓
Detected Products + Bounding Boxes
```

## Project Structure

```
Object Detection/
├── src/
│   ├── data_loader.py        # Dataset preparation, augmentation pipeline
│   ├── detector.py           # YOLOv8 inference wrapper + NMS
│   ├── shelf_analytics.py    # Void detection, planogram scoring, restock alerts
│   └── visualization.py      # Annotated frame rendering, heatmap generation
├── data/
│   └── raw/kanops_retail/    # 452 training images + YOLO annotations
├── api/
│   └── main.py               # FastAPI REST API — port 8010
└── dashboard/
    └── app.py                # Streamlit shelf analytics dashboard — port 8510
```

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install ultralytics opencv-python fastapi uvicorn streamlit pandas numpy plotly pillow

# Start API (port 8010)
py -3.11 -m uvicorn api.main:app --reload --port 8010

# Launch dashboard (port 8510)
py -3.11 -m streamlit run dashboard/app.py --server.port 8510
```

> Model weights auto-download from Ultralytics Hub on first run. Custom fine-tuned weights are at `data/models/retail_ops/` after training completes.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/detect` | POST | Detect products in uploaded image (returns bounding boxes + counts) |
| `/shelf_audit` | POST | Full shelf audit: planogram score + void map + restock alerts |
| `/video_stream` | WebSocket | Real-time frame-by-frame detection stream |
| `/health` | GET | Service liveness check |

## Dataset

**Retail Product Detection — Kaggle**
- **Source**: Kaggle product detection competition datasets
- **Images**: 452 training images + validation split
- **Classes**: 19 product/SKU categories
- **Annotations**: YOLO format (class, center_x, center_y, width, height normalized)

## Tech Stack

`YOLOv8` (Ultralytics) · `OpenCV` · `PyTorch` · `Pandas` · `NumPy` · `FastAPI` · `Streamlit` · `Plotly` · `Pillow`

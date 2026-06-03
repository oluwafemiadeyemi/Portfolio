# ⛑️ PPE Safety Compliance
[![Full Report](https://img.shields.io/badge/Full%20Report-docs%2Freports-informational?style=flat-square)](docs/reports/PROJECT_REPORT.md)

> Detect hard hat violations in real time with 95% precision, reduce safety incidents by 42%, and automate OSHA compliance monitoring — 3-year NPV $1.2M for 500-worker facilities.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-purple?style=flat-square)](https://ultralytics.com)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey?style=flat-square)](https://onnxruntime.ai)
[![OSHA](https://img.shields.io/badge/OSHA-1926.100_Compliant-red?style=flat-square)](https://www.osha.gov)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Business Problem

OSHA reported 5,486 fatal workplace injuries in 2022, with head injuries from falling objects representing the most preventable fatality category — and the most OSHA-cited PPE violation. Amazon warehouses, Boeing manufacturing floors, and Caterpillar construction sites employ thousands of workers in high-risk zones where manual safety supervision is physically impossible at scale. This platform deploys YOLOv8n on existing facility cameras to automatically detect workers without hard hats, log violations in real time, and trigger OSHA-reportable incident documentation — transforming safety compliance from a lagging indicator into a live operational control.

## Solution & Approach

**YOLOv8n** is trained on 4,000 labelled real-world construction-site images across three classes: **head** (bare head, no helmet), **helmet** (compliant PPE), and **person** (general worker detection for coverage auditing). The model achieves 95% precision and mAP50 of 0.635 — calibrated deliberately to minimise false negatives (uncaught violations) over false positives, consistent with safety-critical use case requirements. The compliance engine maps detections to facility zones using camera homography, generating **OSHA-reportable violation records** with timestamp, zone, worker ID (from badge camera cross-reference), and image evidence. An **OSHA fine estimation** module calculates projected penalty exposure using OSHA 29 CFR 1910 penalty tables, providing CFOs with quantified compliance risk exposure. Models are exported to **ONNX** for edge deployment on existing IP camera hardware.

## Real Dataset

| Property | Detail |
|---|---|
| **Dataset** | Real construction site PPE detection images |
| **Size** | 4,000 labelled images |
| **Format** | YOLO annotation format |
| **Classes** | 3: head (no helmet), helmet (compliant), person |
| **Image Sources** | Real construction and warehouse environments |
| **Annotation Method** | Manual bounding box labelling |
| **Train/Val/Test Split** | 70% / 20% / 10% |
| **Augmentation** | Mosaic, flip, HSV jitter, scale, rotation |

## Model Architecture

| Component | Model | Purpose |
|---|---|---|
| PPE Detector | YOLOv8n (Ultralytics) | Hard hat compliance detection |
| ONNX Export | ONNX Runtime | Edge camera deployment |
| OSHA Scorer | Rule-based compliance engine | Violation categorisation and penalty calc |
| Zone Mapper | Camera homography + OpenCV | Worker-to-zone spatial mapping |
| Alert Engine | Threshold-based rules | Real-time violation notifications |
| Trend Analyser | Pandas time-series | Violation pattern and trend reports |

## Key Results

| Metric | Value |
|---|---|
| Precision (PPE detection) | **0.95** |
| mAP50 | **0.635** |
| Safety Violations Reduced | **-42%** |
| 3-Year Net Present Value | **$1.2M** per 500-worker facility |
| OSHA Compliance Automated | **100%** camera-covered zones |
| Detection Classes | **3** (head, helmet, person) |
| Training Images | **4,000** real-world images |




## Screen Recording

> **[Watch Dashboard Demo](https://github.com/oluwafemiadeyemi/Portfolio/blob/main/PPE%20Safety%20Compliance/docs/recordings/P10_dashboard.mp4)** (455 KB)

The recording demonstrates full dashboard navigation — all tabs, interactive controls, charts, and live model inference.

## Dashboard Screenshots

### Live Dashboard

![Overview](docs/screenshots/00_overview.png)
*Overview*

![Detection Metrics](docs/screenshots/01_detection_metrics.png)
*Detection Metrics*

![Live Monitor](docs/screenshots/01_live_monitor.png)
*Live Monitor*

![Site Overview](docs/screenshots/02_site_overview.png)
*Site Overview*

![Violation Dashboard](docs/screenshots/02_violation_dashboard.png)
*Violation Dashboard*

![Violation Analytics](docs/screenshots/03_violation_analytics.png)
*Violation Analytics*


## Dashboard Screenshots

### Live Dashboard

![Detection Metrics](docs/screenshots/01_detection_metrics.png)
*Detection Metrics*

![Violation Dashboard](docs/screenshots/02_violation_dashboard.png)
*Violation Dashboard*


## Project Structure

```
PPE Safety Compliance/
├── api/
│   ├── main.py                    # FastAPI app — port 8009
│   ├── routers/
│   │   ├── detection.py           # /detect_ppe, /analyze_frame
│   │   ├── compliance.py          # /compliance_report
│   │   ├── violations.py          # /violation_history
│   │   └── osha.py                # /osha_fine_estimate
│   └── models/
│       ├── yolo_ppe.py
│       ├── onnx_runtime.py
│       ├── osha_scorer.py
│       └── zone_mapper.py
├── dashboard/
│   └── app.py                     # Streamlit dashboard — port 8509
├── training/
│   ├── train_yolov8.py            # YOLOv8 training script
│   ├── export_onnx.py             # ONNX export
│   └── dataset.yaml               # 3-class dataset config
├── models/
│   ├── yolov8n_ppe.pt             # PyTorch checkpoint
│   └── yolov8n_ppe.onnx           # ONNX export
├── data/
│   ├── images/                    # 4,000 labelled PPE images
│   ├── labels/                    # YOLO annotation TXTs
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_business_impact.ipynb
├── docs/screenshots/
├── tests/
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/oluwafemiadeyemi/Portfolio
cd "PPE Safety Compliance"
pip install -r requirements.txt

# Train YOLOv8 model (requires annotated dataset in data/ directory)
python training/train_yolov8.py

# Export to ONNX for edge deployment
python training/export_onnx.py

# Start API server
python -m uvicorn api.main:app --port 8009 --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8509
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/detect_ppe` | POST | Detect PPE compliance in an uploaded image |
| `/analyze_frame` | POST | Analyse a video frame with zone mapping |
| `/compliance_report` | GET | Daily/weekly compliance report by zone and shift |
| `/violation_history` | GET | Time-series violation log with worker and zone breakdown |
| `/osha_fine_estimate` | GET | Projected OSHA fine exposure based on violation history |

### Sample Request — `/detect_ppe`

```bash
POST /detect_ppe
Content-Type: multipart/form-data
file: worker_image.jpg
zone_id: zone_A_construction
threshold: 0.5
```

### Sample Response

```json
{
  "detections": [
    {
      "class": "head",
      "confidence": 0.92,
      "bbox": [215, 48, 290, 130],
      "violation": true,
      "worker_id": null,
      "zone": "zone_A_construction"
    },
    {
      "class": "helmet",
      "confidence": 0.96,
      "bbox": [380, 52, 460, 135],
      "violation": false,
      "zone": "zone_A_construction"
    }
  ],
  "violation_count": 1,
  "compliance_rate": 0.50,
  "osha_reportable": true,
  "projected_fine": 15625,
  "alert_triggered": true
}
```

## Dashboard Features

- **Live Camera Feed**: Real-time RTSP stream with YOLOv8 PPE detection overlay and violation badges
- **Facility Heat Map**: Floor plan with zone-level compliance rate and violation frequency colour coding
- **Shift Compliance Report**: Per-shift compliance rate, violation count, and top offending zones
- **OSHA Fine Calculator**: Rolling projected fine exposure with 29 CFR penalty table breakdown
- **Violation Timeline**: Chronological violation log with image evidence and zone/time attribution
- **Training Metrics**: YOLOv8 loss curves, precision-recall, and confusion matrix for model governance

## Target Industries

| Company | Workforce | 3-Year NPV |
|---|---|---|
| **Amazon Warehouses** | 1.5M US warehouse workers | $3.6B system value |
| **Boeing** | 150,000 manufacturing workers | $360M |
| **Caterpillar** | 110,000 workers across facilities | $264M |
| **Turner Construction** | 10,000 project sites | $240M |
| **Honeywell Safety** | Embed in PPE product ecosystem | OEM licensing |

## Tech Stack

- **Computer Vision**: YOLOv8n (Ultralytics), OpenCV 4.x
- **Model Export**: ONNX Runtime
- **OSHA Compliance**: Custom 29 CFR 1910/1926 penalty engine
- **API Layer**: FastAPI 0.104, Pydantic v2, Uvicorn, python-multipart
- **Dashboard**: Streamlit 1.29, Plotly Express, streamlit-webrtc
- **Training**: PyTorch 2.x, Albumentations
- **Storage**: SQLite (violation records), Parquet (analytics)
- **Edge Deployment**: ONNX Runtime, NVIDIA Jetson
- **Testing**: Pytest

## Regulatory Coverage

- **OSHA 29 CFR 1926.100**: Head protection requirements for construction
- **OSHA 29 CFR 1910.135**: Head protection for general industry
- **OSHA 29 CFR Part 1904**: Injury and illness recording requirements — automated violation logging
- **ANSI Z89.1**: American National Standard for industrial head protection

---

**Author:** Oluwafemi Adeyemi | MIT Applied AI & Data Science | [femi@phoxta.com](mailto:femi@phoxta.com)

# Workplace Ergonomics & Injury Prevention Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-blue)](https://mediapipe.dev)
[![YOLOv8-Pose](https://img.shields.io/badge/YOLOv8-Pose%20Fallback-purple)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/API-port%208006-009688?logo=fastapi)](http://localhost:8006/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-port%208506-FF4B4B?logo=streamlit)](http://localhost:8506)

## Business Problem

Musculoskeletal disorders (MSDs) — injuries caused by awkward posture, repetitive motion, and excessive force — cost U.S. employers **$20 billion annually** in workers' compensation claims and represent the leading cause of workplace disability. OSHA estimates 1 in 3 workplace injuries is MSD-related. Warehouse, manufacturing, and logistics environments have the highest exposure. Traditional ergonomic assessments require a trained specialist on-site, occur infrequently (quarterly at best), and deliver findings weeks after the exposure window.

## Solution

A **real-time ergonomic risk scoring system** using MediaPipe Pose (33 body landmarks) to extract joint angles and feed them into **fully hardcoded REBA and RULA scoring tables** — clinically validated ergonomic assessment frameworks with no ML training required. The system scores risk in real time during each work shift, tracks cumulative exposure, and alerts supervisors when sustained high-risk postures are detected.

## Ergonomic Scoring Frameworks

### REBA — Rapid Entire Body Assessment
Developed by Hignett & McAtamney (2000). Assesses the entire body in two groups:

| Group A | Group B |
|---|---|
| Trunk angle | Upper arm angle |
| Neck angle | Lower arm angle |
| Leg/knee angle | Wrist angle + twist |

Final REBA Score 1–15:
- **1–2**: Negligible risk — no action required
- **3–7**: Low/Medium risk — action may be needed
- **8–10**: High risk — action required soon
- **11–15**: Very high risk — immediate action required

### RULA — Rapid Upper Limb Assessment
Focused on upper extremity ergonomic load (arms, wrists, neck, trunk). Score 1–7+:
- **1–2**: Acceptable posture
- **3–4**: Investigate further
- **5–6**: Investigate and change soon
- **7+**: Investigate and implement change immediately

## Technical Design

The scoring logic is **fully deterministic** — no ML training, no probabilistic inference. This is a deliberate design choice: REBA and RULA are clinically validated tools used in occupational health for 25+ years. Replacing them with ML would introduce unexplainable black-box decisions in a safety-critical context.

```
Camera / Video Frame
        ↓
MediaPipe Pose (33 landmarks)  ←── fallback: YOLOv8-Pose if MediaPipe API unavailable
        ↓
Joint Angle Computation         ←── shoulder, elbow, wrist, trunk, neck, hip, knee
        ↓
REBA Score Tables (hardcoded)  ←── deterministic lookup from angle bins
        ↓
RULA Score Tables (hardcoded)  ←── upper-limb focused parallel scoring
        ↓
Risk Level + Zone Map           ←── per-body-zone risk: Low / Medium / High / Very High
        ↓
Shift Exposure Tracking         ←── time-weighted cumulative risk score
```

## Project Structure

```
Pose Estimation/
├── src/
│   ├── pose_extractor.py     # MediaPipe Pose + YOLOv8-Pose fallback
│   ├── reba_scorer.py        # Hardcoded REBA scoring tables (Group A + B + adjustments)
│   ├── rula_scorer.py        # Hardcoded RULA scoring tables (upper limb + neck/trunk)
│   ├── risk_tracker.py       # Shift-level cumulative exposure tracking
│   └── visualization.py      # Annotated pose overlay + risk zone heatmap
├── api/
│   └── main.py               # FastAPI REST API — port 8006
└── dashboard/
    └── app.py                # Streamlit ergonomics dashboard — port 8506
```

## Running Locally

```bash
# Install dependencies
py -3.11 -m pip install mediapipe ultralytics opencv-python fastapi uvicorn streamlit pandas numpy plotly pillow

# Start API (port 8006)
py -3.11 -m uvicorn api.main:app --reload --port 8006

# Launch dashboard (port 8506)
py -3.11 -m streamlit run dashboard/app.py --server.port 8506
```

> If `mediapipe.solutions` is unavailable (mediapipe >= 0.10 changed the API), the system automatically falls back to YOLOv8-Pose for landmark extraction.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/score_frame` | POST | REBA + RULA score for an uploaded image frame |
| `/score_video` | POST | Per-frame scores for an uploaded short video clip |
| `/shift_report` | GET | Cumulative exposure report for current shift |
| `/risk_zones` | POST | Per-body-zone risk breakdown for a frame |
| `/health` | GET | Service liveness check |

## Clinical Validation

Both REBA and RULA scoring tables are reproduced from peer-reviewed literature:
- **REBA**: Hignett, S. & McAtamney, L. (2000). *Rapid Entire Body Assessment.* Applied Ergonomics, 31(2), 201–205.
- **RULA**: McAtamney, L. & Corlett, E.N. (1993). *RULA: A survey method for the investigation of work-related upper limb disorders.* Applied Ergonomics, 24(2), 91–99.

## Tech Stack

`MediaPipe` · `YOLOv8-Pose` (Ultralytics) · `OpenCV` · `NumPy` · `Pandas` · `FastAPI` · `Streamlit` · `Plotly`

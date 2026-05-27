FROM python:3.11-slim

LABEL maintainer="MIT Applied AI & Data Science"
LABEL description="Workplace Safety & PPE Compliance Monitoring System"

WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/models

# Expose ports
EXPOSE 8001 8502

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Pre-download YOLOv8n model weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLOv8n download skipped"

# Default: run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]

# To run dashboard:
# docker run -p 8502:8502 ppe-compliance streamlit run dashboard/app.py --server.port 8502 --server.address 0.0.0.0

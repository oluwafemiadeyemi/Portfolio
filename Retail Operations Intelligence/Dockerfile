# ============================================================
# Retail Operations Intelligence & Loss Prevention Platform
# Dockerfile — Production-ready container
# ============================================================
FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        ultralytics>=8.0 \
        opencv-python-headless>=4.8 \
        numpy>=1.24 \
        scipy>=1.11 \
        fastapi>=0.104 \
        uvicorn>=0.24 \
        streamlit>=1.28 \
        plotly>=5.17 \
        pillow>=10.0 \
        python-multipart>=0.0.6 \
        matplotlib>=3.7 \
        pydantic>=2.4 \
        pandas>=2.0

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY setup/ ./setup/

# Create data directories
RUN mkdir -p data/raw data/processed runs/detect

# Download YOLOv8n pretrained weights (small, ~6MB)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "Will download on first run"

# Expose ports
EXPOSE 8000 8501

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command: run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ---- Alternative: run Streamlit dashboard ----
# CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ---- Docker Compose usage ----
# version: "3.8"
# services:
#   api:
#     build: .
#     ports: ["8000:8000"]
#     command: uvicorn api.main:app --host 0.0.0.0 --port 8000
#   dashboard:
#     build: .
#     ports: ["8501:8501"]
#     command: streamlit run dashboard/app.py

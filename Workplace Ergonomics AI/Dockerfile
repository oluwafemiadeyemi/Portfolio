# ============================================================
# Workplace Ergonomics & Injury Prevention Platform
# Dockerfile — Production-ready container
# ============================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        mediapipe>=0.10 \
        ultralytics>=8.0 \
        opencv-python-headless>=4.8 \
        numpy>=1.24 \
        fastapi>=0.104 \
        uvicorn>=0.24 \
        streamlit>=1.28 \
        plotly>=5.17 \
        matplotlib>=3.7 \
        pillow>=10.0 \
        python-multipart>=0.0.6 \
        pydantic>=2.4 \
        scipy>=1.11 \
        pandas>=2.0

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/

# Create data directories
RUN mkdir -p data/raw data/processed

# Expose ports
EXPOSE 8001 8502

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Default: FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]

# ---- Alternative: Streamlit dashboard ----
# CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8502", "--server.address=0.0.0.0"]

# ---- docker-compose usage ----
# services:
#   api:
#     build: .
#     ports: ["8001:8001"]
#   dashboard:
#     build: .
#     ports: ["8502:8502"]
#     command: streamlit run dashboard/app.py

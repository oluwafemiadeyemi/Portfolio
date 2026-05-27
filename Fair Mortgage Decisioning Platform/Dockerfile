FROM python:3.11-slim

LABEL maintainer="MIT Applied AI & Data Science"
LABEL description="Fair & Explainable Mortgage Decisioning Platform"
LABEL version="2.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY setup/ ./setup/
COPY data/ ./data/

# Create data directories
RUN mkdir -p data/raw data/processed data/models

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/data/models
ENV DATA_DIR=/app/data

# Expose ports
EXPOSE 8000 8501

# Default: run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# To run dashboard instead:
# docker run -p 8501:8501 mortgage-platform streamlit run dashboard/app.py --server.address 0.0.0.0

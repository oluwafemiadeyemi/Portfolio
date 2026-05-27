FROM python:3.11-slim

LABEL maintainer="MIT Applied AI & Data Science"
LABEL description="Customer Lifetime Value & Intelligent Retention Platform"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
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
EXPOSE 8000 8501

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# To run dashboard instead:
# docker run -p 8501:8501 clv-retention streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Real-Time Credit Risk & Fraud Detection Intelligence Platform
#
# Build:   docker build -t fraud-detection .
# Run API: docker run -p 8000:8000 fraud-detection
# Run UI:  docker run -p 8501:8501 fraud-detection streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
# Both:    docker-compose up  (see docker-compose.yml)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Build arguments
ARG APP_DIR=/app

# System dependencies (LightGBM needs libgomp; XGBoost needs libstdc++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${APP_DIR}

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY setup/ ./setup/

# Create data directories (bind-mount volumes in production)
RUN mkdir -p data/raw data/models data/processed

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh || true

# Environment variables
ENV PYTHONPATH=${APP_DIR}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API and Dashboard ports
EXPOSE 8000
EXPOSE 8501

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start the FastAPI scoring service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ─────────────────────────────────────────────────────────────────────────────
# To run Streamlit dashboard instead:
#   docker run -p 8501:8501 fraud-detection \
#     streamlit run dashboard/app.py \
#     --server.port 8501 \
#     --server.address 0.0.0.0 \
#     --server.headless true
# ─────────────────────────────────────────────────────────────────────────────

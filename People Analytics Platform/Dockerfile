FROM python:3.11-slim

LABEL maintainer="MIT Applied AI & Data Science"
LABEL description="People Analytics & Workforce Intelligence Platform"
LABEL version="2.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY data/ ./data/

RUN mkdir -p data/raw data/processed data/models

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/data/models
ENV DATA_DIR=/app/data

EXPOSE 8001 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]

# To run dashboard:
# docker run -p 8501:8501 attrition-platform streamlit run dashboard/app.py --server.address 0.0.0.0

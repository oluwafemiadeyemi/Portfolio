FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8002 8502
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8002 & streamlit run dashboard/app.py --server.port 8502 --server.address 0.0.0.0"]

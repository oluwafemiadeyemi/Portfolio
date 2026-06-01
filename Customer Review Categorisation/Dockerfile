FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8007 8507
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8007 & streamlit run dashboard/app.py --server.port 8507 --server.address 0.0.0.0"]

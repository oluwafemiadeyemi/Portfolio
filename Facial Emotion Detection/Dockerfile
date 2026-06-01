FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8005 8505
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8005 & streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0"]

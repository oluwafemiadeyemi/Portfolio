FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8004 8504
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8004 & streamlit run dashboard/app.py --server.port 8504 --server.address 0.0.0.0"]

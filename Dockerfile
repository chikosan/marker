FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir pip && \
    pip install --no-cache-dir fastapi==0.115.4 uvicorn==0.32.0 python-multipart==0.0.16 && \
    pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "marker.scripts.server:app", "--host", "0.0.0.0", "--port", "8000"]

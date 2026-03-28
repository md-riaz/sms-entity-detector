# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies needed for tokenizers/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create runtime directories (also created at startup, but good to have in image)
RUN mkdir -p /app/data/queue /app/data/logs /app/data/sqlite

# Expose custom port
EXPOSE 8117

# Default command: run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8117"]

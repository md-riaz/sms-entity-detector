# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

ARG HF_MODEL_NAME=Davlan/xlm-roberta-base-ner-hrl
ARG PRELOAD_HF_MODEL=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN grep -v '^torch==' requirements.txt > requirements.docker.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu \
    && pip install -r requirements.docker.txt \
    && rm -f requirements.docker.txt

COPY . .

RUN mkdir -p /app/data/queue /app/data/logs /app/data/sqlite "$HF_HOME" \
    && if [ "$PRELOAD_HF_MODEL" = "true" ]; then python -c "from transformers import pipeline; pipeline('ner', model='${HF_MODEL_NAME}', aggregation_strategy='simple', device=-1); print('Preloaded model: ${HF_MODEL_NAME}')"; fi

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/huggingface /opt/huggingface
COPY . .

RUN mkdir -p /app/data/queue /app/data/logs /app/data/sqlite

EXPOSE 8117

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8117"]

# syntax=docker/dockerfile:1
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# NOTE: --mount=type=cache keeps downloaded wheels in a persistent BuildKit
# cache across builds (even failed ones). Without it, a slow/interrupted
# download that fails (hash mismatch, network drop, etc.) forces a full
# re-download from scratch on every retry -- painful on a slow connection
# when this layer pulls multi-GB packages like torch+CUDA.
COPY requirements-base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-base.txt

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bake model weights into image at a fixed path
ENV HF_HOME=/app/.cache/huggingface
RUN python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"

COPY . ./backend/

RUN adduser --disabled-password --gecos "" celeryuser && \
    chown -R celeryuser:celeryuser /app
USER celeryuser
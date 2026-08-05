FROM python:3.12-slim@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-base.txt .
RUN pip install --timeout 300 -r requirements-base.txt
COPY requirements.txt .
RUN pip install --timeout 300 -r requirements.txt
# Bake model weights into image at a fixed path
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=120
RUN --mount=type=cache,target=/app/.cache/huggingface \
    python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"
COPY . ./backend/
RUN adduser --disabled-password --gecos "" celeryuser && \
    chown -R celeryuser:celeryuser /app
USER celeryuser
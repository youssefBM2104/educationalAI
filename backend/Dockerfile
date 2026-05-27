FROM python:3.12-slim

WORKDIR /app

COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into image at a fixed path
ENV HF_HOME=/app/.cache/huggingface
RUN python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"

COPY . ./backend/

RUN adduser --disabled-password --gecos "" celeryuser && \
    chown -R celeryuser:celeryuser /app
USER celeryuser
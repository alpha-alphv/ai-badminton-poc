FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/data/yolo_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch (slim image). On the Grafilab GPU host we run this service
# natively (not in this container) using a CUDA-enabled torch wheel — see
# scripts/grafilab_setup.sh.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.0 torchvision==0.19.0

COPY services/inference_api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY shared /app/shared
COPY services/inference_api /app/services/inference_api

ENV PYTHONPATH=/app

# Model weights are downloaded on first request into $YOLO_CONFIG_DIR
# (which is on the /data volume, so the download is paid once and
# persists across container restarts).

EXPOSE 8000

CMD ["uvicorn", "services.inference_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

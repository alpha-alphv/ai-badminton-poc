FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY workers/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY shared /app/shared
COPY workers /app/workers

ENV PYTHONPATH=/app

CMD ["celery", "-A", "workers.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

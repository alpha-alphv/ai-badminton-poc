FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/web_app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY shared /app/shared
COPY services/web_app /app/services/web_app

ENV PYTHONPATH=/app \
    FLASK_APP=services.web_app.wsgi:app

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "services.web_app.wsgi:app"]

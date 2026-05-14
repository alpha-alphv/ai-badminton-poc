import logging

from fastapi import FastAPI

from .routers import health, jobs
from .settings import settings

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Badminton AI — Inference API",
    version="0.1.0",
    description="Video performance analysis service (YOLOv8 pipeline).",
)

app.include_router(health.router)
app.include_router(jobs.router, prefix="/v1")

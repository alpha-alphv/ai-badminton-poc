import uuid
from pathlib import Path

from celery import Celery
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from shared.db import Job, JobStatus, get_session

bp = Blueprint("upload", __name__, url_prefix="/upload")

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _celery() -> Celery:
    return Celery("web_app", broker=current_app.config["REDIS_URL"])


@bp.get("/")
def form():
    return render_template("upload.html")


@bp.post("/")
def submit():
    file = request.files.get("video")
    if not file or not file.filename:
        flash("No file selected", "error")
        return redirect(url_for("upload.form"))

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        flash(f"Unsupported extension {ext}", "error")
        return redirect(url_for("upload.form"))

    job_id = str(uuid.uuid4())
    data_dir = Path(current_app.config["DATA_DIR"]) / "uploads" / job_id
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"source{ext}"
    file.save(target)

    with get_session() as session:
        session.add(Job(
            id=job_id,
            filename=file.filename,
            video_path=str(target),
            status=JobStatus.pending,
        ))

    _celery().send_task("workers.tasks.run_job", args=[job_id])

    return redirect(url_for("jobs.detail", job_id=job_id))

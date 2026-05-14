from pathlib import Path

from flask import Blueprint, abort, current_app, render_template, send_from_directory

from shared.db import Job, get_session

bp = Blueprint("jobs", __name__, url_prefix="/jobs")


@bp.get("/")
def list_jobs():
    with get_session() as session:
        rows = session.query(Job).order_by(Job.created_at.desc()).limit(100).all()
        jobs = [
            {
                "id": j.id,
                "filename": j.filename,
                "status": j.status.value,
                "created_at": j.created_at,
                "progress": j.progress,
            }
            for j in rows
        ]
    return render_template("jobs.html", jobs=jobs)


@bp.get("/<job_id>")
def detail(job_id: str):
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            abort(404)
        data = {
            "id": job.id,
            "filename": job.filename,
            "status": job.status.value,
            "progress": job.progress,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
    return render_template("job_detail.html", job=data)


@bp.get("/<job_id>.json")
def detail_json(job_id: str):
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            abort(404)
        return {
            "id": job.id,
            "status": job.status.value,
            "progress": job.progress,
            "result": job.result,
            "error": job.error,
        }


@bp.get("/<job_id>/artifacts/<path:filename>")
def artifact(job_id: str, filename: str):
    artifact_dir = Path(current_app.config["DATA_DIR"]) / "artifacts" / job_id
    if not artifact_dir.exists():
        abort(404)
    return send_from_directory(artifact_dir, filename)

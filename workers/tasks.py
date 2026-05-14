import logging
import os
from base64 import b64decode
from pathlib import Path

import httpx

from shared.db import Job, JobStatus, get_session

from .celery_app import celery_app

logger = logging.getLogger(__name__)

INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://inference_api:8000")
INFERENCE_TRANSPORT = os.environ.get("INFERENCE_TRANSPORT", "upload")
INFERENCE_TIMEOUT_S = float(os.environ.get("INFERENCE_TIMEOUT_S", "3600"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))


_ARTIFACT_FIELDS = {
    "heatmap_png_b64": "heatmap.png",
    "sample_frame_png_b64": "sample_frame.png",
}


def _hydrate_artifacts(job_id: str, artifacts: dict | None) -> dict:
    """Decode any base64 artifact blobs into /data/artifacts/<job_id>/ so Flask
    can serve them. Returns a {logical_name: filename} map for the DB record."""
    if not artifacts:
        return {}
    out_dir = DATA_DIR / "artifacts" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for field, filename in _ARTIFACT_FIELDS.items():
        blob = artifacts.get(field)
        if not blob:
            continue
        (out_dir / filename).write_bytes(b64decode(blob))
        saved[field.removesuffix("_b64")] = filename
    return saved


def _post_to_inference(job_id: str, video_path: str) -> dict:
    if INFERENCE_TRANSPORT == "path":
        with httpx.Client(timeout=INFERENCE_TIMEOUT_S) as client:
            resp = client.post(
                f"{INFERENCE_API_URL}/v1/jobs/run",
                json={"job_id": job_id, "video_path": video_path},
            )
    else:
        with open(video_path, "rb") as f, httpx.Client(timeout=INFERENCE_TIMEOUT_S) as client:
            resp = client.post(
                f"{INFERENCE_API_URL}/v1/jobs/upload",
                data={"job_id": job_id},
                files={"video": (Path(video_path).name, f, "video/mp4")},
            )
    resp.raise_for_status()
    return resp.json()


@celery_app.task(name="workers.tasks.run_job", bind=True, max_retries=0)
def run_job(self, job_id: str) -> dict:
    logger.info("starting job %s (transport=%s url=%s)", job_id, INFERENCE_TRANSPORT, INFERENCE_API_URL)

    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError(f"job {job_id} not found")
        job.status = JobStatus.running
        job.progress = 10
        video_path = job.video_path

    try:
        payload = _post_to_inference(job_id, video_path)
    except Exception as exc:
        logger.exception("inference failed for job %s", job_id)
        with get_session() as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.failed
                job.error = str(exc)[:2000]
                job.progress = 0
        raise

    local_files = _hydrate_artifacts(job_id, payload.get("artifacts"))

    # Strip base64 blobs so the DB result stays small; keep references.
    stored = dict(payload)
    artifacts = dict(stored.get("artifacts") or {})
    for f in _ARTIFACT_FIELDS:
        artifacts.pop(f, None)
    artifacts["local_files"] = local_files
    stored["artifacts"] = artifacts

    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError(f"job {job_id} disappeared mid-run")
        job.status = JobStatus.succeeded
        job.progress = 100
        job.result = stored
        job.error = None

    logger.info("completed job %s", job_id)
    return stored

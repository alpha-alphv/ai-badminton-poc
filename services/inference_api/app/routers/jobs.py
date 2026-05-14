import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..pipeline.player_tracking import run_player_tracking
from ..schemas.jobs import JobRunRequest, JobRunResponse
from ..settings import settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/run", response_model=JobRunResponse)
def run_job(payload: JobRunRequest) -> JobRunResponse:
    """Path-based variant — for local dev where the inference service
    can read the video off a shared filesystem."""
    video_path = Path(payload.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")
    result = run_player_tracking(payload.job_id, video_path)
    return JobRunResponse(job_id=payload.job_id, **result)


@router.post("/upload", response_model=JobRunResponse)
async def run_upload(
    job_id: str = Form(...),
    video: UploadFile = File(...),
) -> JobRunResponse:
    """Multipart variant — used over the SSH tunnel from the local worker.
    The video bytes are streamed in the request body; the service saves
    them to its local /data dir before running the pipeline."""
    suffix = Path(video.filename or "source.mp4").suffix or ".mp4"
    target = Path(settings.data_dir) / "uploads" / job_id / f"source{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as f:
            shutil.copyfileobj(video.file, f)
    finally:
        await video.close()

    result = run_player_tracking(job_id, target)
    return JobRunResponse(job_id=job_id, **result)

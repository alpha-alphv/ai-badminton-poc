import json
import logging
import time
from base64 import b64encode
from pathlib import Path

import cv2
from ultralytics import YOLO

from ..settings import settings
from .visualize import render_heatmap, render_sample_frame

logger = logging.getLogger(__name__)

_model_cache: dict[str, YOLO] = {}


def _get_model(name: str) -> YOLO:
    if name not in _model_cache:
        logger.info("loading YOLO model %s on device=%s", name, settings.inference_device)
        _model_cache[name] = YOLO(name)
    return _model_cache[name]


def _video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps if fps > 0 else 0.0,
        }
    finally:
        cap.release()


def _tracker_path() -> str:
    p = Path(__file__).parent / "tracker_config.yaml"
    return str(p) if p.exists() else "bytetrack.yaml"


def run_player_tracking(job_id: str, video_path: Path) -> dict:
    """YOLOv8-pose + ByteTrack player tracking.

    Persists full per-frame tracks JSON to /data/artifacts/<job_id>/tracks.json
    and returns summaries + base64-encoded heatmap/sample frame so the caller
    (possibly behind an SSH tunnel) can hydrate artifacts locally.
    """
    start = time.time()
    metadata = _video_metadata(video_path)

    model = _get_model(settings.inference_model)
    max_frames = settings.inference_max_frames if settings.inference_max_frames > 0 else None

    tracks: dict[int, list[dict]] = {}
    foot_positions: list[tuple[int, int]] = []
    frames_processed = 0
    sample_frame_idx = None
    sample_payload = None  # (frame, xyxy, ids, kpts)

    results = model.track(
        source=str(video_path),
        tracker=_tracker_path(),
        persist=True,
        device=settings.inference_device,
        stream=True,
        verbose=False,
        classes=[0],
        conf=settings.inference_conf_threshold,
        iou=settings.inference_iou_threshold,
        imgsz=settings.inference_imgsz,
    )

    for r in results:
        if max_frames is not None and frames_processed >= max_frames:
            break

        boxes = r.boxes
        if boxes is None or boxes.id is None:
            frames_processed += 1
            continue

        ids = boxes.id.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        kpts_arr = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i].tolist()
            record = {
                "frame": frames_processed,
                "bbox": [x1, y1, x2, y2],
                "conf": float(conf[i]),
            }
            if kpts_arr is not None:
                record["keypoints"] = kpts_arr[i].tolist()
            tracks.setdefault(int(tid), []).append(record)
            foot_positions.append((int((x1 + x2) / 2), int(y2)))

        # Capture a mid-pipeline sample frame for the dashboard
        if sample_frame_idx is None and len(ids) >= 2:
            sample_frame_idx = frames_processed
            sample_payload = (r.orig_img.copy(), xyxy, ids, kpts_arr)

        frames_processed += 1

    out_dir = Path(settings.data_dir) / "artifacts" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_file = out_dir / "tracks.json"
    with tracks_file.open("w") as f:
        json.dump({"metadata": metadata, "tracks": tracks}, f)

    heatmap_path = out_dir / "heatmap.png"
    render_heatmap(foot_positions, metadata["width"], metadata["height"], heatmap_path)
    heatmap_b64 = b64encode(heatmap_path.read_bytes()).decode("ascii")

    sample_b64 = None
    if sample_payload is not None:
        sample_path = out_dir / "sample_frame.png"
        frame, xyxy, ids, kpts = sample_payload
        render_sample_frame(frame, xyxy, ids, kpts, sample_path)
        sample_b64 = b64encode(sample_path.read_bytes()).decode("ascii")

    summaries = []
    for tid, frames in tracks.items():
        confs = [f["conf"] for f in frames]
        summaries.append({
            "track_id": tid,
            "first_frame": frames[0]["frame"],
            "last_frame": frames[-1]["frame"],
            "frame_count": len(frames),
            "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
        })
    summaries.sort(key=lambda s: -s["frame_count"])

    elapsed = time.time() - start
    logger.info("job %s: %d frames, %d tracks, %.1fs", job_id, frames_processed, len(summaries), elapsed)

    return {
        "pipeline": "player_tracking",
        "model": settings.inference_model,
        "device": settings.inference_device,
        "frames_processed": frames_processed,
        "elapsed_seconds": elapsed,
        "metadata": metadata,
        "tracks": summaries,
        "artifacts": {
            "heatmap_png_b64": heatmap_b64,
            "sample_frame_png_b64": sample_b64,
            "tracks_json_path": str(tracks_file),
        },
    }

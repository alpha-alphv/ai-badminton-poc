import json
import logging
import time
from base64 import b64encode
from pathlib import Path

import cv2
from ultralytics import YOLO

from ..settings import settings
from .shuttle import ShuttleDetector, segment_rallies, shuttle_summary
from .visualize import (
    VideoAnnotator,
    draw_hud,
    draw_players,
    draw_shuttle,
    make_shuttle_trail,
    render_heatmap,
    render_sample_frame,
)

logger = logging.getLogger(__name__)

_model_cache: dict[str, YOLO] = {}


def _resolve_device(setting: str) -> str:
    """Translate 'auto' into 'cuda' when available, else 'cpu'."""
    if setting != "auto":
        return setting
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model(name: str) -> YOLO:
    if name not in _model_cache:
        logger.info("loading YOLO model %s", name)
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


def _current_rally_id(frame_idx: int, rallies: list[dict]) -> int | None:
    for r in rallies:
        if r["start_frame"] <= frame_idx <= r["end_frame"]:
            return r["rally_id"]
    return None


def run_player_tracking(job_id: str, video_path: Path) -> dict:
    """YOLOv8-pose + ByteTrack player tracking with shuttle heuristic.

    Writes:
      - tracks.json (per-frame player tracks)
      - annotated.mp4 (full overlay: bboxes, IDs, skeleton, shuttle marker)
      - heatmap.png (court coverage)
      - sample_frame.png (mid-rally snapshot)
      - rallies.json (segmented rally summaries)
    and returns small summaries + base64-encoded artifacts so the caller
    (possibly behind an SSH tunnel) can hydrate them locally.
    """
    start = time.time()
    metadata = _video_metadata(video_path)
    fps = metadata["fps"] or 25.0
    width, height = metadata["width"], metadata["height"]

    device = _resolve_device(settings.inference_device)
    logger.info("job %s: model=%s device=%s (requested=%s)",
                job_id, settings.inference_model, device, settings.inference_device)

    model = _get_model(settings.inference_model)
    max_frames = settings.inference_max_frames if settings.inference_max_frames > 0 else None

    tracks: dict[int, list[dict]] = {}
    foot_positions: list[tuple[int, int]] = []
    shuttle_track: list[dict] = []
    frames_processed = 0
    sample_frame_idx = None
    sample_payload = None  # (frame, xyxy, ids, kpts)

    out_dir = Path(settings.data_dir) / "artifacts" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "annotated.mp4"
    annotator = VideoAnnotator(annotated_path, width, height, fps)
    shuttle_detector = ShuttleDetector()
    shuttle_trail = make_shuttle_trail()

    results = model.track(
        source=str(video_path),
        tracker=_tracker_path(),
        persist=True,
        device=device,
        stream=True,
        verbose=False,
        classes=[0],
        conf=settings.inference_conf_threshold,
        iou=settings.inference_iou_threshold,
        imgsz=settings.inference_imgsz,
    )

    try:
        for r in results:
            if max_frames is not None and frames_processed >= max_frames:
                break

            frame = r.orig_img

            boxes = r.boxes
            xyxy = None
            ids = None
            kpts_arr = None
            if boxes is not None and boxes.id is not None:
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

                if sample_frame_idx is None and len(ids) >= 2:
                    sample_frame_idx = frames_processed
                    sample_payload = (frame.copy(), xyxy, ids, kpts_arr)

            shuttle_pos = shuttle_detector.detect(frame, exclude_boxes=xyxy)
            if shuttle_pos is not None:
                shuttle_trail.append(shuttle_pos)
                shuttle_track.append({
                    "frame": frames_processed,
                    "x": round(float(shuttle_pos[0]), 1),
                    "y": round(float(shuttle_pos[1]), 1),
                })

            canvas = frame.copy()
            if xyxy is not None and ids is not None:
                draw_players(canvas, xyxy, ids, kpts_arr)
            draw_shuttle(canvas, shuttle_pos, shuttle_trail)
            draw_hud(
                canvas,
                frame_idx=frames_processed,
                rally_id=None,  # filled in pass 2 if you want — keep HUD simple
                shuttle_detected=shuttle_pos is not None,
            )
            annotator.write(canvas)

            frames_processed += 1
    finally:
        annotator.close()

    tracks_file = out_dir / "tracks.json"
    with tracks_file.open("w") as f:
        json.dump({"metadata": metadata, "tracks": tracks}, f)

    rallies = segment_rallies(shuttle_track, frames_processed, fps)
    rallies_file = out_dir / "rallies.json"
    with rallies_file.open("w") as f:
        json.dump({"fps": fps, "rallies": rallies}, f)

    summary = shuttle_summary(shuttle_track, fps)

    heatmap_path = out_dir / "heatmap.png"
    render_heatmap(foot_positions, width, height, heatmap_path)
    heatmap_b64 = b64encode(heatmap_path.read_bytes()).decode("ascii")

    sample_b64 = None
    if sample_payload is not None:
        sample_path = out_dir / "sample_frame.png"
        frame, xyxy, ids, kpts = sample_payload
        render_sample_frame(frame, xyxy, ids, kpts, sample_path)
        sample_b64 = b64encode(sample_path.read_bytes()).decode("ascii")

    annotated_b64 = (
        b64encode(annotated_path.read_bytes()).decode("ascii")
        if annotated_path.exists() and annotated_path.stat().st_size > 0
        else None
    )

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
    logger.info(
        "job %s: %d frames, %d tracks, %d rallies, %d shuttle hits, %.1fs",
        job_id, frames_processed, len(summaries), len(rallies),
        summary["detections"], elapsed,
    )

    return {
        "pipeline": "player_tracking",
        "model": settings.inference_model,
        "device": device,
        "frames_processed": frames_processed,
        "elapsed_seconds": elapsed,
        "metadata": metadata,
        "tracks": summaries,
        "rallies": rallies,
        "shuttle": summary,
        "artifacts": {
            "heatmap_png_b64": heatmap_b64,
            "sample_frame_png_b64": sample_b64,
            "annotated_video_mp4_b64": annotated_b64,
            "tracks_json_path": str(tracks_file),
            "rallies_json_path": str(rallies_file),
        },
    }

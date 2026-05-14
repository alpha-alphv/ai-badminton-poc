"""Shuttle detection + rally segmentation.

Two detectors are exposed:

1. ``YoloShuttleDetector`` — wraps a custom-trained YOLO11 model
   (mirrors the Roboflow ``Shuttlecock.v1i.yolov11`` workflow used in
   the upstream Badminton_Analytics_Project). Path to the ``.pt``
   weights is taken from ``settings.shuttle_model_path``.

2. ``ShuttleDetector`` — MOG2 background-subtraction fallback for
   environments without trained weights. Same ``detect()`` signature so
   the pipeline can swap one for the other.

Rally segmentation: a rally is a contiguous stretch of frames where the
shuttle is detected with at most ``rally_gap_frames`` of silence in
between. Periods longer than that delimit rally boundaries.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShuttleDetector:
    """Motion-based shuttle candidate detector with simple linking.

    The search radius grows with consecutive misses so a brief occlusion
    doesn't permanently lock the detector to a stale anchor, and ``_last``
    is cleared after ``max_misses`` empty frames to allow re-acquisition
    elsewhere on the court (e.g. after a serve from the other side).
    """

    min_area: int = 6
    max_area: int = 1200
    max_aspect: float = 3.0
    max_link_dist_px: float = 220.0
    history: int = 200
    warmup_frames: int = 15
    max_misses: int = 30          # frames of silence before _last is dropped
    miss_radius_growth: float = 1.6  # search radius multiplier per missed frame
    miss_radius_cap: float = 4.0     # ceiling on the growth multiplier
    exclude_shrink: float = 0.15  # shrink player bboxes inward by this fraction

    _bg: Any = field(init=False, default=None, repr=False)
    _last: tuple[float, float] | None = field(init=False, default=None, repr=False)
    _misses: int = field(init=False, default=0, repr=False)
    _frame_idx: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=24, detectShadows=False
        )

    def _shrink_box(self, x1: float, y1: float, x2: float, y2: float):
        """Erode a bbox inward so the shuttle near a player's racquet isn't masked out."""
        if self.exclude_shrink <= 0:
            return int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1
        dx = w * self.exclude_shrink
        dy = h * self.exclude_shrink
        return int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy)

    def detect(self, frame: np.ndarray, exclude_boxes=None) -> tuple[float, float] | None:
        """Return the best shuttle candidate (x, y) for this frame, or None."""
        self._frame_idx += 1
        mask = self._bg.apply(frame)
        # MOG2 needs a few frames to stabilise; suppress output during warmup.
        if self._frame_idx <= self.warmup_frames:
            return None
        mask = cv2.medianBlur(mask, 3)
        # Drop player regions so we don't latch onto a swinging arm. Shrink
        # each box inward so a shuttle right at the racquet head still passes.
        if exclude_boxes is not None:
            for x1, y1, x2, y2 in exclude_boxes:
                sx1, sy1, sx2, sy2 = self._shrink_box(x1, y1, x2, y2)
                if sx2 > sx1 and sy2 > sy1:
                    cv2.rectangle(mask, (sx1, sy1), (sx2, sy2), 0, -1)

        # Search radius expands with consecutive misses (shuttle may have
        # travelled further during the gap), capped to avoid latching onto noise.
        if self._last is not None and self._misses > 0:
            grow = min(1.0 + self.miss_radius_growth * self._misses, self.miss_radius_cap)
        else:
            grow = 1.0
        link_radius = self.max_link_dist_px * grow

        num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best: tuple[float, float] | None = None
        best_score = -math.inf
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if w == 0 or h == 0:
                continue
            aspect = max(w / h, h / w)
            if aspect > self.max_aspect:
                continue
            cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
            # Prefer candidates that continue the previous trajectory.
            if self._last is not None:
                d = math.hypot(cx - self._last[0], cy - self._last[1])
                if d > link_radius:
                    continue
                score = -d - 0.05 * area
            else:
                # Cold start (or re-acquisition): prefer compact, mid-sized
                # blobs over either single-pixel noise or large motion smears.
                score = -abs(area - 40)
            if score > best_score:
                best_score = score
                best = (cx, cy)

        if best is not None:
            self._last = best
            self._misses = 0
        else:
            self._misses += 1
            if self._misses >= self.max_misses:
                self._last = None
        return best


@dataclass
class YoloShuttleDetector:
    """YOLO11 single-class shuttle detector.

    Loads weights from ``model_path`` (anything ultralytics ``YOLO()``
    accepts). ``class_names`` is a case-insensitive substring match —
    e.g. ``["shuttle", "shuttlecock"]`` picks up either label without
    needing to know the exact class index for the trained weights.

    Picks the highest-confidence shuttle box per frame; if the previous
    detection is available, prefers candidates close to it so quick
    rallies don't jitter to spurious far-away boxes.
    """

    model_path: str
    conf_threshold: float = 0.25
    class_names: tuple[str, ...] = ("shuttle", "shuttlecock")
    device: str = "cpu"
    imgsz: int = 640
    max_link_dist_px: float = 280.0

    _model: Any = field(init=False, default=None, repr=False)
    _shuttle_class_ids: set[int] = field(init=False, default_factory=set, repr=False)
    _last: tuple[float, float] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        from ultralytics import YOLO  # local import: optional dep at module load

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"shuttle weights not found: {self.model_path}")
        self._model = YOLO(self.model_path)
        names = self._model.names if isinstance(self._model.names, dict) else dict(enumerate(self._model.names))
        wanted = tuple(s.lower() for s in self.class_names)
        for cls_id, name in names.items():
            if any(w in str(name).lower() for w in wanted):
                self._shuttle_class_ids.add(int(cls_id))
        if not self._shuttle_class_ids:
            # Single-class shuttle models often just label class 0 "0" or "object" —
            # fall back to class 0 rather than silently returning no detections.
            logger.warning(
                "shuttle weights %s have no class matching %s; defaulting to class 0 (names=%s)",
                self.model_path, self.class_names, names,
            )
            self._shuttle_class_ids = {0}
        logger.info("YoloShuttleDetector ready: weights=%s shuttle_classes=%s",
                    self.model_path, sorted(self._shuttle_class_ids))

    def detect(self, frame: np.ndarray, exclude_boxes=None) -> tuple[float, float] | None:
        """Return the best shuttle (x, y) for this frame, or None.

        ``exclude_boxes`` is accepted for signature parity with the
        motion detector but ignored — a trained model handles
        player/shuttle separation natively.
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        best: tuple[float, float] | None = None
        best_score = -math.inf
        for i in range(len(xyxy)):
            if int(cls[i]) not in self._shuttle_class_ids:
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            score = float(confs[i])
            if self._last is not None:
                d = math.hypot(cx - self._last[0], cy - self._last[1])
                # Penalise candidates that teleport away from last known position.
                if d > self.max_link_dist_px:
                    score -= 0.3
            if score > best_score:
                best_score = score
                best = (cx, cy)

        if best is not None:
            self._last = best
        return best


def build_shuttle_detector(settings_obj, device: str):
    """Return a YOLO11 detector if weights are configured & loadable,
    otherwise the motion-based fallback. Logged either way so the user
    can see which path the run is on."""
    path = getattr(settings_obj, "shuttle_model_path", "") or ""
    if path and Path(path).exists():
        try:
            class_names = tuple(
                s.strip() for s in getattr(settings_obj, "shuttle_class_names", "shuttle").split(",")
                if s.strip()
            ) or ("shuttle",)
            return YoloShuttleDetector(
                model_path=path,
                conf_threshold=settings_obj.shuttle_conf_threshold,
                class_names=class_names,
                device=device,
                imgsz=settings_obj.inference_imgsz,
            )
        except Exception:
            logger.exception("failed to load YOLO11 shuttle detector at %s — falling back to motion heuristic", path)
    else:
        if path:
            logger.warning("shuttle_model_path=%s not found; using motion fallback", path)
        else:
            logger.info("shuttle_model_path empty; using motion-based shuttle fallback")
    return ShuttleDetector()


def segment_rallies(
    trajectory: list[dict],
    total_frames: int,
    fps: float,
    rally_gap_frames: int = 45,
    min_rally_frames: int = 15,
) -> list[dict]:
    """Cluster shuttle observations into rallies.

    ``trajectory`` is a list of ``{"frame": int, "x": float, "y": float}``.
    A rally is a run of detections separated by at most ``rally_gap_frames``
    of silence; runs shorter than ``min_rally_frames`` are dropped as noise.
    """
    if not trajectory:
        return []

    fps = fps if fps > 0 else 30.0
    frames = sorted(t["frame"] for t in trajectory)
    by_frame: dict[int, tuple[float, float]] = {int(t["frame"]): (float(t["x"]), float(t["y"])) for t in trajectory}

    rallies: list[dict] = []
    start = frames[0]
    prev = frames[0]
    hit_frames: list[int] = [frames[0]]

    def _finalize(start_f: int, end_f: int, hits: list[int]) -> None:
        if end_f - start_f + 1 < min_rally_frames:
            return
        path = [(by_frame[f][0], by_frame[f][1]) for f in hits]
        distance_px = sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        peak_speed = 0.0
        for i in range(len(hits) - 1):
            dt = (hits[i + 1] - hits[i]) / fps
            if dt <= 0:
                continue
            d = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            peak_speed = max(peak_speed, d / dt)
        rallies.append({
            "rally_id": len(rallies) + 1,
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "duration_seconds": round((end_f - start_f + 1) / fps, 2),
            "shuttle_hits": len(hits),
            "path_length_px": round(distance_px, 1),
            "peak_speed_px_per_s": round(peak_speed, 1),
        })

    for f in frames[1:]:
        if f - prev > rally_gap_frames:
            _finalize(start, prev, hit_frames)
            start = f
            hit_frames = []
        hit_frames.append(f)
        prev = f
    _finalize(start, prev, hit_frames)
    return rallies


def shuttle_summary(trajectory: list[dict], fps: float) -> dict:
    """Aggregate stats over the full shuttle trajectory."""
    if not trajectory:
        return {
            "detections": 0,
            "total_distance_px": 0.0,
            "mean_speed_px_per_s": 0.0,
            "peak_speed_px_per_s": 0.0,
        }
    fps = fps if fps > 0 else 30.0
    pts = sorted(trajectory, key=lambda t: t["frame"])
    total = 0.0
    peak = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1]["x"] - pts[i]["x"]
        dy = pts[i + 1]["y"] - pts[i]["y"]
        d = math.hypot(dx, dy)
        total += d
        dt = (pts[i + 1]["frame"] - pts[i]["frame"]) / fps
        if dt > 0:
            peak = max(peak, d / dt)
    span_frames = pts[-1]["frame"] - pts[0]["frame"]
    span_s = span_frames / fps if span_frames > 0 else 0.0
    return {
        "detections": len(pts),
        "total_distance_px": round(total, 1),
        "mean_speed_px_per_s": round(total / span_s, 1) if span_s > 0 else 0.0,
        "peak_speed_px_per_s": round(peak, 1),
    }

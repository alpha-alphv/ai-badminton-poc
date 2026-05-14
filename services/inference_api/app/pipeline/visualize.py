from collections import deque
from pathlib import Path

import cv2
import numpy as np

# COCO keypoint skeleton (used by yolov8-pose)
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),    # torso
    (0, 5), (0, 6),                         # head-shoulders
]

_TRACK_COLORS = [
    (66, 135, 245), (245, 66, 92), (66, 245, 132),
    (245, 191, 66), (191, 66, 245), (66, 224, 245),
    (245, 66, 161), (118, 245, 66),
]

_SHUTTLE_COLOR = (0, 255, 255)
_SHUTTLE_TRAIL_LEN = 18


def render_heatmap(positions: list[tuple[int, int]], width: int, height: int, output_path: Path) -> None:
    canvas = np.zeros((height, width), dtype=np.float32)
    for x, y in positions:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 25, 1.0, -1)
    if canvas.max() > 0:
        canvas = cv2.GaussianBlur(canvas, (61, 61), 0)
        canvas = canvas / canvas.max()
    colored = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_JET)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)


def draw_players(canvas: np.ndarray, boxes_xyxy, track_ids, keypoints) -> None:
    """Draw bboxes, IDs, and pose skeleton onto ``canvas`` in place."""
    for i, (xyxy, tid) in enumerate(zip(boxes_xyxy, track_ids)):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = _TRACK_COLORS[int(tid) % len(_TRACK_COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"ID {int(tid)}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if keypoints is not None and i < len(keypoints):
            kpts = keypoints[i]  # shape (17, 3) -> (x, y, conf)
            for a, b in _SKELETON:
                if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                    pa = (int(kpts[a, 0]), int(kpts[a, 1]))
                    pb = (int(kpts[b, 0]), int(kpts[b, 1]))
                    cv2.line(canvas, pa, pb, color, 2)
            for k in range(kpts.shape[0]):
                if kpts[k, 2] > 0.3:
                    cv2.circle(canvas, (int(kpts[k, 0]), int(kpts[k, 1])), 3, color, -1)


def draw_shuttle(canvas: np.ndarray, position: tuple[float, float] | None,
                 trail: deque | None = None) -> None:
    """Draw the shuttle marker plus a fading trail of recent positions."""
    if trail is not None:
        pts = list(trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (int(_SHUTTLE_COLOR[0] * alpha),
                     int(_SHUTTLE_COLOR[1] * alpha),
                     int(_SHUTTLE_COLOR[2] * alpha))
            cv2.line(canvas,
                     (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])),
                     color, 2)
    if position is not None:
        x, y = int(position[0]), int(position[1])
        cv2.circle(canvas, (x, y), 10, _SHUTTLE_COLOR, 2)
        cv2.circle(canvas, (x, y), 2, _SHUTTLE_COLOR, -1)
        cv2.putText(canvas, "shuttle", (x + 12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _SHUTTLE_COLOR, 1, cv2.LINE_AA)


def draw_hud(canvas: np.ndarray, *, frame_idx: int, rally_id: int | None,
             shuttle_detected: bool) -> None:
    """Top-left HUD: frame index, current rally, shuttle status."""
    h = 28
    cv2.rectangle(canvas, (0, 0), (260, h * 3 + 6), (15, 23, 38), -1)
    cv2.putText(canvas, f"frame {frame_idx}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    rally_txt = f"rally {rally_id}" if rally_id is not None else "rally —"
    cv2.putText(canvas, rally_txt, (10, 22 + h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    s_color = (0, 255, 0) if shuttle_detected else (120, 120, 120)
    s_txt = "shuttle: tracking" if shuttle_detected else "shuttle: lost"
    cv2.putText(canvas, s_txt, (10, 22 + 2 * h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1, cv2.LINE_AA)


class VideoAnnotator:
    """Encodes annotated frames into an H.264 MP4 (with mp4v fallback)."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps if fps > 0 else 25.0

        self._writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            self.fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def render_sample_frame(
    frame: np.ndarray,
    boxes_xyxy,
    track_ids,
    keypoints,
    output_path: Path,
) -> None:
    """Draw bboxes, IDs, and pose skeleton onto a frame and save as PNG."""
    canvas = frame.copy()
    draw_players(canvas, boxes_xyxy, track_ids, keypoints)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def make_shuttle_trail() -> deque:
    return deque(maxlen=_SHUTTLE_TRAIL_LEN)

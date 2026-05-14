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


def render_sample_frame(
    frame: np.ndarray,
    boxes_xyxy,
    track_ids,
    keypoints,
    output_path: Path,
) -> None:
    """Draw bboxes, IDs, and pose skeleton onto a frame and save as PNG."""
    canvas = frame.copy()
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

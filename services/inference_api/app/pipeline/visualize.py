import logging
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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


_ENCODER_UNPROBED = object()
_ffmpeg_encoder_cache: object = _ENCODER_UNPROBED


def _detect_ffmpeg_encoder() -> str | None:
    """Pick the best available H.264 encoder, preferring NVIDIA NVENC.

    Returns the codec name (``h264_nvenc`` or ``libx264``) or ``None`` if
    ffmpeg isn't on PATH. Result is cached for the process lifetime.
    Honours ``VIDEO_ENCODER`` env (``h264_nvenc`` | ``libx264`` | ``cv2``).
    """
    global _ffmpeg_encoder_cache
    if _ffmpeg_encoder_cache is not _ENCODER_UNPROBED:
        return _ffmpeg_encoder_cache  # type: ignore[return-value]

    forced = os.environ.get("VIDEO_ENCODER")
    if forced == "cv2":
        _ffmpeg_encoder_cache = None
        return None
    if shutil.which("ffmpeg") is None:
        _ffmpeg_encoder_cache = None
        return None
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        logger.warning("ffmpeg -encoders probe failed: %s", exc)
        _ffmpeg_encoder_cache = None
        return None

    if forced in ("h264_nvenc", "libx264") and forced in out:
        _ffmpeg_encoder_cache = forced
        return forced
    if forced and forced not in ("h264_nvenc", "libx264", "cv2"):
        logger.warning("ignoring unknown VIDEO_ENCODER=%s", forced)

    for codec in ("h264_nvenc", "libx264"):
        if codec in out:
            _ffmpeg_encoder_cache = codec
            return codec
    _ffmpeg_encoder_cache = None
    return None


class VideoAnnotator:
    """Encodes annotated BGR frames into an H.264 MP4.

    Prefers ``h264_nvenc`` (NVIDIA GPU) → ``libx264`` (CPU) via a piped
    ffmpeg subprocess so we get a browser-playable file and avoid
    OpenCV's flaky FFmpeg backend auto-selection (which on some Linux
    boxes mis-picks ``h264_v4l2m2m`` and dies). Falls back to
    ``cv2.VideoWriter`` with mp4v if ffmpeg isn't available.
    """

    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = fps if fps > 0 else 25.0
        self._proc: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        self._encoder: str = "none"

        encoder = _detect_ffmpeg_encoder()
        if encoder is not None:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", f"{self.fps:.3f}",
                "-i", "-",
                "-c:v", encoder,
            ]
            if encoder == "h264_nvenc":
                cmd += ["-preset", "p4", "-cq", "23"]
            else:
                cmd += ["-preset", "fast", "-crf", "23"]
            cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
                )
                self._encoder = encoder
                logger.info("VideoAnnotator using ffmpeg encoder=%s", encoder)
                return
            except Exception as exc:
                logger.exception("failed to launch ffmpeg (%s); falling back: %s", encoder, exc)
                self._proc = None

        self._writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        self._encoder = "cv2:mp4v"
        if not self._writer.isOpened():
            logger.error("cv2.VideoWriter could not open %s with mp4v", output_path)
            self._writer = None
            self._encoder = "none"

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                stderr = b""
                if self._proc.stderr is not None:
                    try:
                        stderr = self._proc.stderr.read()
                    except Exception:
                        pass
                logger.error("ffmpeg (%s) closed stdin early: %s",
                             self._encoder, stderr.decode("utf-8", errors="ignore"))
                self._proc = None
            return
        if self._writer is not None:
            self._writer.write(frame)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                rc = self._proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                rc = self._proc.wait()
                logger.error("ffmpeg encoder %s timed out and was killed", self._encoder)
            stderr = b""
            if self._proc.stderr is not None:
                try:
                    stderr = self._proc.stderr.read()
                except Exception:
                    pass
                finally:
                    try:
                        self._proc.stderr.close()
                    except Exception:
                        pass
            if rc != 0:
                logger.error("ffmpeg encoder %s exited rc=%d: %s",
                             self._encoder, rc, stderr.decode("utf-8", errors="ignore"))
            self._proc = None
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


def mux_audio_with_moviepy(
    silent_video_path: Path,
    source_video_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Use MoviePy to copy the audio track from ``source_video_path``
    onto the silent annotated mp4. Writes to ``output_path`` (defaults
    to overwriting ``silent_video_path``) and returns the final path.

    No-ops gracefully when the source has no audio or MoviePy can't open
    one of the files — the silent annotated video stays usable.
    """
    output_path = output_path or silent_video_path
    if not silent_video_path.exists() or silent_video_path.stat().st_size == 0:
        logger.warning("mux_audio: no annotated video at %s; skipping", silent_video_path)
        return silent_video_path
    try:
        # moviepy 1.x ships everything from moviepy.editor; the package
        # is heavy so import is local to keep cold-start fast.
        from moviepy.editor import AudioFileClip, VideoFileClip
    except Exception as exc:
        logger.warning("moviepy not available — leaving annotated video silent (%s)", exc)
        return silent_video_path

    video_clip = None
    source_clip = None
    audio_clip = None
    tmp_path = output_path.with_suffix(".muxed.mp4")
    try:
        source_clip = VideoFileClip(str(source_video_path))
        if source_clip.audio is None:
            logger.info("mux_audio: source %s has no audio track; skipping", source_video_path)
            return silent_video_path
        video_clip = VideoFileClip(str(silent_video_path))
        # Trim the audio to the annotated duration (we may have stopped
        # encoding early via inference_max_frames).
        audio_clip = source_clip.audio.subclip(0, min(video_clip.duration, source_clip.audio.duration))
        muxed = video_clip.set_audio(audio_clip)
        muxed.write_videofile(
            str(tmp_path),
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            threads=2,
            logger=None,
            verbose=False,
            temp_audiofile=str(tmp_path.with_suffix(".aac.tmp")),
            remove_temp=True,
        )
    except Exception as exc:
        logger.exception("mux_audio failed (%s); keeping silent annotated video", exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return silent_video_path
    finally:
        for c in (audio_clip, video_clip, source_clip):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass

    try:
        tmp_path.replace(output_path)
    except Exception as exc:
        logger.exception("mux_audio: rename %s -> %s failed (%s)", tmp_path, output_path, exc)
        return silent_video_path
    logger.info("mux_audio: wrote %s with audio from %s", output_path, source_video_path)
    return output_path

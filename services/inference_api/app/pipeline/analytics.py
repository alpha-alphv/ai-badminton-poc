"""Pandas + Matplotlib analytics over tracked players and shuttle.

Mirrors the analysis stage of the reference Badminton_Analytics_Project
notebook (player heatmaps, speed-over-time, court zone occupancy, shot
detection from shuttle acceleration spikes). Pure CPU — runs after the
YOLO/tracking pass has produced the raw per-frame data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server — no Qt/Tk display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def tracks_to_dataframe(tracks: dict[int, list[dict]]) -> pd.DataFrame:
    """Flatten ``{track_id: [{frame,bbox,conf,...}, ...]}`` to a long-form
    ``(track_id, frame, x, y, conf)`` frame where ``(x, y)`` is the foot
    midpoint (bottom-centre of the bbox), the same anchor the upstream
    notebook uses for court-coordinate analytics."""
    rows: list[dict] = []
    for tid, frames in tracks.items():
        for f in frames:
            x1, y1, x2, y2 = f["bbox"]
            rows.append({
                "track_id": int(tid),
                "frame": int(f["frame"]),
                "x": float((x1 + x2) / 2.0),
                "y": float(y2),
                "conf": float(f.get("conf", 0.0)),
            })
    if not rows:
        return pd.DataFrame(columns=["track_id", "frame", "x", "y", "conf"])
    return pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)


def shuttle_to_dataframe(shuttle_track: list[dict]) -> pd.DataFrame:
    if not shuttle_track:
        return pd.DataFrame(columns=["frame", "x", "y"])
    return pd.DataFrame(shuttle_track).sort_values("frame").reset_index(drop=True)


def compute_player_speeds(
    df_tracks: pd.DataFrame,
    fps: float,
    scale_m_per_px: float,
) -> pd.DataFrame:
    """Per-track speed series in m/s. Frame-to-frame Euclidean delta
    scaled by ``scale_m_per_px`` (≈ court-length-m / image-width-px)."""
    if df_tracks.empty:
        return pd.DataFrame(columns=["track_id", "frame", "speed_mps", "dist_m"])
    fps = fps if fps > 0 else 30.0
    out: list[pd.DataFrame] = []
    for tid, g in df_tracks.groupby("track_id"):
        g = g.sort_values("frame")
        dx = g["x"].diff().to_numpy()
        dy = g["y"].diff().to_numpy()
        dist_m = np.hypot(dx, dy) * scale_m_per_px
        speed = dist_m * fps
        df = pd.DataFrame({
            "track_id": tid,
            "frame": g["frame"].to_numpy(),
            "speed_mps": speed,
            "dist_m": dist_m,
        }).iloc[1:]  # drop the first row whose diff is NaN
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["track_id", "frame", "speed_mps", "dist_m"]
    )


def render_speed_chart(
    df_speeds: pd.DataFrame, fps: float, output_path: Path
) -> None:
    """Speed-over-time line plot, one series per track. Time on X, m/s on Y."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    if df_speeds.empty:
        ax.text(0.5, 0.5, "No tracks detected", ha="center", va="center",
                transform=ax.transAxes)
    else:
        fps = fps if fps > 0 else 30.0
        # Light rolling smoothing per track so the plot reads as a trace
        # rather than per-frame noise.
        for tid, g in df_speeds.groupby("track_id"):
            g = g.sort_values("frame")
            t = g["frame"].to_numpy() / fps
            smoothed = g["speed_mps"].rolling(window=int(fps), min_periods=1).mean()
            ax.plot(t, smoothed, label=f"Track {int(tid)}", linewidth=1.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Player speed over time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_zone_occupancy(
    df_tracks: pd.DataFrame,
    width: int,
    height: int,
    output_path: Path,
    rows: int = 3,
    cols: int = 3,
) -> dict[str, list[list[int]]]:
    """Quantise foot positions into a ``rows x cols`` court grid and
    plot a stacked bar of per-track zone occupancy. Returns the raw
    occupancy matrix keyed by track id so callers can persist the
    numbers alongside the chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    matrices: dict[str, list[list[int]]] = {}
    if df_tracks.empty:
        ax.text(0.5, 0.5, "No tracks detected", ha="center", va="center",
                transform=ax.transAxes)
    else:
        bin_x = np.linspace(0, max(width, 1), cols + 1)
        bin_y = np.linspace(0, max(height, 1), rows + 1)
        labels = [f"Z{r}{c}" for r in range(rows) for c in range(cols)]
        zone_counts: dict[int, np.ndarray] = {}
        for tid, g in df_tracks.groupby("track_id"):
            hist, _, _ = np.histogram2d(
                g["x"], g["y"], bins=[bin_x, bin_y]
            )
            counts = hist.astype(int)
            zone_counts[int(tid)] = counts
            matrices[str(int(tid))] = counts.tolist()

        bottoms = np.zeros(len(labels), dtype=float)
        for tid, counts in zone_counts.items():
            flat = counts.T.flatten()  # transpose so rows iterate top→bottom
            ax.bar(labels, flat, bottom=bottoms, label=f"Track {tid}")
            bottoms += flat
        ax.set_ylabel("Frames in zone")
        ax.set_title(f"Court zone occupancy ({rows}x{cols} grid)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return matrices


def render_player_heatmap_mpl(
    df_tracks: pd.DataFrame,
    width: int,
    height: int,
    output_path: Path,
    bins: int = 80,
    contrast_power: float = 0.5,
) -> None:
    """Matplotlib heatmap — per-track 2D histogram with contrast boost
    so low-density zones stay visible (upstream notebook's STEP 6)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df_tracks.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
        ax.text(0.5, 0.5, "No tracks detected", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    track_ids = sorted(df_tracks["track_id"].unique())
    n = len(track_ids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), dpi=110, squeeze=False)
    for i, tid in enumerate(track_ids):
        g = df_tracks[df_tracks["track_id"] == tid]
        heatmap, _, _ = np.histogram2d(
            g["x"], g["y"], bins=bins,
            range=[[0, max(width, 1)], [0, max(height, 1)]],
        )
        heatmap = np.rot90(heatmap)
        heatmap = np.flipud(heatmap)
        peak = heatmap.max()
        if peak > 0:
            enhanced = np.power(heatmap / peak, contrast_power)
        else:
            enhanced = heatmap
        ax = axes[0, i]
        ax.imshow(enhanced, cmap="jet", extent=[0, width, height, 0], aspect="auto")
        ax.set_title(f"Track {int(tid)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Player movement heatmaps")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def detect_shots(
    df_shuttle: pd.DataFrame,
    df_tracks: pd.DataFrame,
    fps: float,
    scale_m_per_px: float,
    accel_z: float = 1.6,
    min_gap_frames: int = 4,
) -> list[dict]:
    """Find shot candidates by spikes in shuttle acceleration, then
    attribute each to the nearest player at that frame. Mirrors STEP 12
    of the reference notebook."""
    if df_shuttle.empty or len(df_shuttle) < 4:
        return []
    fps = fps if fps > 0 else 30.0
    s = df_shuttle.sort_values("frame").reset_index(drop=True)
    vx = s["x"].diff().to_numpy()
    vy = s["y"].diff().to_numpy()
    speed_px_per_frame = np.hypot(vx, vy)
    speed_mps = speed_px_per_frame * scale_m_per_px * fps
    accel = np.abs(np.diff(speed_mps))
    if accel.size == 0:
        return []
    threshold = float(np.nanmean(accel) + accel_z * np.nanstd(accel))

    spike_idx = np.where(accel > threshold)[0]
    merged: list[int] = []
    last = -10_000
    for i in spike_idx:
        if i - last > min_gap_frames:
            merged.append(int(i))
            last = int(i)

    shots: list[dict] = []
    prev_player: int | None = None
    for n, idx in enumerate(merged, start=1):
        idx = min(idx, len(s) - 1)
        fnum = int(s.loc[idx, "frame"])
        sx = float(s.loc[idx, "x"])
        sy = float(s.loc[idx, "y"])

        # Nearest player within ±3 frames of the shot.
        nearby = df_tracks[(df_tracks["frame"] >= fnum - 3) & (df_tracks["frame"] <= fnum + 3)]
        hitter: int | None = None
        if not nearby.empty:
            d = np.hypot(nearby["x"].to_numpy() - sx, nearby["y"].to_numpy() - sy)
            hitter = int(nearby.iloc[int(np.argmin(d))]["track_id"])

        # Force-alternate if the same track repeats (rallies trade shots).
        if hitter is not None and hitter == prev_player:
            others = [t for t in df_tracks["track_id"].unique() if t != hitter]
            if others:
                hitter = int(others[0])
        if hitter is not None:
            prev_player = hitter

        # Classify by speed + outgoing angle (notebook STEP 12 heuristic).
        nxt = min(idx + 1, len(s) - 1)
        dx = float(s.loc[nxt, "x"] - s.loc[idx, "x"])
        dy = float(s.loc[nxt, "y"] - s.loc[idx, "y"])
        angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))
        v_at = float(speed_mps[min(idx, len(speed_mps) - 1)])
        speed_75 = float(np.nanpercentile(speed_mps, 75))
        if v_at > speed_75:
            stype = "smash"
        elif angle_deg < 20:
            stype = "drive"
        elif angle_deg > 45:
            stype = "clear"
        else:
            stype = "net/drop"

        shots.append({
            "shot_number": n,
            "frame": fnum,
            "shot_type": stype,
            "player_track_id": hitter,
            "speed_mps": round(v_at, 2),
            "angle_deg": round(angle_deg, 1),
        })
    return shots


def build_analytics(
    tracks: dict[int, list[dict]],
    shuttle_track: list[dict],
    fps: float,
    width: int,
    height: int,
    court_length_m: float,
    out_dir: Path,
) -> dict:
    """Run all analytics + render artifacts. Returns a payload dict with
    PNG paths, per-player aggregates, and detected shots.
    """
    df_tracks = tracks_to_dataframe(tracks)
    df_shuttle = shuttle_to_dataframe(shuttle_track)

    scale_m_per_px = (court_length_m / width) if width > 0 else 0.0

    df_speeds = compute_player_speeds(df_tracks, fps, scale_m_per_px)
    shots = detect_shots(df_shuttle, df_tracks, fps, scale_m_per_px)

    speed_chart = out_dir / "speed_chart.png"
    zone_chart = out_dir / "zone_chart.png"
    heatmap_mpl = out_dir / "heatmap_mpl.png"
    render_speed_chart(df_speeds, fps, speed_chart)
    zone_matrix = render_zone_occupancy(df_tracks, width, height, zone_chart)
    render_player_heatmap_mpl(df_tracks, width, height, heatmap_mpl)

    # Per-player aggregates — pandas describe() but compact.
    player_summary: list[dict] = []
    if not df_speeds.empty:
        agg = df_speeds.groupby("track_id").agg(
            distance_m=("dist_m", "sum"),
            mean_speed_mps=("speed_mps", "mean"),
            peak_speed_mps=("speed_mps", "max"),
        ).reset_index()
        for row in agg.itertuples(index=False):
            player_summary.append({
                "track_id": int(row.track_id),
                "distance_m": round(float(row.distance_m), 2),
                "mean_speed_mps": round(float(row.mean_speed_mps), 2),
                "peak_speed_mps": round(float(row.peak_speed_mps), 2),
            })

    # Persist long-form CSVs alongside PNGs for downstream tools.
    if not df_tracks.empty:
        df_tracks.to_csv(out_dir / "tracks_long.csv", index=False)
    if not df_speeds.empty:
        df_speeds.to_csv(out_dir / "player_speeds.csv", index=False)
    if not df_shuttle.empty:
        df_shuttle.to_csv(out_dir / "shuttle_trajectory.csv", index=False)
    if shots:
        pd.DataFrame(shots).to_csv(out_dir / "shots.csv", index=False)

    return {
        "scale_m_per_px": scale_m_per_px,
        "player_summary": player_summary,
        "shots": shots,
        "zone_matrix": zone_matrix,
        "artifact_paths": {
            "speed_chart": speed_chart,
            "zone_chart": zone_chart,
            "heatmap_mpl": heatmap_mpl,
        },
    }

from pydantic import BaseModel, Field


class JobRunRequest(BaseModel):
    job_id: str = Field(..., description="Caller-supplied job identifier")
    video_path: str = Field(..., description="Path to a video readable by the service")


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str | None = None


class TrackSummary(BaseModel):
    track_id: int
    first_frame: int
    last_frame: int
    frame_count: int
    mean_confidence: float


class RallySummary(BaseModel):
    rally_id: int
    start_frame: int
    end_frame: int
    duration_seconds: float
    shuttle_hits: int
    path_length_px: float
    peak_speed_px_per_s: float


class ShuttleSummary(BaseModel):
    detections: int = 0
    total_distance_px: float = 0.0
    mean_speed_px_per_s: float = 0.0
    peak_speed_px_per_s: float = 0.0


class PlayerSummary(BaseModel):
    track_id: int
    distance_m: float = 0.0
    mean_speed_mps: float = 0.0
    peak_speed_mps: float = 0.0


class Shot(BaseModel):
    shot_number: int
    frame: int
    shot_type: str
    player_track_id: int | None = None
    speed_mps: float = 0.0
    angle_deg: float = 0.0


class JobArtifacts(BaseModel):
    heatmap_png_b64: str | None = None
    sample_frame_png_b64: str | None = None
    annotated_video_mp4_b64: str | None = None
    speed_chart_png_b64: str | None = None
    zone_chart_png_b64: str | None = None
    heatmap_mpl_png_b64: str | None = None
    tracks_json_path: str | None = None
    rallies_json_path: str | None = None


class JobRunResponse(BaseModel):
    job_id: str
    pipeline: str
    model: str | None = None
    device: str | None = None
    frames_processed: int = 0
    elapsed_seconds: float = 0.0
    metadata: VideoMetadata
    tracks: list[TrackSummary] = Field(default_factory=list)
    rallies: list[RallySummary] = Field(default_factory=list)
    shuttle: ShuttleSummary = Field(default_factory=ShuttleSummary)
    players: list[PlayerSummary] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    scale_m_per_px: float = 0.0
    artifacts: JobArtifacts = Field(default_factory=JobArtifacts)
    notes: str | None = None

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: str = "/data"
    log_level: str = "INFO"

    inference_device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0" | "0"
    inference_model: str = "yolov8s-pose.pt"
    inference_max_frames: int = 1000
    inference_conf_threshold: float = 0.4
    inference_iou_threshold: float = 0.5
    inference_imgsz: int = 640

    # Custom-trained YOLO11 shuttle detector. Empty string keeps the
    # motion-based fallback (MOG2) so the service still runs without
    # weights on disk.
    shuttle_model_path: str = ""
    shuttle_conf_threshold: float = 0.25
    shuttle_class_names: str = "shuttle,shuttlecock"  # comma-separated; case-insensitive substring match
    # Badminton singles court is 13.4 m long. Used to convert pixel speeds → m/s.
    court_length_m: float = 13.4


settings = Settings()

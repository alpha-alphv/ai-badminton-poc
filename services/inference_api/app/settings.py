from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: str = "/data"
    log_level: str = "INFO"

    inference_device: str = "cpu"
    inference_model: str = "yolov8s-pose.pt"
    inference_max_frames: int = 1000
    inference_conf_threshold: float = 0.4
    inference_iou_threshold: float = 0.5
    inference_imgsz: int = 640


settings = Settings()

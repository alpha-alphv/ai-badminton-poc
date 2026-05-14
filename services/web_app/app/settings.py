from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    flask_secret_key: str = "dev-secret-change-me"
    data_dir: str = "/data"
    redis_url: str = "redis://redis:6379/0"
    inference_api_url: str = "http://inference_api:8000"
    max_upload_mb: int = 4096
    log_level: str = "INFO"


settings = Settings()

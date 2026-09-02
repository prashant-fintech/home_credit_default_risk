"""Central configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    mlflow_tracking_uri: str = "mlruns"
    model_registry_name: str = "home-credit-lgbm"


settings = Settings()

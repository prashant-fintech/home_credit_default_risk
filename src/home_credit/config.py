"""Central configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    # MLflow 3.x rejects the file-store backend ("mlruns"); use a SQLite DB instead
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_registry_name: str = "home-credit-lgbm"
    model_alias: str = "champion"  # registry alias served by the API / predict script

    # Set to load CSVs straight from S3 instead of data/raw/
    # e.g. s3://home-credit-default-risk-405894863747/raw
    s3_data_uri: str | None = None


settings = Settings()

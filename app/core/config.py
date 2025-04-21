import os
import yaml
from pydantic import BaseSettings  # Thay đổi từ pydantic_settings.BaseSettings
from typing import Dict, List, Optional, Any
from functools import lru_cache


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "ml-multiclass-app"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API settings
    API_PREFIX: str = "/api/v1"

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH: str = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "data", "models"))
    CONFIG_PATH: str = os.environ.get("CONFIG_PATH", os.path.join(BASE_DIR, "configs"))

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


@lru_cache()
def load_config(config_file: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = os.path.join(settings.CONFIG_PATH, config_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config() -> Dict[str, Any]:
    """Get models configuration."""
    config_file = os.environ.get("MODEL_CONFIG", "model_config.yaml")
    return load_config(config_file)


def get_app_config() -> Dict[str, Any]:
    """Get application configuration."""
    config_file = os.environ.get("APP_CONFIG", "app_config.yaml")
    return load_config(config_file)
import os
from typing import List, Union, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables with sensible defaults."""

    PROJECT_NAME: str = "TruthLens"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./truthlens.db"
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Model and Inference Configuration
    MODEL_SOURCE: str = "huggingface"  # 'huggingface' or 'local'
    LOCAL_MODEL_PATH: str = "models/deberta-v3-liar"
    HF_MODEL_REPO: str = "soah-13/truthlens-deberta-v3-liar"
    HF_TOKEN: Optional[str] = None
    METRICS_FILE_PATH: str = "models/deberta-v3-liar/evaluation_results.json"
    MAX_INPUT_LENGTH: int = 5000
    MIN_INPUT_LENGTH: int = 3

    # Tavily Search API Configuration (Phase 3B)
    TAVILY_API_KEY: Optional[str] = None
    TAVILY_SEARCH_DEPTH: str = "basic"  # 'basic' or 'advanced'
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_TIMEOUT_SECONDS: float = 10.0

    # Telegram Bot Configuration (Phase 3C & Final Webhook)
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_WEBHOOK_SECRET: Optional[str] = None
    TELEGRAM_WEBHOOK_URL: Optional[str] = None
    API_BASE_URL: str = "http://127.0.0.1:8000"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                items = [item.strip(" \"'") for item in v[1:-1].split(",") if item.strip()]
                return items
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=[
            ".env",
            "backend/.env",
            os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        ],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

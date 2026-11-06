"""
TruthLens Telegram Bot Configuration
Loads bot token and API base URL from application settings or environment variables.
"""

from typing import Optional
from app.core.config import settings

BOT_TOKEN: Optional[str] = settings.TELEGRAM_BOT_TOKEN
API_BASE_URL: str = settings.API_BASE_URL.rstrip("/")
REQUEST_TIMEOUT: float = 30.0  # Seconds allowed for FastAPI analysis call
MAX_CLAIM_LENGTH: int = settings.MAX_INPUT_LENGTH
MIN_CLAIM_LENGTH: int = settings.MIN_INPUT_LENGTH

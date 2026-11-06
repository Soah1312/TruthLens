"""
TruthLens Telegram Bot Module
Provides a Telegram bot interface communicating with the FastAPI backend.
Communicates strictly via REST API (does not load RoBERTa model or Tavily client independently).
"""

from app.telegram.bot import create_bot_application, run_bot

__all__ = ["create_bot_application", "run_bot"]

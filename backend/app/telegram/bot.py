"""
TruthLens Telegram Bot Runner
Configures and launches the python-telegram-bot application via long polling.
"""

import sys
import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

from app.telegram.config import BOT_TOKEN
from app.telegram.handlers import (
    start_command,
    help_command,
    about_command,
    handle_claim_message,
    error_handler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger("truthlens.telegram.bot")


def create_bot_application(token: Optional[str] = None) -> Application:
    """
    Constructs and configures the Telegram Application instance with all command
    and message handlers registered.
    """
    active_token = token or BOT_TOKEN
    if not active_token or "your_telegram_bot_token" in active_token.lower():
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not configured. Please set TELEGRAM_BOT_TOKEN "
            "in your environment or .env file before running the bot."
        )

    builder = ApplicationBuilder().token(active_token)
    app = builder.build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))

    # Register text message handler for claim submissions
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_claim_message)
    )

    # Register centralized error handler
    app.add_error_handler(error_handler)

    return app


def run_bot() -> None:
    """Runs the Telegram bot using long polling."""
    try:
        app = create_bot_application()
        logger.info("Starting TruthLens Telegram bot polling...")
        app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except Exception as exc:
        logger.error("Failed to start Telegram bot: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    run_bot()

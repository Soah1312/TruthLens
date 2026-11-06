#!/usr/bin/env python3
"""
TruthLens Telegram Webhook Management CLI
Allows operators to register, inspect, and delete Telegram bot webhooks safely.
Credentials are read strictly from environment variables or settings without logging secrets.
"""

import os
import sys
import argparse
import asyncio
import logging

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telegram import Bot
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("truthlens.webhook_cli")


def get_configured_bot(token: str = None) -> Bot:
    """Instantiates a telegram Bot instance with credentials."""
    active_token = token or settings.TELEGRAM_BOT_TOKEN
    if not active_token or "your_telegram_bot_token" in active_token.lower():
        logger.error("TELEGRAM_BOT_TOKEN is not configured in settings or environment.")
        sys.exit(1)
    return Bot(token=active_token)


async def get_webhook_status(bot: Bot) -> None:
    """Retrieves and prints current webhook registration info."""
    logger.info("Fetching webhook info from Telegram API...")
    info = await bot.get_webhook_info()
    
    print("\n" + "=" * 55)
    print("           TRUTHLENS TELEGRAM WEBHOOK STATUS       ")
    print("=" * 55)
    print(f" Webhook URL            : {info.url if info.url else '(None / Not configured)'}")
    print(f" Has Custom Certificate : {info.has_custom_certificate}")
    print(f" Pending Update Count   : {info.pending_update_count}")
    print(f" Max Connections        : {info.max_connections or 'Default (40)'}")
    print(f" Allowed Updates        : {info.allowed_updates or 'All'}")
    if info.last_error_date:
        print(f" Last Error Date        : {info.last_error_date}")
        print(f" Last Error Message     : {info.last_error_message}")
    if info.last_synchronization_error_date:
        print(f" Sync Error Date        : {info.last_synchronization_error_date}")
    print("=" * 55 + "\n")


async def register_webhook(bot: Bot, url: str, secret: str = None, drop_pending: bool = False) -> None:
    """Registers the webhook URL with optional secret token."""
    target_url = url or settings.TELEGRAM_WEBHOOK_URL
    if not target_url:
        logger.error("No webhook URL provided. Pass --url or set TELEGRAM_WEBHOOK_URL.")
        sys.exit(1)

    if not target_url.startswith("https://"):
        logger.error("Telegram requires an HTTPS webhook URL. Provided: %s", target_url)
        sys.exit(1)

    secret_token = secret or settings.TELEGRAM_WEBHOOK_SECRET
    masked_url = target_url.split("?")[0]
    logger.info("Registering webhook: %s (Drop pending: %s)", masked_url, drop_pending)

    success = await bot.set_webhook(
        url=target_url,
        secret_token=secret_token if secret_token else None,
        drop_pending_updates=drop_pending,
        allowed_updates=["message"],
    )

    if success:
        logger.info("✅ Webhook successfully registered with Telegram!")
        await get_webhook_status(bot)
    else:
        logger.error("❌ Failed to register webhook.")
        sys.exit(1)


async def delete_webhook(bot: Bot, drop_pending: bool = False) -> None:
    """Deletes existing webhook registration to allow reverting to local long polling."""
    logger.info("Deleting webhook registration from Telegram...")
    success = await bot.delete_webhook(drop_pending_updates=drop_pending)
    if success:
        logger.info("✅ Webhook successfully removed! You may now run the bot via long polling.")
    else:
        logger.error("❌ Failed to delete webhook.")
        sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(
        description="TruthLens Telegram Webhook Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Inspect current webhook registration status.")
    group.add_argument("--register", action="store_true", help="Register a webhook URL with Telegram.")
    group.add_argument("--delete", action="store_true", help="Delete the current webhook (enables long polling).")

    parser.add_argument("--url", type=str, default=None, help="Webhook URL (e.g., https://your-backend.onrender.com/api/telegram/webhook).")
    parser.add_argument("--secret", type=str, default=None, help="Optional X-Telegram-Bot-Api-Secret-Token.")
    parser.add_argument("--drop-pending", action="store_true", help="Drop pending updates during registration or deletion.")

    args = parser.parse_args()

    bot = get_configured_bot()
    try:
        await bot.initialize()
        if args.status:
            await get_webhook_status(bot)
        elif args.register:
            await register_webhook(bot, url=args.url, secret=args.secret, drop_pending=args.drop_pending)
        elif args.delete:
            await delete_webhook(bot, drop_pending=args.drop_pending)
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

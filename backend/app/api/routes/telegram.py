"""
TruthLens API Route: /api/telegram/webhook
Handles incoming Telegram Bot webhook updates, validates secret tokens,
processes commands (/start, /help, /about), and analyzes plain-text claims.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from telegram import Bot, Update

from app.core.config import settings
from app.schemas.analyze import AnalyzeRequest
from app.api.routes.analyze import analyze_claim
from app.services.classifier import ModelNotAvailableError, ModelInferenceError
from app.telegram.formatter import format_analysis_message

logger = logging.getLogger("truthlens.routes.telegram")
router = APIRouter(prefix="/telegram", tags=["Telegram"])

# Texts for standard bot commands (shared with long-polling handlers)
START_TEXT = (
    "Welcome to TruthLens!\n\n"
    "TruthLens is an NLP-based fake news analysis system that evaluates claims "
    "using a fine-tuned RoBERTa model and external web search.\n\n"
    "Send me a claim or statement to begin."
)

HELP_TEXT = (
    "📖 TRUTHLENS USER GUIDE\n\n"
    "How to Submit a Claim:\n"
    "Simply type or paste any news claim, political statement, or assertion into this chat.\n\n"
    "How Evidence is Retrieved:\n"
    "TruthLens calls the Tavily Search API to retrieve up to 5 relevant external web sources "
    "to give you immediate factual context.\n\n"
    "The 6 Truthfulness Labels:\n"
    "• Pants-fire: Completely false, ridiculous fabrication with no basis in fact.\n"
    "• False: The statement is not accurate.\n"
    "• Barely-true: Contains a kernel of truth, but leaves out critical context or distorts facts.\n"
    "• Half-true: Partially accurate, but omits important information or context.\n"
    "• Mostly-true: Accurate, but needs minor clarification or additional details.\n"
    "• True: The statement is accurate and nothing significant is omitted.\n\n"
    "System Limitations:\n"
    "• Model predictions reflect linguistic and stylistic patterns learned from the LIAR benchmark.\n"
    "• Predictions are neural estimates, not verified fact-checking determinations.\n"
    "• External search results should always be independently inspected."
)

ABOUT_TEXT = (
    "ℹ️ ABOUT TRUTHLENS\n\n"
    "Project: TruthLens – An NLP-Based Fake News Detection and Robustness Evaluation System\n\n"
    "Architecture & Tech Stack:\n"
    "• Backend: FastAPI (Python 3.10+)\n"
    "• Neural Model: Fine-tuned RoBERTa-base (soah-13/truthlens-roberta-liar-meta)\n"
    "• Training Benchmark: LIAR Dataset (12,800+ statements, 6 canonical veracity classes)\n"
    "• Input Formulation: Sentence-pair format (statement + speaker/context metadata)\n"
    "• External Evidence: Tavily Search API\n"
    "• Interfaces: Telegram Bot & React Dashboard\n\n"
    "Academic Objective:\n"
    "To provide calibrated claim veracity analysis while empirically testing neural robustness "
    "against adversarial textual manipulations."
)


def get_telegram_bot() -> Optional[Bot]:
    """Instantiates a telegram Bot client safely without leaking tokens."""
    token = settings.TELEGRAM_BOT_TOKEN
    if not token or "your_telegram_bot_token" in token.lower():
        return None
    return Bot(token=token)


@router.post(
    "/webhook",
    summary="Telegram Webhook Receiver",
    description=(
        "Processes incoming Telegram webhook updates securely. "
        "Verifies the X-Telegram-Bot-Api-Secret-Token header when configured, "
        "handles standard commands, and dispatches claims to the TruthLens analysis engine."
    ),
    responses={
        200: {"description": "Webhook update received and processed."},
        403: {"description": "Invalid secret token."},
        503: {"description": "Telegram bot token is not configured on the server."},
    },
)
async def telegram_webhook(request: Request) -> JSONResponse:
    """Processes incoming Telegram updates dispatched by the Telegram Bot API."""
    # 1. Security Check: Validate secret token header if configured
    if settings.TELEGRAM_WEBHOOK_SECRET:
        secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not secret_header or secret_header != settings.TELEGRAM_WEBHOOK_SECRET:
            logger.warning("Rejected Telegram webhook request: invalid or missing secret token header.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid Telegram webhook secret token.",
            )

    bot = get_telegram_bot()
    if not bot:
        logger.error("Telegram webhook called but TELEGRAM_BOT_TOKEN is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram bot token is not configured on the backend.",
        )

    # 2. Extract and parse request body safely
    try:
        data = await request.json()
    except Exception as exc:
        logger.warning("Failed to decode Telegram webhook JSON payload: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": "malformed_json"},
        )

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": "invalid_payload_type"},
        )

    try:
        update = Update.de_json(data, bot)
    except Exception as exc:
        logger.warning("Failed to construct Update object from payload: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": "unparseable_update"},
        )

    if not update or not update.message:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": "no_message_in_update"},
        )

    chat_id = update.message.chat_id
    raw_text = (update.message.text or "").strip()

    if not raw_text:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": "non_text_message"},
        )

    # 3. Route Commands
    command = raw_text.split()[0].lower() if raw_text.startswith("/") else None

    try:
        if command == "/start":
            await bot.send_message(chat_id=chat_id, text=START_TEXT)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok", "action": "start"})

        if command == "/help":
            await bot.send_message(chat_id=chat_id, text=HELP_TEXT)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok", "action": "help"})

        if command == "/about":
            await bot.send_message(chat_id=chat_id, text=ABOUT_TEXT)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok", "action": "about"})

        # 4. Handle Plain-Text Claim Submissions
        if len(raw_text) < settings.MIN_INPUT_LENGTH:
            await bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Statement is too short. Please provide a statement of at least {settings.MIN_INPUT_LENGTH} characters.",
            )
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok", "action": "too_short"})

        if len(raw_text) > settings.MAX_INPUT_LENGTH:
            await bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Statement exceeds maximum length of {settings.MAX_INPUT_LENGTH} characters. Please submit a shorter claim.",
            )
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok", "action": "too_long"})

        # Send initial processing status message
        status_msg = await bot.send_message(
            chat_id=chat_id,
            text="🔍 Analyzing statement with TruthLens neural model and searching for web evidence...",
        )

        try:
            # Internal call to existing analyze_claim service
            analysis_response = await analyze_claim(
                AnalyzeRequest(
                    text=raw_text,
                    metadata=None,
                    include_evidence=True,
                )
            )
            formatted_report = format_analysis_message(raw_text, analysis_response.model_dump())

            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text=formatted_report,
                )
            except Exception:
                await bot.send_message(chat_id=chat_id, text=formatted_report)

        except ModelNotAvailableError:
            error_msg = (
                "⚠️ Model Service Unavailable\n\n"
                "The RoBERTa classification head is currently loading or unreachable. "
                "Please verify backend settings and try again in a moment."
            )
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text=error_msg,
            )
        except Exception as exc:
            logger.error("Internal error during webhook claim analysis: %s", exc, exc_info=True)
            fallback_err = "⚠️ An unexpected error occurred while analyzing the claim. Please try again later."
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text=fallback_err,
                )
            except Exception:
                await bot.send_message(chat_id=chat_id, text=fallback_err)

    except Exception as exc:
        logger.error("Failed to deliver Telegram response via bot API: %s", exc)

    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok"})

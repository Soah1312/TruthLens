"""
TruthLens Telegram Bot Command and Message Handlers
Handles /start, /help, /about commands and routes claim text submissions to the FastAPI backend.
"""

import logging
from typing import Optional
import httpx
from telegram import Update
from telegram.ext import ContextTypes

from app.telegram.config import (
    API_BASE_URL,
    REQUEST_TIMEOUT,
    MIN_CLAIM_LENGTH,
    MAX_CLAIM_LENGTH,
)
from app.telegram.formatter import format_analysis_message

logger = logging.getLogger("truthlens.telegram.handlers")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the official TruthLens welcome message."""
    welcome_text = (
        "Welcome to TruthLens!\n\n"
        "TruthLens is an NLP-based fake news analysis system that evaluates claims "
        "using a fine-tuned RoBERTa model and external web search.\n\n"
        "Send me a claim or statement to begin."
    )
    if update.message:
        await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explains how to submit claims, labels, and system limitations."""
    help_text = (
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
    if update.message:
        await update.message.reply_text(help_text)


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides architectural and model provenance details."""
    about_text = (
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
    if update.message:
        await update.message.reply_text(about_text)


async def handle_claim_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Processes incoming text claims by validating input and dispatching to FastAPI /api/analyze.
    """
    if not update.message or not update.message.text:
        return

    raw_text = update.message.text.strip()

    # Input length validation
    if len(raw_text) < MIN_CLAIM_LENGTH:
        await update.message.reply_text(
            f"⚠️ Statement is too short. Please provide a statement of at least {MIN_CLAIM_LENGTH} characters."
        )
        return

    if len(raw_text) > MAX_CLAIM_LENGTH:
        await update.message.reply_text(
            f"⚠️ Statement exceeds maximum length of {MAX_CLAIM_LENGTH} characters. Please submit a shorter claim."
        )
        return

    # Acknowledge user with a temporary processing indicator
    status_msg = await update.message.reply_text(
        "🔍 Analyzing statement with TruthLens neural model and searching for web evidence..."
    )

    analyze_url = f"{API_BASE_URL}/api/analyze"
    payload = {
        "text": raw_text,
        "metadata": None,
        "include_evidence": True,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(analyze_url, json=payload)

        if response.status_code == 200:
            analysis_data = response.json()
            formatted_report = format_analysis_message(raw_text, analysis_data)
            await status_msg.edit_text(formatted_report)
        elif response.status_code == 503:
            await status_msg.edit_text(
                "⚠️ Model Service Unavailable\n\n"
                "The RoBERTa classification model is currently loading or unreachable. "
                "Please verify backend settings and try again in a moment."
            )
        elif response.status_code == 422:
            detail = response.json().get("detail", "Invalid input parameters.")
            await status_msg.edit_text(f"⚠️ Validation Error: {detail}")
        else:
            await status_msg.edit_text(
                f"⚠️ TruthLens Analysis Error (HTTP {response.status_code}). Please try again later."
            )

    except httpx.ConnectError:
        logger.error("Failed to connect to FastAPI backend at: %s", analyze_url)
        await status_msg.edit_text(
            "⚠️ Backend Unreachable\n\n"
            f"Unable to connect to the TruthLens backend at `{API_BASE_URL}`.\n"
            "Please ensure the FastAPI server daemon is active and running."
        )
    except httpx.TimeoutException:
        logger.warning("Request to FastAPI backend timed out for claim: %s", raw_text[:50])
        await status_msg.edit_text(
            "⏱️ Analysis Request Timed Out\n\n"
            "The backend took too long to complete inference and evidence retrieval. "
            "Please try again in a few moments."
        )
    except Exception as exc:
        logger.error("Unexpected error handling Telegram claim message: %s", exc, exc_info=True)
        await status_msg.edit_text(
            "⚠️ An unexpected error occurred while analyzing the claim. Please try again later."
        )


async def error_handler(update: Optional[object], context: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs unexpected errors without exposing credentials or internal traces."""
    logger.error("Telegram bot error caught in handler: %s", context.error, exc_info=False)

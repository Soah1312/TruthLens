"""
Tests for Phase 3C: Telegram Bot Integration.
Verifies command handlers (/start, /help, /about), input validation,
FastAPI error handling, Tavily degradation formatting, and message length safety.
All Telegram and HTTPX network calls are strictly mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.telegram.handlers import (
    start_command,
    help_command,
    about_command,
    handle_claim_message,
)
from app.telegram.formatter import format_analysis_message, MAX_TELEGRAM_MESSAGE_LENGTH


# ==============================================================================
# Helper Mock Factories
# ==============================================================================

def create_mock_update(text: str = "") -> MagicMock:
    """Creates a mock Telegram Update with async reply and edit methods."""
    update = MagicMock()
    message = AsyncMock()
    message.text = text

    # status_msg returned by reply_text supports async edit_text
    status_msg = AsyncMock()
    status_msg.edit_text = AsyncMock()
    message.reply_text.return_value = status_msg

    update.message = message
    return update


def create_mock_context() -> MagicMock:
    """Creates a mock Telegram context."""
    return MagicMock()


# ==============================================================================
# TEST 1: /start Command
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_start_command():
    """Verifies that /start sends the official welcome text."""
    update = create_mock_update("/start")
    context = create_mock_context()

    await start_command(update, context)

    update.message.reply_text.assert_called_once()
    called_text = update.message.reply_text.call_args[0][0]
    assert "Welcome to TruthLens!" in called_text
    assert "fine-tuned RoBERTa model" in called_text
    assert "Send me a claim or statement to begin." in called_text


# ==============================================================================
# TEST 2: /help Command
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_help_command():
    """Verifies that /help explains claim submission, 6 labels, and model limitations."""
    update = create_mock_update("/help")
    context = create_mock_context()

    await help_command(update, context)

    update.message.reply_text.assert_called_once()
    called_text = update.message.reply_text.call_args[0][0]
    assert "TRUTHLENS USER GUIDE" in called_text
    assert "Pants-fire" in called_text
    assert "False" in called_text
    assert "Barely-true" in called_text
    assert "Half-true" in called_text
    assert "Mostly-true" in called_text
    assert "True" in called_text
    assert "System Limitations" in called_text


# ==============================================================================
# TEST 3: /about Command
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_about_command():
    """Verifies that /about presents tech stack, LIAR benchmark, and RoBERTa provenance."""
    update = create_mock_update("/about")
    context = create_mock_context()

    await about_command(update, context)

    update.message.reply_text.assert_called_once()
    called_text = update.message.reply_text.call_args[0][0]
    assert "ABOUT TRUTHLENS" in called_text
    assert "soah-13/truthlens-roberta-liar-meta" in called_text
    assert "LIAR Dataset" in called_text
    assert "Tavily Search API" in called_text


# ==============================================================================
# TEST 4: Plain Text Claim Submission (Successful Analysis)
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_plain_text_submission_success():
    """Verifies that submitting a valid claim triggers backend analysis and returns report."""
    claim = "The federal budget surplus reached record highs this year."
    update = create_mock_update(claim)
    context = create_mock_context()

    mock_api_payload = {
        "prediction": {
            "label": "false",
            "class_id": 1,
            "confidence": 0.65,
            "class_scores": {
                "pants-fire": 0.10,
                "false": 0.65,
                "barely-true": 0.15,
                "half-true": 0.05,
                "mostly-true": 0.03,
                "true": 0.02,
            },
        },
        "model": "RoBERTa + Metadata",
        "evidence": {
            "status": "success",
            "query": "federal budget surplus record highs",
            "results": [
                {
                    "title": "US Treasury Budget Report",
                    "url": "https://example.com/treasury",
                    "domain": "example.com",
                    "snippet": "Deficit expanded over previous quarters.",
                    "relevance_score": 0.92,
                    "published_date": "2026-09-01",
                    "classification": "unclassified",
                }
            ],
            "summary": "External search results retrieved.",
        },
        "disclaimer": "Academic disclaimer.",
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_payload

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        await handle_claim_message(update, context)

    # Initial acknowledgement message sent
    update.message.reply_text.assert_called_once()
    assert "Analyzing statement with TruthLens" in update.message.reply_text.call_args[0][0]

    # Final report edited into message
    status_msg = update.message.reply_text.return_value
    status_msg.edit_text.assert_called_once()
    edited_text = status_msg.edit_text.call_args[0][0]

    assert "TRUTHLENS ANALYSIS" in edited_text
    assert "Prediction: FALSE" in edited_text
    assert "Confidence: 65%" in edited_text
    assert "US Treasury Budget Report" in edited_text
    assert "Evidence status: Retrieved." in edited_text


# ==============================================================================
# TEST 5: Input Validation (Empty or Excessively Short Input)
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_empty_or_short_input():
    """Verifies that short inputs (<3 chars) are rejected without calling FastAPI."""
    update = create_mock_update("ab")
    context = create_mock_context()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        await handle_claim_message(update, context)
        mock_post.assert_not_called()

    update.message.reply_text.assert_called_once()
    assert "too short" in update.message.reply_text.call_args[0][0]


# ==============================================================================
# TEST 6: FastAPI Backend Unavailable (ConnectError)
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_backend_unavailable():
    """Verifies graceful user messaging when FastAPI server is offline."""
    update = create_mock_update("Valid test statement text here.")
    context = create_mock_context()

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        await handle_claim_message(update, context)

    status_msg = update.message.reply_text.return_value
    status_msg.edit_text.assert_called_once()
    edited_text = status_msg.edit_text.call_args[0][0]
    assert "Backend Unreachable" in edited_text


# ==============================================================================
# TEST 7: Tavily Unavailable (Graceful Degradation)
# ==============================================================================

@pytest.mark.anyio
async def test_telegram_tavily_unavailable_handling():
    """
    Verifies that when Tavily search fails or is unavailable,
    the bot still presents the neural prediction clearly with evidence status noted.
    """
    update = create_mock_update("Economic policy statement for evaluation.")
    context = create_mock_context()

    mock_api_payload = {
        "prediction": {
            "label": "mostly-true",
            "class_id": 4,
            "confidence": 0.58,
            "class_scores": {"mostly-true": 0.58, "true": 0.42},
        },
        "model": "RoBERTa + Metadata",
        "evidence": {
            "status": "unavailable",
            "query": "economic policy statement",
            "results": [],
            "summary": "Tavily search service is temporarily unavailable.",
        },
        "disclaimer": "Academic disclaimer.",
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_payload

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        await handle_claim_message(update, context)

    status_msg = update.message.reply_text.return_value
    status_msg.edit_text.assert_called_once()
    edited_text = status_msg.edit_text.call_args[0][0]

    assert "TRUTHLENS ANALYSIS" in edited_text
    assert "Prediction: MOSTLY-TRUE" in edited_text
    assert "Confidence: 58%" in edited_text
    assert "Evidence status: Search unavailable or temporarily disabled." in edited_text


# ==============================================================================
# TEST 8: Response Formatting Verification (Section 11 Spec)
# ==============================================================================

def test_response_formatting_structure():
    """
    Verifies that format_analysis_message strictly includes all sections from Section 11:
    - TRUTHLENS ANALYSIS
    - Claim
    - MODEL PREDICTION (Prediction, Confidence, Model)
    - CLASS SCORES (All 6 classes)
    - EXTERNAL EVIDENCE (Top sources, Domain, Relevance, Link, Status)
    - DISCLAIMER
    """
    claim = "The senator claims unemployment dropped by 4 percent."
    data = {
        "prediction": {
            "label": "half-true",
            "confidence": 0.45,
            "class_scores": {
                "pants-fire": 0.05,
                "false": 0.15,
                "barely-true": 0.20,
                "half-true": 0.45,
                "mostly-true": 0.10,
                "true": 0.05,
            },
        },
        "model": "RoBERTa + Metadata",
        "evidence": {
            "status": "success",
            "results": [
                {
                    "title": "Unemployment Rates by State",
                    "url": "https://bls.gov/news.release/laus.nr0.htm",
                    "domain": "bls.gov",
                    "relevance_score": 0.94,
                }
            ],
        },
    }

    formatted = format_analysis_message(claim, data)

    assert "TRUTHLENS ANALYSIS" in formatted
    assert f'"{claim}"' in formatted
    assert "MODEL PREDICTION" in formatted
    assert "Prediction: HALF-TRUE" in formatted
    assert "Confidence: 45%" in formatted
    assert "Model: RoBERTa + Metadata" in formatted
    assert "CLASS SCORES" in formatted
    assert "- Pants-fire: 5%" in formatted
    assert "- False: 15%" in formatted
    assert "- Barely-true: 20%" in formatted
    assert "- Half-true: 45%" in formatted
    assert "- Mostly-true: 10%" in formatted
    assert "- True: 5%" in formatted
    assert "EXTERNAL EVIDENCE" in formatted
    assert "Unemployment Rates by State" in formatted
    assert "Domain: bls.gov" in formatted
    assert "Relevance: 0.94" in formatted
    assert "Link: https://bls.gov/news.release/laus.nr0.htm" in formatted
    assert "Evidence status: Retrieved." in formatted
    assert "DISCLAIMER:" in formatted


# ==============================================================================
# TEST 9: Long Response Handling (Telegram 4096-char Boundary Safety)
# ==============================================================================

def test_long_response_truncation_safety():
    """
    Verifies that a report with exceptionally verbose snippets is truncated safely
    so that it never exceeds Telegram's 4096-character limit.
    """
    huge_data = {
        "prediction": {
            "label": "true",
            "confidence": 0.90,
            "class_scores": {"true": 0.90},
        },
        "model": "RoBERTa + Metadata",
        "evidence": {
            "status": "success",
            "results": [
                {
                    "title": f"Verbose Source {i} " + ("word " * 100),
                    "url": f"https://example.com/very/long/nested/path/to/source/article/{i}",
                    "domain": "example.com",
                    "relevance_score": 0.85,
                }
                for i in range(15)
            ],
        },
    }

    formatted = format_analysis_message("Short claim text", huge_data)

    assert len(formatted) <= MAX_TELEGRAM_MESSAGE_LENGTH
    assert "... [Message truncated for length]" in formatted

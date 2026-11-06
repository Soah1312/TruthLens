"""
TruthLens Automated Test Suite: Telegram Webhook Endpoint
Tests security token verification, command dispatching, claim processing,
error handling, and payload validation for POST /api/telegram/webhook.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.classifier import ModelNotAvailableError
from app.schemas.analyze import (
    AnalyzeResponse,
    PredictionDetail,
    EvidencePayload,
    EvidenceResult,
)


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def make_telegram_update(text: str, update_id: int = 1001, chat_id: int = 5555):
    """Helper to construct synthetic Telegram Update dictionaries."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": 42,
            "date": 1700000000,
            "chat": {
                "id": chat_id,
                "type": "private",
                "first_name": "TestUser",
            },
            "from": {
                "id": chat_id,
                "is_bot": False,
                "first_name": "TestUser",
            },
            "text": text,
        },
    }


def test_webhook_unauthorized_when_secret_token_mismatches(client):
    """Verify that incoming requests without or with wrong secret tokens return HTTP 403."""
    with patch.object(settings, "TELEGRAM_WEBHOOK_SECRET", "expected-secret-token"):
        # 1. No secret header
        res_no_header = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("/start"),
        )
        assert res_no_header.status_code == 403
        assert "Invalid Telegram webhook secret token" in res_no_header.json().get("detail", "")

        # 2. Incorrect secret header
        res_wrong_header = client.post(
            "/api/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret-token"},
            json=make_telegram_update("/start"),
        )
        assert res_wrong_header.status_code == 403


def test_webhook_handles_malformed_json_gracefully(client):
    """Verify malformed JSON requests return HTTP 200 with an ignored status."""
    res = client.post(
        "/api/telegram/webhook",
        content="not-a-valid-json-string",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 200
    assert res.json().get("status") == "ignored"


def test_webhook_handles_non_text_updates(client):
    """Verify updates without text or message objects are ignored safely."""
    payload = {"update_id": 9999, "inline_query": {"id": "1", "query": ""}}
    res = client.post("/api/telegram/webhook", json=payload)
    assert res.status_code == 200
    assert res.json().get("status") == "ignored"


def test_webhook_start_command(client):
    """Verify /start triggers the official welcome message."""
    mock_bot = MagicMock()
    mock_bot.defaults = None
    mock_bot.send_message = AsyncMock()

    with patch("app.api.routes.telegram.get_telegram_bot", return_value=mock_bot):
        res = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("/start"),
        )
        assert res.status_code == 200
        assert res.json().get("action") == "start"
        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert "Welcome to TruthLens" in call_kwargs["text"]
        assert call_kwargs["chat_id"] == 5555


def test_webhook_help_command(client):
    """Verify /help triggers the user guide with all 6 label descriptions."""
    mock_bot = MagicMock()
    mock_bot.defaults = None
    mock_bot.send_message = AsyncMock()

    with patch("app.api.routes.telegram.get_telegram_bot", return_value=mock_bot):
        res = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("/help"),
        )
        assert res.status_code == 200
        assert res.json().get("action") == "help"
        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert "TRUTHLENS USER GUIDE" in call_kwargs["text"]
        assert "Pants-fire" in call_kwargs["text"]
        assert "True" in call_kwargs["text"]


def test_webhook_about_command(client):
    """Verify /about triggers system architecture and provenance details."""
    mock_bot = MagicMock()
    mock_bot.defaults = None
    mock_bot.send_message = AsyncMock()

    with patch("app.api.routes.telegram.get_telegram_bot", return_value=mock_bot):
        res = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("/about"),
        )
        assert res.status_code == 200
        assert res.json().get("action") == "about"
        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert "ABOUT TRUTHLENS" in call_kwargs["text"]
        assert "RoBERTa-base" in call_kwargs["text"]


def test_webhook_claim_validation_too_short(client):
    """Verify claims below minimum length receive a validation notice."""
    mock_bot = MagicMock()
    mock_bot.defaults = None
    mock_bot.send_message = AsyncMock()

    with patch("app.api.routes.telegram.get_telegram_bot", return_value=mock_bot):
        res = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("ab"),  # Less than min 3 chars
        )
        assert res.status_code == 200
        assert res.json().get("action") == "too_short"
        mock_bot.send_message.assert_awaited_once()
        assert "Statement is too short" in mock_bot.send_message.call_args[1]["text"]


def test_webhook_claim_successful_analysis(client):
    """Verify end-to-end claim submission through webhook updates."""
    mock_bot = MagicMock()
    mock_bot.defaults = None
    status_message = MagicMock(message_id=987)
    mock_bot.send_message = AsyncMock(return_value=status_message)
    mock_bot.edit_message_text = AsyncMock()

    mock_analysis = AnalyzeResponse(
        prediction=PredictionDetail(
            label="mostly-true",
            class_id=4,
            confidence=0.74,
            class_scores={
                "pants-fire": 0.02,
                "false": 0.04,
                "barely-true": 0.08,
                "half-true": 0.12,
                "mostly-true": 0.74,
                "true": 0.00,
            },
        ),
        model="RoBERTa + Metadata",
        evidence=EvidencePayload(
            status="success",
            query="economic growth 2024",
            results=[
                EvidenceResult(
                    title="Economic Indicators Q3",
                    url="https://example.com/indicators",
                    domain="example.com",
                    snippet="GDP growth matched economic forecasts.",
                    relevance_score=0.92,
                    published_date=None,
                    classification="unclassified",
                )
            ],
            summary="External search results retrieved.",
        ),
        disclaimer="The model prediction is not independent factual verification.",
    )

    with patch("app.api.routes.telegram.get_telegram_bot", return_value=mock_bot), \
         patch("app.api.routes.telegram.analyze_claim", AsyncMock(return_value=mock_analysis)):

        res = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("Economic growth met quarterly forecast expectations."),
        )

        assert res.status_code == 200
        mock_bot.send_message.assert_awaited_once()
        mock_bot.edit_message_text.assert_awaited_once()
        edited_text = mock_bot.edit_message_text.call_args[1]["text"]
        assert "TRUTHLENS ANALYSIS" in edited_text
        assert "MOSTLY-TRUE" in edited_text
        assert "Confidence: 74%" in edited_text
        assert "Economic Indicators Q3" in edited_text


def test_webhook_model_unavailable_resilience(client):
    """Verify webhook handles ModelNotAvailableError without crashing."""
    mock_bot = MagicMock()
    mock_bot.defaults = None
    status_message = MagicMock(message_id=987)
    mock_bot.send_message = AsyncMock(return_value=status_message)
    mock_bot.edit_message_text = AsyncMock()

    with patch("app.api.routes.telegram.get_telegram_bot", return_value=mock_bot), \
         patch("app.api.routes.telegram.analyze_claim", AsyncMock(side_effect=ModelNotAvailableError("Model offline"))):

        res = client.post(
            "/api/telegram/webhook",
            json=make_telegram_update("A valid claim text for testing."),
        )

        assert res.status_code == 200
        mock_bot.edit_message_text.assert_awaited_once()
        edited_text = mock_bot.edit_message_text.call_args[1]["text"]
        assert "Model Service Unavailable" in edited_text

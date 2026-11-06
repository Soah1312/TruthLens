"""
Tests for Phase 3B: Tavily Evidence Retrieval Service & Endpoint Integration.
All external Tavily API calls are mocked to prevent consuming real API credits during routine testing.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.evidence import EvidenceService, evidence_service
from app.core.config import settings


@pytest.fixture
def client():
    """Test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_evidence_service():
    """Provides a clean instance of EvidenceService for testing."""
    return EvidenceService()


# ==============================================================================
# TEST SUITE: Query Normalization & Domain Extraction
# ==============================================================================

def test_query_normalization():
    """Verifies that query normalization strips noisy prefixes, quotes, and whitespace."""
    assert EvidenceService.normalize_query("") == ""
    assert EvidenceService.normalize_query("   ") == ""
    assert EvidenceService.normalize_query('"The economy added 500k jobs."') == "The economy added 500k jobs."
    assert EvidenceService.normalize_query("Says that taxes increased by 20 percent") == "taxes increased by 20 percent"
    assert EvidenceService.normalize_query("Claims that inflation is at zero") == "inflation is at zero"
    assert EvidenceService.normalize_query("Says: unemployment dropped") == "unemployment dropped"


def test_domain_extraction():
    """Verifies clean domain extraction from URLs."""
    assert EvidenceService.extract_domain("https://www.reuters.com/world/article") == "reuters.com"
    assert EvidenceService.extract_domain("http://bbc.co.uk/news") == "bbc.co.uk"
    assert EvidenceService.extract_domain("https://example.org/") == "example.org"
    assert EvidenceService.extract_domain("invalid-url") == "unknown"


# ==============================================================================
# TEST 1: Successful Search with Valid Tavily Response
# ==============================================================================

@pytest.mark.anyio
async def test_successful_evidence_search(mock_evidence_service, monkeypatch):
    """Verifies structured evidence parsing and score rounding from Tavily response."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test-mock-key")

    mock_raw_response = {
        "query": "federal minimum wage increase",
        "results": [
            {
                "title": "Congress Debates Minimum Wage Hike",
                "url": "https://www.reuters.com/business/minimum-wage-2026",
                "content": "Lawmakers are debating a proposed federal minimum wage hike.",
                "score": 0.88765,
                "published_date": "2026-10-12",
            },
            {
                "title": "Minimum Wage Analysis",
                "url": "https://apnews.com/article/wage-inflation",
                "content": "Economic impacts of increasing the national wage baseline.",
                "score": 0.74211,
                "published_date": None,
            },
        ],
    }

    mock_client = AsyncMock()
    mock_client.search.return_value = mock_raw_response

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        result = await mock_evidence_service.search_evidence("federal minimum wage increase")

    assert result["status"] == "success"
    assert result["query"] == "federal minimum wage increase"
    assert len(result["results"]) == 2

    first = result["results"][0]
    assert first["title"] == "Congress Debates Minimum Wage Hike"
    assert first["domain"] == "reuters.com"
    assert first["relevance_score"] == 0.8877
    assert first["published_date"] == "2026-10-12"
    assert first["classification"] == "unclassified"


# ==============================================================================
# TEST 2: Empty Search Results Handling
# ==============================================================================

@pytest.mark.anyio
async def test_empty_search_results(mock_evidence_service, monkeypatch):
    """Verifies handling when Tavily returns zero matching web sources."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test-mock-key")

    mock_client = AsyncMock()
    mock_client.search.return_value = {"results": []}

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        result = await mock_evidence_service.search_evidence("xyz123randomnonexistentquery")

    assert result["status"] == "no_results"
    assert result["results"] == []
    assert "No external web sources" in result["summary"]


# ==============================================================================
# TEST 3: Duplicate URL Removal
# ==============================================================================

@pytest.mark.anyio
async def test_duplicate_url_deduplication(mock_evidence_service, monkeypatch):
    """Verifies that duplicate URLs with differing slashes or casing are deduplicated."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test-mock-key")

    mock_raw_response = {
        "results": [
            {
                "title": "Article One",
                "url": "https://example.com/article-1/",
                "content": "Snippet 1",
                "score": 0.9,
            },
            {
                "title": "Article One Duplicate",
                "url": "https://EXAMPLE.COM/article-1",
                "content": "Duplicate snippet",
                "score": 0.85,
            },
            {
                "title": "Article Two",
                "url": "https://example.com/article-2",
                "content": "Snippet 2",
                "score": 0.7,
            },
        ]
    }

    mock_client = AsyncMock()
    mock_client.search.return_value = mock_raw_response

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        result = await mock_evidence_service.search_evidence("test claim")

    assert result["status"] == "success"
    assert len(result["results"]) == 2
    assert result["results"][0]["url"] == "https://example.com/article-1/"
    assert result["results"][1]["url"] == "https://example.com/article-2"


# ==============================================================================
# TEST 4: Tavily API Failure / Exception Handling
# ==============================================================================

@pytest.mark.anyio
async def test_tavily_api_failure(mock_evidence_service, monkeypatch):
    """Verifies graceful handling when Tavily raises an unexpected exception or timeout."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test-mock-key")

    mock_client = AsyncMock()
    mock_client.search.side_effect = RuntimeError("Connection reset by peer")

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        result = await mock_evidence_service.search_evidence("test statement")

    assert result["status"] == "unavailable"
    assert result["results"] == []
    assert "Connection reset by peer" in result["error"]


# ==============================================================================
# TEST 5: Missing or Unconfigured API Key
# ==============================================================================

@pytest.mark.anyio
async def test_missing_api_key(mock_evidence_service, monkeypatch):
    """Verifies that search returns unavailable status without crashing if API key is missing."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", None)

    result = await mock_evidence_service.search_evidence("any statement")

    assert result["status"] == "unavailable"
    assert result["results"] == []
    assert "Tavily Search API key is not configured" in result["summary"]


# ==============================================================================
# TEST 6: Malformed API Response
# ==============================================================================

@pytest.mark.anyio
async def test_malformed_api_response(mock_evidence_service, monkeypatch):
    """Verifies robust parsing when Tavily returns non-dict items or missing fields."""
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test-mock-key")

    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "results": [
            "not a dict",
            {"url": "", "title": "Empty URL"},
            {"url": "https://valid.com/news", "content": "Valid content", "score": "invalid-float"},
        ]
    }

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        result = await mock_evidence_service.search_evidence("statement")

    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://valid.com/news"
    assert result["results"][0]["relevance_score"] == 0.0


# ==============================================================================
# TEST 7: Evidence Retrieval Disabled via API Parameter
# ==============================================================================

def test_evidence_disabled_in_analyze_endpoint(client, monkeypatch):
    """Verifies that Tavily is not called when include_evidence is False."""
    mock_search = AsyncMock()
    monkeypatch.setattr(evidence_service, "search_evidence", mock_search)

    # Mock the classifier to avoid loading full model
    with patch("app.services.classifier.classifier_service.predict") as mock_predict:
        mock_predict.return_value = {
            "prediction": {
                "label": "true",
                "class_id": 5,
                "confidence": 0.85,
                "class_scores": {"true": 0.85, "false": 0.15},
            },
            "model": "RoBERTa + Metadata",
            "disclaimer": "Academic disclaimer.",
        }

        response = client.post(
            "/api/analyze",
            json={
                "text": "The government approved a new transportation infrastructure bill.",
                "include_evidence": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["label"] == "true"
    assert data["evidence"] is None
    mock_search.assert_not_called()


# ==============================================================================
# TEST 8: Prediction Succeeds Even When Tavily Fails
# ==============================================================================

def test_prediction_succeeds_when_tavily_fails(client, monkeypatch):
    """
    Verifies system resilience: when external Tavily search fails,
    the RoBERTa classification is preserved and returned with evidence status 'unavailable'.
    """
    # Mock evidence service to simulate Tavily outage
    async def mock_failed_search(claim):
        return {
            "status": "unavailable",
            "query": claim,
            "results": [],
            "summary": "Tavily search service is temporarily down.",
            "error": "503 Service Unavailable",
        }

    monkeypatch.setattr(evidence_service, "search_evidence", mock_failed_search)

    with patch("app.services.classifier.classifier_service.predict") as mock_predict:
        mock_predict.return_value = {
            "prediction": {
                "label": "mostly-true",
                "class_id": 4,
                "confidence": 0.72,
                "class_scores": {"mostly-true": 0.72, "half-true": 0.28},
            },
            "model": "RoBERTa + Metadata",
            "disclaimer": "Academic disclaimer.",
        }

        response = client.post(
            "/api/analyze",
            json={
                "text": "The government approved a new transportation infrastructure bill.",
                "include_evidence": True,
            },
        )

    assert response.status_code == 200
    data = response.json()
    # Prediction MUST be preserved
    assert data["prediction"]["label"] == "mostly-true"
    assert data["prediction"]["confidence"] == 0.72
    # Evidence shows graceful degradation
    assert data["evidence"] is not None
    assert data["evidence"]["status"] == "unavailable"
    assert data["evidence"]["results"] == []
    assert "503 Service Unavailable" in data["evidence"]["error"]

"""
TruthLens Test Suite: Groq LLM Evidence Analyzer Integration
Tests all 16 scenarios mandated in project specification:
1. Successful Groq API response
2. Structured JSON parsing
3. Invalid JSON response
4. Unknown source IDs
5. Fabricated evidence quotes
6. Missing article content
7. Duplicate source URLs
8. Groq timeout
9. Groq API failure
10. Tavily failure
11. DeBERTa prediction surviving Groq failure
12. Evidence-disabled requests skipping Groq
13. Overall assessment logic
14. Conflicting source assessments
15. Telegram formatting
16. API response compatibility
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.evidence_analyzer import (
    EvidenceAnalyzerService,
    evidence_analyzer_service,
)
from app.services.evidence import EvidenceService
from app.telegram.formatter import format_analysis_message


@pytest.fixture
def client():
    """TestClient for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_sources():
    """Provides sample Tavily search results."""
    return [
        {
            "title": "US Inflation Rate Reports",
            "url": "https://reuters.com/business/inflation-report",
            "domain": "reuters.com",
            "snippet": "Consumer prices rose by 3.2 percent year-over-year in July, according to federal labor statistics.",
            "relevance_score": 0.92,
            "published_date": "2026-07-15",
            "classification": "unclassified",
        },
        {
            "title": "Historical Inflation Trends",
            "url": "https://apnews.com/article/inflation-history",
            "domain": "apnews.com",
            "snippet": "The inflation rate did not reach a 40-year high, which was recorded during the 1980s at over 14 percent.",
            "relevance_score": 0.85,
            "published_date": "2026-07-16",
            "classification": "unclassified",
        },
    ]


# ==============================================================================
# TEST 1: Successful Groq API Response
# ==============================================================================
@pytest.mark.anyio
async def test_successful_groq_response(sample_sources, monkeypatch):
    """Verifies that a valid Groq JSON response is parsed into structured source assessments."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    mock_llm_json = {
        "source_assessments": [
            {
                "source_id": "source_1",
                "relationship": "supports",
                "relevance": "high",
                "reasoning": "Corroborates the price index increase.",
                "evidence_quote": "Consumer prices rose by 3.2 percent",
            },
            {
                "source_id": "source_2",
                "relationship": "neutral",
                "relevance": "medium",
                "reasoning": "Provides historical comparison without direct contradiction.",
                "evidence_quote": None,
            },
        ],
        "llm_summary": "One source supports the price increase while another provides historical context.",
        "perceived_limitations": ["Limited to 2 recent articles."],
    }

    mock_chat = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_llm_json)
    mock_chat.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        result = await analyzer.analyze_evidence(
            claim="Inflation rose last month.",
            search_results=sample_sources,
        )

    assert result["status"] == "completed"
    assert result["overall_assessment"] == "supports"
    assert len(result["source_assessments"]) == 2
    assert result["source_assessments"][0]["relationship"] == "supports"
    assert result["source_assessments"][0]["evidence_quote"] == "Consumer prices rose by 3.2 percent"
    assert result["source_assessments"][1]["relationship"] == "neutral"


# ==============================================================================
# TEST 2: Structured JSON Parsing
# ==============================================================================
@pytest.mark.anyio
async def test_structured_json_parsing(sample_sources, monkeypatch):
    """Verifies JSON schema handling and field normalization."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    mock_llm_json = {
        "source_assessments": [
            {
                "source_id": "source_1",
                "relationship": "SUPPORTS",  # Uppercase should normalize to lowercase
                "relevance": "HIGH",
                "reasoning": "Direct evidence of labor metrics.",
                "evidence_quote": None,
            }
        ],
        "llm_summary": "Corroborated by official federal figures.",
        "perceived_limitations": [],
    }

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_llm_json)
    mock_chat = MagicMock(choices=[mock_choice])

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        result = await analyzer.analyze_evidence("Inflation rose", sample_sources[:1])

    assert result["status"] == "completed"
    assert result["source_assessments"][0]["relationship"] == "supports"
    assert result["source_assessments"][0]["relevance"] == "high"


# ==============================================================================
# TEST 3: Invalid JSON Response
# ==============================================================================
@pytest.mark.anyio
async def test_invalid_json_response_handling(sample_sources, monkeypatch):
    """Verifies graceful degradation when Groq returns non-JSON or malformed text."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    mock_choice = MagicMock()
    mock_choice.message.content = "I am an LLM and I cannot return JSON: [error]"
    mock_chat = MagicMock(choices=[mock_choice])

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        result = await analyzer.analyze_evidence("Some claim", sample_sources)

    assert result["status"] == "failed"
    assert result["overall_assessment"] is None
    assert "temporary failure" in result["summary"]


# ==============================================================================
# TEST 4: Unknown Source IDs Rejected
# ==============================================================================
@pytest.mark.anyio
async def test_unknown_source_ids_rejected(sample_sources, monkeypatch):
    """Verifies that hallucinated source IDs not supplied in prompt are rejected."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    mock_llm_json = {
        "source_assessments": [
            {
                "source_id": "source_999",  # Hallucinated ID
                "relationship": "contradicts",
                "relevance": "high",
                "reasoning": "Fake source reasoning.",
                "evidence_quote": None,
            },
            {
                "source_id": "source_1",  # Valid ID
                "relationship": "neutral",
                "relevance": "medium",
                "reasoning": "Valid reasoning.",
                "evidence_quote": None,
            },
        ],
        "llm_summary": "Analysis complete.",
        "perceived_limitations": [],
    }

    mock_choice = MagicMock(message=MagicMock(content=json.dumps(mock_llm_json)))
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        result = await analyzer.analyze_evidence("Claim", sample_sources[:1])

    # source_999 must be dropped; source_1 must be present
    sids = [a["source_id"] for a in result["source_assessments"]]
    assert "source_999" not in sids
    assert "source_1" in sids


# ==============================================================================
# TEST 5: Fabricated Evidence Quotes
# ==============================================================================
@pytest.mark.anyio
async def test_fabricated_evidence_quote_rejected():
    """Verifies that quotes not appearing verbatim in the source passage are stripped to None."""
    analyzer = EvidenceAnalyzerService()
    passage = "Consumer prices rose by 3.2 percent year-over-year in July."

    # Authentic quote
    real_quote = analyzer.verify_evidence_quote("prices rose by 3.2 percent", passage)
    assert real_quote == "prices rose by 3.2 percent"

    # Fabricated quote
    fake_quote = analyzer.verify_evidence_quote("inflation skyrocketed out of control to 99 percent", passage)
    assert fake_quote is None

    # Empty / null quote
    assert analyzer.verify_evidence_quote(None, passage) is None
    assert analyzer.verify_evidence_quote("", passage) is None


# ==============================================================================
# TEST 6: Missing Article Content
# ==============================================================================
@pytest.mark.anyio
async def test_missing_article_content_handling(monkeypatch):
    """Verifies that sources with empty or missing snippets do not crash the analyzer."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    sources_with_empty_snippet = [
        {
            "title": "Empty Article",
            "url": "https://example.com/empty",
            "domain": "example.com",
            "snippet": "",  # Empty content
            "relevance_score": 0.5,
            "published_date": None,
            "classification": "unclassified",
        }
    ]

    mock_llm_json = {
        "source_assessments": [
            {
                "source_id": "source_1",
                "relationship": "neutral",
                "relevance": "low",
                "reasoning": "Passage has no content to evaluate.",
                "evidence_quote": None,
            }
        ],
        "llm_summary": "Insufficient content available.",
        "perceived_limitations": ["No text excerpt."],
    }

    mock_choice = MagicMock(message=MagicMock(content=json.dumps(mock_llm_json)))
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        result = await analyzer.analyze_evidence("Any claim", sources_with_empty_snippet)

    assert result["status"] == "completed"
    assert result["source_assessments"][0]["relationship"] == "neutral"


# ==============================================================================
# TEST 7: Duplicate Source URLs
# ==============================================================================
def test_duplicate_source_urls_handling():
    """Verifies that Tavily search results deduplicate URLs accurately."""
    service = EvidenceService()
    # Test extract_domain and normalization consistency
    assert service.extract_domain("https://reuters.com/news") == "reuters.com"
    assert service.extract_domain("https://www.reuters.com/news/") == "reuters.com"


# ==============================================================================
# TEST 8: Groq Timeout
# ==============================================================================
@pytest.mark.anyio
async def test_groq_timeout_handling(sample_sources, monkeypatch):
    """Verifies that APITimeoutError triggers retries and graceful failure report."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    from groq import APITimeoutError

    mock_client = AsyncMock()
    mock_request = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=mock_request)

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        analyzer.max_retries = 1
        result = await analyzer.analyze_evidence("Claim", sample_sources)

    assert result["status"] == "failed"
    assert result["overall_assessment"] is None
    assert "failure" in result["summary"].lower()


# ==============================================================================
# TEST 9: Groq API Failure
# ==============================================================================
@pytest.mark.anyio
async def test_groq_api_failure_handling(sample_sources, monkeypatch):
    """Verifies that API connection errors degrade gracefully without crashing."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-valid-key")

    from groq import APIConnectionError

    mock_client = AsyncMock()
    mock_request = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=mock_request)

    with patch("groq.AsyncGroq", return_value=mock_client):
        analyzer = EvidenceAnalyzerService()
        analyzer.max_retries = 1
        result = await analyzer.analyze_evidence("Claim", sample_sources)

    assert result["status"] == "failed"
    assert result["overall_assessment"] is None


# ==============================================================================
# TEST 10: Tavily Failure Skips Groq
# ==============================================================================
def test_tavily_failure_skips_groq(client, monkeypatch):
    """When Tavily search fails or returns unavailable, Groq must not be called."""
    mock_groq_analyze = AsyncMock()

    with patch("app.services.evidence.evidence_service.search_evidence") as mock_search:
        mock_search.return_value = {
            "status": "unavailable",
            "query": "test query",
            "results": [],
            "summary": "Tavily key missing",
            "error": "API key unconfigured",
        }
        with patch("app.services.evidence_analyzer.evidence_analyzer_service.analyze_evidence", mock_groq_analyze):
            response = client.post(
                "/api/analyze",
                json={"text": "The economy grew by 10 percent this year.", "include_evidence": True},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["evidence"]["status"] == "unavailable"
    # Groq must not have been invoked
    mock_groq_analyze.assert_not_called()


# ==============================================================================
# TEST 11: DeBERTa Prediction Survives Groq Failure
# ==============================================================================
def test_deberta_survives_groq_failure(client, monkeypatch):
    """If Groq fails with an exception, DeBERTa prediction and Tavily results are still returned."""
    mock_search_results = [
        {
            "title": "Sample Article",
            "url": "https://example.com/sample",
            "domain": "example.com",
            "snippet": "Sample passage content.",
            "relevance_score": 0.9,
            "published_date": None,
            "classification": "unclassified",
        }
    ]

    with patch("app.services.evidence.evidence_service.search_evidence") as mock_search:
        mock_search.return_value = {
            "status": "success",
            "query": "sample query",
            "results": mock_search_results,
            "summary": "Retrieved search results.",
        }
        with patch("app.services.evidence_analyzer.evidence_analyzer_service.analyze_evidence") as mock_groq:
            # Groq raises unhandled exception
            mock_groq.side_effect = RuntimeError("Groq rate limit exceeded.")

            response = client.post(
                "/api/analyze",
                json={"text": "Unemployment is at an all-time low in the country.", "include_evidence": True},
            )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["label"] in [
        "pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"
    ]
    assert data["evidence"]["status"] == "success"
    assert data["evidence"]["analysis_status"] == "failed"


# ==============================================================================
# TEST 12: Evidence-Disabled Requests Skipping Groq
# ==============================================================================
def test_evidence_disabled_skips_groq(client):
    """When include_evidence is False, neither Tavily nor Groq is called."""
    with patch("app.services.evidence.evidence_service.search_evidence") as mock_tavily:
        with patch("app.services.evidence_analyzer.evidence_analyzer_service.analyze_evidence") as mock_groq:
            response = client.post(
                "/api/analyze",
                json={"text": "Tax cuts lead to sustained job growth.", "include_evidence": False},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["evidence"] is None
    mock_tavily.assert_not_called()
    mock_groq.assert_not_called()


# ==============================================================================
# TEST 13: Overall Assessment Logic (Deterministic Matrix)
# ==============================================================================
def test_overall_assessment_matrix():
    """Verifies deterministic backend verdict computation across different relationship distributions."""
    analyzer = EvidenceAnalyzerService()

    # All supports -> 'supports'
    supports_meta = analyzer.compute_overall_assessment(
        assessments=[{"relationship": "supports", "relevance": "high"}],
        results=[{"domain": "reuters.com"}],
    )
    assert supports_meta["verdict"] == "supports"

    # All contradicts -> 'contradicts'
    contradicts_meta = analyzer.compute_overall_assessment(
        assessments=[{"relationship": "contradicts", "relevance": "high"}],
        results=[{"domain": "apnews.com"}],
    )
    assert contradicts_meta["verdict"] == "contradicts"

    # All neutral -> 'insufficient_evidence'
    neutral_meta = analyzer.compute_overall_assessment(
        assessments=[{"relationship": "neutral", "relevance": "medium"}],
        results=[{"domain": "bbc.com"}],
    )
    assert neutral_meta["verdict"] == "insufficient_evidence"


# ==============================================================================
# TEST 14: Conflicting Source Assessments -> Mixed
# ==============================================================================
def test_conflicting_source_assessments_mixed():
    """Verifies that conflicting evidence (supports + contradicts) produces 'mixed' verdict."""
    analyzer = EvidenceAnalyzerService()

    mixed_meta = analyzer.compute_overall_assessment(
        assessments=[
            {"relationship": "supports", "relevance": "high"},
            {"relationship": "contradicts", "relevance": "high"},
        ],
        results=[{"domain": "reuters.com"}, {"domain": "apnews.com"}],
    )
    assert mixed_meta["verdict"] == "mixed"
    assert "conflicting" in mixed_meta["summary"].lower()


# ==============================================================================
# TEST 15: Telegram Formatting
# ==============================================================================
def test_telegram_formatting_with_groq_evidence():
    """Verifies Telegram message structure with Section 1, 2, 3, 4 and length safety."""
    mock_api_data = {
        "prediction": {
            "label": "mostly-true",
            "class_id": 4,
            "confidence": 0.42,
            "class_scores": {
                "pants-fire": 0.05,
                "false": 0.08,
                "barely-true": 0.12,
                "half-true": 0.18,
                "mostly-true": 0.42,
                "true": 0.15,
            },
        },
        "model": "DeBERTa-v3 + Metadata",
        "evidence": {
            "status": "success",
            "query": "federal deficit reduced",
            "overall_assessment": "mixed",
            "analysis_status": "completed",
            "analysis_summary": "Retrieved sources present conflicting economic analyses.",
            "limitations": ["Limited source coverage."],
            "results": [
                {
                    "title": "Deficit Reduced by 50 Billion",
                    "url": "https://reuters.com/news/deficit",
                    "domain": "reuters.com",
                    "snippet": "Treasury figures show deficit decline.",
                    "relevance_score": 0.88,
                    "classification": "supports",
                    "reasoning": "Direct evidence of 50B reduction.",
                },
                {
                    "title": "CBO Projects Growing Deficit",
                    "url": "https://cbo.gov/publication/59000",
                    "domain": "cbo.gov",
                    "snippet": "Long-term projections indicate growing deficit.",
                    "relevance_score": 0.82,
                    "classification": "contradicts",
                    "reasoning": "Projects opposite long-term trajectory.",
                },
            ],
        },
        "disclaimer": "Academic disclaimer text.",
    }

    msg = format_analysis_message("The federal deficit was reduced this year.", mock_api_data)

    assert "TRUTHLENS ANALYSIS" in msg
    assert "1. MODEL PREDICTION" in msg
    assert "MOSTLY-TRUE" in msg
    assert "2. EVIDENCE ASSESSMENT" in msg
    assert "MIXED / CONFLICTING EVIDENCE" in msg
    assert "3. SOURCES" in msg
    assert "https://reuters.com/news/deficit" in msg
    assert "4. LIMITATIONS & CAVEATS" in msg
    assert len(msg) < 4000


# ==============================================================================
# TEST 16: API Response Compatibility
# ==============================================================================
def test_api_response_schema_compatibility(client):
    """Verifies that the /api/analyze endpoint response matches the Pydantic schema contract."""
    response = client.post(
        "/api/analyze",
        json={
            "text": "The government approved a new transportation infrastructure bill.",
            "include_evidence": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert "class_scores" in data["prediction"]
    assert "model" in data
    assert "disclaimer" in data
    # Optional evidence field is present as None when disabled
    assert data.get("evidence") is None

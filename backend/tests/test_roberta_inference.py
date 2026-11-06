"""
TruthLens Phase 2 & 3 - Comprehensive Test Suite for RoBERTa Inference
Covers:
1. Model loading (mocked)
2. Successful text prediction
3. Correct 6-class output mapping (pants-fire through true)
4. Missing text validation (422)
5. Empty / whitespace text validation (422)
6. Missing metadata handling (defaults to 'unknown')
7. API response structure verification (/api/analyze)
8. Live Hugging Face inference test
"""

import sys
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app
from app.core.config import settings
from app.services.classifier import (
    RoBERTaInferenceService,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    MODEL_IDENTIFIER,
    DEFAULT_DISCLAIMER,
)
from app.schemas.analyze import ClaimMetadata


@pytest.fixture
def client():
    """Provides a TestClient for FastAPI endpoints."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_service():
    """Provides an isolated RoBERTaInferenceService instance for testing."""
    service = RoBERTaInferenceService()
    return service


# ==============================================================================
# TEST 1: Model Loading (Mocked)
# ==============================================================================
def test_model_loading_mocked(mock_service):
    """Verifies model loading lifecycle and error handling with mocked transformers."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_mod, \
         patch("torch.cuda.is_available", return_value=False):

        mock_tok_instance = MagicMock()
        mock_mod_instance = MagicMock()
        mock_tok.return_value = mock_tok_instance
        mock_mod.return_value = mock_mod_instance

        mock_service.load_model()

        assert mock_service.is_loaded is True
        assert mock_service.load_error is None
        assert mock_service.tokenizer is mock_tok_instance
        assert mock_service.model is mock_mod_instance
        mock_mod_instance.eval.assert_called_once()


# ==============================================================================
# TEST 2: Successful Text Prediction (Mocked)
# ==============================================================================
def test_successful_text_prediction(mock_service):
    """Verifies successful forward pass and prediction formatting."""
    import torch

    # Simulated logits where class index 4 ('mostly-true') is the highest
    simulated_logits = torch.tensor([[-2.0, -1.0, 0.5, 1.0, 3.5, 0.2]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[0, 10, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = simulated_logits
    mock_model.return_value = mock_output

    mock_service.tokenizer = mock_tokenizer
    mock_service.model = mock_model
    mock_service.is_loaded = True
    mock_service.device = torch.device("cpu")

    result = mock_service.predict(
        text="The state reported a 3 percent decrease in sales tax.",
        metadata={"speaker": "State Treasurer", "context": "Budget Report"},
    )

    pred = result["prediction"]
    assert pred["label"] == "mostly-true"
    assert pred["class_id"] == 4
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["class_scores"]) == 6
    assert result["model"] == MODEL_IDENTIFIER
    assert result["disclaimer"] == DEFAULT_DISCLAIMER
    # Check probabilities sum to approximately 1.0
    total_prob = sum(pred["class_scores"].values())
    assert abs(total_prob - 1.0) < 0.05


# ==============================================================================
# TEST 3: Correct Six-Class Output Mapping
# ==============================================================================
@pytest.mark.parametrize("class_idx,expected_label", list(enumerate(LABEL_LIST)))
def test_correct_six_class_output_mapping(mock_service, class_idx, expected_label):
    """
    Verifies that every single canonical LIAR class index (0 to 5)
    maps exactly to its expected label.
    """
    import torch

    # Construct logits where class_idx is strongly dominant
    logits_values = [-5.0] * 6
    logits_values[class_idx] = 10.0
    simulated_logits = torch.tensor([logits_values])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = simulated_logits
    mock_model.return_value = mock_output

    mock_service.tokenizer = mock_tokenizer
    mock_service.model = mock_model
    mock_service.is_loaded = True
    mock_service.device = torch.device("cpu")

    result = mock_service.predict("Sample political statement text.")
    pred = result["prediction"]

    assert pred["class_id"] == class_idx
    assert pred["label"] == expected_label
    assert pred["confidence"] > 0.90


# ==============================================================================
# TEST 4: Missing Text Validation
# ==============================================================================
def test_missing_text_validation(client):
    """Verifies that requests missing the 'text' key return HTTP 422."""
    response = client.post("/api/analyze", json={"metadata": {"speaker": "Politician"}})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


# ==============================================================================
# TEST 5: Empty and Whitespace Text Validation
# ==============================================================================
@pytest.mark.parametrize("empty_input", ["", "   ", "\n\t", "ab"])
def test_empty_text_validation(client, empty_input):
    """Verifies that empty, whitespace-only, or too short text returns HTTP 422."""
    response = client.post("/api/analyze", json={"text": empty_input})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


# ==============================================================================
# TEST 6: Missing Metadata Fallback
# ==============================================================================
def test_missing_metadata_formatting(mock_service):
    """
    Verifies that missing metadata explicitly defaults to 'unknown'
    and never fabricates speaker identities or political affiliations.
    """
    # 1. When metadata is None
    formatted_none = mock_service.format_metadata(None)
    assert formatted_none == "Speaker: unknown | Context: unknown"

    # 2. When metadata has speaker but missing context
    meta_speaker_only = ClaimMetadata(speaker="Jane Doe")
    formatted_speaker = mock_service.format_metadata(meta_speaker_only)
    assert formatted_speaker == "Speaker: Jane Doe | Context: unknown"

    # 3. When metadata has context but missing speaker
    meta_context_only = ClaimMetadata(context="Twitter Debate")
    formatted_context = mock_service.format_metadata(meta_context_only)
    assert formatted_context == "Speaker: unknown | Context: Twitter Debate"

    # 4. When metadata is completely empty dict
    formatted_empty_dict = mock_service.format_metadata({})
    assert formatted_empty_dict == "Speaker: unknown | Context: unknown"


# ==============================================================================
# TEST 7: API Response Structure (/api/analyze)
# ==============================================================================
def test_api_response_structure(client):
    """
    Verifies that /api/analyze conforms to the exact schema specification:
    {
      "prediction": {
        "label": "...",
        "class_id": 4,
        "confidence": 0.42,
        "class_scores": { ... }
      },
      "model": "RoBERTa + Metadata",
      "disclaimer": "..."
    }
    """
    from app.services.classifier import classifier_service
    import torch

    # Mock classifier_service so the endpoint can return deterministically without network download
    simulated_logits = torch.tensor([[-1.0, -0.5, 0.2, 0.8, 2.5, 0.1]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = simulated_logits
    mock_model.return_value = mock_output

    with patch.object(classifier_service, "is_loaded", True), \
         patch.object(classifier_service, "tokenizer", mock_tokenizer), \
         patch.object(classifier_service, "model", mock_model), \
         patch.object(classifier_service, "device", torch.device("cpu")):

        payload = {
            "text": "The government announced a new economic policy.",
            "metadata": {
                "speaker": "Government Spokesperson",
                "context": "Press Briefing"
            }
        }
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "model" in data
        assert "disclaimer" in data

        pred = data["prediction"]
        assert pred["label"] in LABEL_LIST
        assert isinstance(pred["class_id"], int)
        assert 0 <= pred["class_id"] <= 5
        assert isinstance(pred["confidence"], float)
        assert isinstance(pred["class_scores"], dict)
        for label in LABEL_LIST:
            assert label in pred["class_scores"]

        assert data["model"] == MODEL_IDENTIFIER
        assert data["disclaimer"] == DEFAULT_DISCLAIMER


# ==============================================================================
# TEST 8: Health Endpoint Model Status
# ==============================================================================
def test_health_endpoint_model_status(client):
    """Verifies that /api/health reflects model load status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "model_repo" in data
    assert "device" in data


# ==============================================================================
# TEST 9: Live Hugging Face Inference Test
# ==============================================================================
@pytest.mark.live
def test_live_huggingface_inference():
    """
    Executes actual neural model inference directly against the Hugging Face repository
    'soah-13/truthlens-roberta-liar-meta' using HF_TOKEN credentials.
    """
    from app.services.classifier import classifier_service

    if not settings.HF_TOKEN:
        pytest.skip("HF_TOKEN is not configured; skipping live Hugging Face test.")

    # Load actual model from Hugging Face
    classifier_service.load_model()
    assert classifier_service.is_loaded is True

    # Real prediction call
    result = classifier_service.predict(
        text="The state added forty thousand manufacturing jobs over the previous calendar year.",
        metadata={"speaker": "Department of Labor", "context": "Annual Employment Report"}
    )

    pred = result["prediction"]
    assert pred["label"] in LABEL_LIST
    assert 0 <= pred["class_id"] <= 5
    assert pred["confidence"] > 0.0
    assert len(pred["class_scores"]) == 6
    assert result["model"] == MODEL_IDENTIFIER
    print(f"\nLive RoBERTa Prediction: {pred['label']} (Confidence: {pred['confidence']})")

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app
from app.services.classifier import classifier_service, LABEL_LIST, ModelNotAvailableError
from training.preprocess import (
    load_liar_dataset,
    get_label_distribution,
    LABEL_LIST as PREPROCESS_LABELS,
    LABEL2ID,
    ID2LABEL,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# 1. Verify health endpoint remains intact
def test_health_endpoint_preserved(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "application": "TruthLens",
        "version": "1.0.0",
    }


# 2. Test metrics endpoint when unpopulated
def test_metrics_endpoint_unavailable(client: TestClient):
    with patch("os.path.exists", return_value=False):
        response = client.get("/api/model/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert "Colab" in data["message"]
        assert data["accuracy"] is None


# 3. Test metrics endpoint when populated
def test_metrics_endpoint_available(client: TestClient, tmp_path):
    mock_metrics = {
        "model_name": "distilbert-base-uncased (Fine-Tuned)",
        "test_metrics": {
            "accuracy": 0.285,
            "macro_precision": 0.274,
            "macro_recall": 0.271,
            "macro_f1": 0.268,
            "weighted_f1": 0.275,
        },
        "evaluation_date": "2026-11-05 18:00:00 UTC",
    }
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(mock_metrics))

    with patch("app.core.config.settings.METRICS_FILE_PATH", str(metrics_file)):
        response = client.get("/api/model/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "available"
        assert data["accuracy"] == 0.285
        assert data["macro_f1"] == 0.268
        assert data["weighted_f1"] == 0.275
        assert data["model_name"] == "distilbert-base-uncased (Fine-Tuned)"


# 4. Test prediction endpoint validation for empty / short text
def test_predict_validation_empty_and_short(client: TestClient):
    # Empty string
    response = client.post("/api/predict", json={"text": ""})
    assert response.status_code == 422

    # Whitespace only (< 3 chars after strip)
    response2 = client.post("/api/predict", json={"text": "   "})
    assert response2.status_code == 422

    # Less than 3 characters
    response3 = client.post("/api/predict", json={"text": "ab"})
    assert response3.status_code == 422


# 5. Test prediction endpoint validation for excessively long text
def test_predict_validation_excessively_long(client: TestClient):
    long_text = "word " * 1200  # > 5000 chars
    response = client.post("/api/predict", json={"text": long_text})
    assert response.status_code == 422


# 6. Test missing model handling returns HTTP 503
def test_predict_model_not_installed_returns_503(client: TestClient):
    with patch.object(
        classifier_service,
        "predict",
        side_effect=ModelNotAvailableError("Model weights not found"),
    ):
        response = client.post(
            "/api/predict",
            json={"text": "The government released new economic job statistics."},
        )
        assert response.status_code == 503
        data = response.json()
        assert "not available" in data["detail"].lower()


# 7. Test prediction output structure using mocked model inference
def test_predict_output_structure_mocked(client: TestClient):
    mock_prediction_result = {
        "prediction": "half-true",
        "confidence": 0.62,
        "probabilities": {
            "pants-fire": 0.03,
            "false": 0.08,
            "barely-true": 0.12,
            "half-true": 0.62,
            "mostly-true": 0.10,
            "true": 0.05,
        },
        "model": "distilbert-liar",
        "disclaimer": "This is a model prediction, not a verified fact-check.",
    }

    with patch.object(classifier_service, "predict", return_value=mock_prediction_result):
        response = client.post(
            "/api/predict",
            json={"text": "The mayor claims traffic accidents declined by half."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "half-true"
        assert data["confidence"] == 0.62
        assert len(data["probabilities"]) == 6
        for label in LABEL_LIST:
            assert label in data["probabilities"]
        assert data["disclaimer"] == "This is a model prediction, not a verified fact-check."


# 8. Test dataset loading and 6-class preservation
def test_dataset_loading_and_labels():
    dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "liar")
    train, val, test = load_liar_dataset(dataset_dir)
    assert len(train) == 10240
    assert len(val) == 1284
    assert len(test) == 1267

    dist = get_label_distribution(train)
    assert len(dist) == 6
    for label in PREPROCESS_LABELS:
        assert label in dist
        assert dist[label]["count"] > 0
        assert dist[label]["id"] == LABEL2ID[label]

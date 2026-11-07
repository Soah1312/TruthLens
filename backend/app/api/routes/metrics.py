import os
import json
import logging
from fastapi import APIRouter, status
from app.schemas.metrics import ModelMetricsResponse
from app.core.config import settings

logger = logging.getLogger("truthlens.metrics")
router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Evaluation Metrics",
    description="Returns the empirical benchmark evaluation results produced from Google Colab training.",
)
async def get_model_metrics() -> ModelMetricsResponse:
    """
    Reads and returns the evaluated metrics from Colab execution.
    If metrics are not yet generated, returns an explicit 'unavailable' status.
    """
    metrics_path = settings.METRICS_FILE_PATH

    # Check relative to workspace if not absolute
    if not os.path.isabs(metrics_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidate = os.path.join(base_dir, metrics_path)
        if os.path.exists(candidate):
            metrics_path = candidate

    if not os.path.exists(metrics_path):
        return ModelMetricsResponse(
            status="unavailable",
            message=(
                "Model evaluation metrics are not yet available. "
                "Execute the training pipeline on Google Colab, export 'metrics.json', "
                "and place it in 'models/distilbert-liar/metrics.json'."
            ),
            model_name="distilbert-base-uncased",
            accuracy=None,
            macro_precision=None,
            macro_recall=None,
            macro_f1=None,
            weighted_f1=None,
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_m = (
            data.get("test_evaluation_metrics")
            or data.get("test_metrics_6class")
            or data.get("test_metrics", {})
        )
        model_name = (
            data.get("architecture")
            or data.get("model_checkpoint")
            or data.get("model_name", "DeBERTa-v3 + Metadata")
        )
        return ModelMetricsResponse(
            status="available",
            message="Evaluation metrics loaded successfully from training artifacts.",
            model_name=model_name,
            accuracy=test_m.get("accuracy"),
            macro_precision=test_m.get("macro_precision"),
            macro_recall=test_m.get("macro_recall"),
            macro_f1=test_m.get("macro_f1"),
            weighted_f1=test_m.get("weighted_f1"),
            evaluation_date=data.get("export_timestamp") or data.get("evaluation_date"),
            baseline_metrics=data.get("baseline_roberta_comparison") or data.get("baseline_metrics"),
            per_class_metrics=test_m.get("per_class_f1") or data.get("per_class_metrics"),
        )
    except Exception as exc:
        logger.error("Failed to parse metrics file at %s: %s", metrics_path, exc)
        return ModelMetricsResponse(
            status="unavailable",
            message=f"Error reading metrics artifact: {str(exc)}",
            model_name="distilbert-base-uncased",
            accuracy=None,
            macro_precision=None,
            macro_recall=None,
            macro_f1=None,
            weighted_f1=None,
        )

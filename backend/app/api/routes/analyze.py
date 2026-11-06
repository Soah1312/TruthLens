"""
TruthLens API Route: /api/analyze
Executes neural 6-class credibility classification using the fine-tuned RoBERTa + Metadata model.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.classifier import (
    classifier_service,
    ModelNotAvailableError,
    ModelInferenceError,
)

logger = logging.getLogger("truthlens.routes.analyze")
router = APIRouter(prefix="/analyze", tags=["Inference"])


@router.post(
    "",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze News Claim or Statement",
    description=(
        "Evaluates a text statement using fine-tuned RoBERTa with sentence-pair metadata inputs (Speaker, Context). "
        "Returns 6-class truthfulness distribution, predicted category, confidence score, and academic disclaimer."
    ),
    responses={
        200: {
            "description": "Successful classification.",
            "content": {
                "application/json": {
                    "example": {
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
                        "model": "RoBERTa + Metadata",
                        "disclaimer": "This is a model prediction, not independent factual verification.",
                    }
                }
            },
        },
        422: {"description": "Validation error (e.g. text empty or out of range)."},
        503: {
            "description": "Model is not loaded or unreachable.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "RoBERTa model is currently unavailable for inference."
                    }
                }
            },
        },
        500: {
            "description": "Internal server or inference failure.",
            "content": {
                "application/json": {
                    "example": {"detail": "Neural inference failed."}
                }
            },
        },
    },
)
async def analyze_claim(request: AnalyzeRequest) -> AnalyzeResponse:
    """Classify a statement's truthfulness using RoBERTa + Metadata."""
    cleaned_text = request.text.strip() if request.text else ""
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Statement text cannot be empty or solely whitespace.",
        )

    if len(cleaned_text) < settings.MIN_INPUT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Statement text must be at least {settings.MIN_INPUT_LENGTH} characters.",
        )

    if len(cleaned_text) > settings.MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Statement text exceeds maximum allowed length of {settings.MAX_INPUT_LENGTH} characters.",
        )

    try:
        result = classifier_service.predict(
            text=cleaned_text,
            metadata=request.metadata,
        )
        return AnalyzeResponse(**result)
    except ModelNotAvailableError as exc:
        logger.error("Model unavailable for inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"RoBERTa model is currently unavailable: {str(exc)}. "
                "Please verify Hugging Face credentials (HF_TOKEN) or local checkpoint files."
            ),
        )
    except ModelInferenceError as exc:
        logger.error("Inference execution error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural inference error: {str(exc)}",
        )
    except Exception as exc:
        logger.error("Unexpected error during analysis: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(exc)}",
        )

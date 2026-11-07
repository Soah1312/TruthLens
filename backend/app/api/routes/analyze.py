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
from app.services.evidence import evidence_service
from app.services.evidence_analyzer import evidence_analyzer_service

logger = logging.getLogger("truthlens.routes.analyze")
router = APIRouter(prefix="/analyze", tags=["Inference"])


@router.post(
    "",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze News Claim or Statement",
    description=(
        "Evaluates a text statement using fine-tuned RoBERTa with sentence-pair metadata inputs (Speaker, Context). "
        "Optionally retrieves relevant external web evidence via Tavily Search API when include_evidence is enabled. "
        "Returns 6-class truthfulness distribution, predicted category, confidence score, evidence context, and academic disclaimer."
    ),
    responses={
        200: {
            "description": "Successful classification with optional external evidence.",
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
                        "evidence": {
                            "status": "success",
                            "query": "government new economic policy",
                            "results": [
                                {
                                    "title": "Government Announces New Fiscal Plan",
                                    "url": "https://example.com/article",
                                    "domain": "example.com",
                                    "snippet": "The government released a new policy statement today.",
                                    "relevance_score": 0.91,
                                    "published_date": None,
                                    "classification": "unclassified",
                                }
                            ],
                            "summary": "External search results retrieved. Their factual relevance has not been independently verified.",
                        },
                        "disclaimer": "The model prediction is not independent factual verification. Retrieved search results should be evaluated against their original sources.",
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
    """Classify a statement's truthfulness using RoBERTa + Metadata, with optional Tavily web evidence."""
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

    # 1. Neural RoBERTa Classification (Always executed)
    try:
        prediction_result = classifier_service.predict(
            text=cleaned_text,
            metadata=request.metadata,
        )
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

    # 2. External Web Evidence Retrieval via Tavily & LLM Evidence Analysis via Groq
    evidence_payload = None
    if request.include_evidence:
        try:
            evidence_payload = await evidence_service.search_evidence(cleaned_text)
        except Exception as exc:
            logger.warning("Evidence retrieval failed gracefully: %s", exc)
            evidence_payload = {
                "status": "unavailable",
                "query": cleaned_text,
                "results": [],
                "summary": "External evidence retrieval encountered a temporary error.",
                "error": str(exc),
            }

        # 3. Grounded Evidence Relationship Analysis via Groq LLM
        if evidence_payload and evidence_payload.get("status") == "success" and evidence_payload.get("results"):
            try:
                analysis = await evidence_analyzer_service.analyze_evidence(
                    claim=cleaned_text,
                    search_results=evidence_payload["results"],
                )
                evidence_payload["overall_assessment"] = analysis.get("overall_assessment")
                evidence_payload["analysis_status"] = analysis.get("status", "completed")
                evidence_payload["analysis_summary"] = analysis.get("summary")
                evidence_payload["limitations"] = analysis.get("limitations", [])
                evidence_payload["source_assessments"] = analysis.get("source_assessments", [])
            except Exception as exc:
                logger.warning("Groq evidence analysis failed gracefully: %s", exc)
                evidence_payload["overall_assessment"] = None
                evidence_payload["analysis_status"] = "failed"
                evidence_payload["analysis_summary"] = "External evidence relationship analysis encountered a temporary failure."
                evidence_payload["limitations"] = ["LLM evidence reasoning pass could not be completed."]
                evidence_payload["source_assessments"] = []
        elif evidence_payload and evidence_payload.get("status") == "no_results":
            evidence_payload["overall_assessment"] = "insufficient_evidence"
            evidence_payload["analysis_status"] = "skipped"
            evidence_payload["analysis_summary"] = "No external web sources were found to substantiate or refute the claim."
            evidence_payload["limitations"] = ["Zero external articles returned by search engine."]
            evidence_payload["source_assessments"] = []
        elif evidence_payload:
            evidence_payload["overall_assessment"] = None
            evidence_payload["analysis_status"] = "unavailable"
            evidence_payload["analysis_summary"] = None
            evidence_payload["limitations"] = []
            evidence_payload["source_assessments"] = []

    return AnalyzeResponse(
        prediction=prediction_result["prediction"],
        model=prediction_result.get("model", "DeBERTa-v3 + Metadata"),
        evidence=evidence_payload,
        disclaimer=prediction_result.get(
            "disclaimer",
            "The model prediction is not independent factual verification. "
            "Retrieved search results should be evaluated against their original sources.",
        ),
    )

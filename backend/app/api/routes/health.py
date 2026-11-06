from fastapi import APIRouter, status
from app.core.config import settings
from app.schemas.health import HealthResponse
from app.services.classifier import classifier_service

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Health Check",
    description="Returns the operating status, application identifier, current version, and neural model loading state.",
)
async def get_health() -> HealthResponse:
    """Return application health status and version metadata."""
    return HealthResponse(
        status="healthy",
        application=settings.PROJECT_NAME,
        version=settings.VERSION,
        model_loaded=classifier_service.is_loaded,
        model_repo=getattr(classifier_service, "model_repo", settings.HF_MODEL_REPO),
        device=str(classifier_service.device) if classifier_service.device else None,
        model_error=classifier_service.load_error,
    )

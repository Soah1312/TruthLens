from fastapi import APIRouter, status
from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Health Check",
    description="Returns the operating status, application identifier, and current version.",
)
async def get_health() -> HealthResponse:
    """Return application health status and version metadata."""
    return HealthResponse(
        status="healthy",
        application=settings.PROJECT_NAME,
        version=settings.VERSION,
    )

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router
from app.api.routes.metrics import router as metrics_router
from app.core.config import settings
from app.database.database import engine

# Configure logging for academic and production traceability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("truthlens.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup checks (database connectivity) and graceful shutdown procedures.
    """
    logger.info("Starting up %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    try:
        # Verify database connection without creating unnecessary tables in Phase 1/2
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("SQLite database connection established successfully at: %s", settings.DATABASE_URL)
    except Exception as exc:
        logger.error("Database connection failed during startup: %s", exc)
        raise exc

    yield

    logger.info("Shutting down %s...", settings.PROJECT_NAME)


# Initialize FastAPI application instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "TruthLens API — An NLP-Based Fake News Detection and Robustness Evaluation System. "
        "Provides backend services for news verification, linguistic analysis, and robustness benchmarking."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS) for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions with structured response format."""
    logger.warning("HTTPException [%s] on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic request validation errors."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "status_code": 422,
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected server errors without leaking raw stack traces to clients."""
    logger.error("Unhandled exception on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "detail": "An unexpected internal server error occurred.",
        },
    )


# Root Endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="Root Service Status",
    description="Provides basic service discovery and links to documentation and health endpoints.",
)
async def root() -> Dict[str, Any]:
    """Root endpoint welcoming clients and providing service endpoints."""
    return {
        "application": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "health": "/api/health",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "model_metrics": "/api/model/metrics",
        },
    }


# Include Routers
app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.ENVIRONMENT == "development"),
    )

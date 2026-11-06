from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema representing the application health status and model metadata."""

    status: str = Field(
        ...,
        description="Health check status",
        json_schema_extra={"example": "healthy"},
    )
    application: str = Field(
        ...,
        description="Application name",
        json_schema_extra={"example": "TruthLens"},
    )
    version: str = Field(
        ...,
        description="Semantic version string",
        json_schema_extra={"example": "1.0.0"},
    )
    model_loaded: bool = Field(
        False,
        description="Indicates whether the neural NLP model is loaded and ready for inference",
        json_schema_extra={"example": True},
    )
    model_repo: Optional[str] = Field(
        None,
        description="Configured Hugging Face or local model repository",
        json_schema_extra={"example": "soah-13/truthlens-roberta-liar-meta"},
    )
    device: Optional[str] = Field(
        None,
        description="Hardware device used for inference (cpu or cuda)",
        json_schema_extra={"example": "cpu"},
    )
    model_error: Optional[str] = Field(
        None,
        description="Details of any model loading error if applicable",
    )

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema representing the application health status and metadata."""

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

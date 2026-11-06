"""
TruthLens Schemas for /api/analyze endpoint.
Preserves canonical 6-class LIAR mapping and academic integrity.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ClaimMetadata(BaseModel):
    """
    Optional metadata accompanying the statement.
    Reflects the exact fields used during RoBERTa model training.
    """
    speaker: Optional[str] = Field(
        None,
        description="The speaker or source of the statement",
        json_schema_extra={"example": "Barack Obama"},
    )
    context: Optional[str] = Field(
        None,
        description="The venue, medium, or context of the statement",
        json_schema_extra={"example": "Press conference speech"},
    )
    subject: Optional[str] = Field(
        None,
        description="The subject or topic area of the claim",
        json_schema_extra={"example": "economy"},
    )
    state: Optional[str] = Field(
        None,
        description="State information if available",
        json_schema_extra={"example": "Washington D.C."},
    )
    party: Optional[str] = Field(
        None,
        description="Political party affiliation if explicitly known",
        json_schema_extra={"example": "democrat"},
    )
    job: Optional[str] = Field(
        None,
        description="Speaker's job or position if explicitly known",
        json_schema_extra={"example": "President"},
    )


class AnalyzeRequest(BaseModel):
    """Schema for news claim analysis request."""
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The news claim or statement text to evaluate",
        json_schema_extra={"example": "The government announced a new economic policy."},
    )
    metadata: Optional[ClaimMetadata] = Field(
        None,
        description="Optional statement metadata. When omitted, missing attributes explicitly default to 'unknown'.",
    )


class PredictionDetail(BaseModel):
    """Detailed breakdown of model classification."""
    label: str = Field(
        ...,
        description="Predicted canonical truthfulness category",
        json_schema_extra={"example": "mostly-true"},
    )
    class_id: int = Field(
        ...,
        ge=0,
        le=5,
        description="Canonical class index (0: pants-fire, 1: false, 2: barely-true, 3: half-true, 4: mostly-true, 5: true)",
        json_schema_extra={"example": 4},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model classification probability score for the top class (not factual veracity probability)",
        json_schema_extra={"example": 0.42},
    )
    class_scores: Dict[str, float] = Field(
        ...,
        description="Softmax probability distribution across all six canonical classes",
        json_schema_extra={
            "example": {
                "pants-fire": 0.05,
                "false": 0.08,
                "barely-true": 0.12,
                "half-true": 0.18,
                "mostly-true": 0.42,
                "true": 0.15,
            }
        },
    )


class AnalyzeResponse(BaseModel):
    """Schema for the /api/analyze response."""
    prediction: PredictionDetail
    model: str = Field(
        ...,
        description="Name of the model architecture used for inference",
        json_schema_extra={"example": "RoBERTa + Metadata"},
    )
    disclaimer: str = Field(
        default="This is a model prediction, not independent factual verification.",
        description="Mandatory academic disclaimer",
        json_schema_extra={"example": "This is a model prediction, not independent factual verification."},
    )

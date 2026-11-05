from typing import Dict
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Schema for prediction request containing a statement to verify."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The statement or news text claim to be classified",
        json_schema_extra={"example": "The senator claims unemployment dropped by 4 percent last quarter."},
    )


class PredictResponse(BaseModel):
    """Schema for 6-class credibility prediction and probability distribution."""

    prediction: str = Field(
        ...,
        description="Predicted truthfulness label",
        json_schema_extra={"example": "half-true"},
    )
    confidence: float = Field(
        ...,
        description="Highest class probability score (0.0 to 1.0)",
        json_schema_extra={"example": 0.62},
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Softmax probability distribution across all 6 canonical truthfulness classes",
        json_schema_extra={
            "example": {
                "pants-fire": 0.03,
                "false": 0.08,
                "barely-true": 0.12,
                "half-true": 0.62,
                "mostly-true": 0.10,
                "true": 0.05,
            }
        },
    )
    model: str = Field(
        ...,
        description="Model architecture and identifier",
        json_schema_extra={"example": "distilbert-liar"},
    )
    disclaimer: str = Field(
        ...,
        description="Mandatory academic disclaimer",
        json_schema_extra={"example": "This is a model prediction, not a verified fact-check."},
    )

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelMetricsResponse(BaseModel):
    """Schema representing model evaluation benchmark metrics."""

    status: str = Field(
        ...,
        description="'available' if Colab evaluation metrics are loaded, otherwise 'unavailable'",
        json_schema_extra={"example": "available"},
    )
    message: Optional[str] = Field(
        None,
        description="Informational message regarding metrics readiness",
        json_schema_extra={"example": "Evaluation metrics loaded successfully."},
    )
    model_name: Optional[str] = Field(
        None,
        description="Identifier of the evaluated model",
        json_schema_extra={"example": "distilbert-base-uncased (Fine-Tuned)"},
    )
    accuracy: Optional[float] = Field(
        None,
        description="Overall test set accuracy",
        json_schema_extra={"example": 0.285},
    )
    macro_precision: Optional[float] = Field(
        None,
        description="Macro-averaged precision across 6 classes",
        json_schema_extra={"example": 0.274},
    )
    macro_recall: Optional[float] = Field(
        None,
        description="Macro-averaged recall across 6 classes",
        json_schema_extra={"example": 0.271},
    )
    macro_f1: Optional[float] = Field(
        None,
        description="Macro-averaged F1 score",
        json_schema_extra={"example": 0.268},
    )
    weighted_f1: Optional[float] = Field(
        None,
        description="Weighted-averaged F1 score",
        json_schema_extra={"example": 0.275},
    )
    evaluation_date: Optional[str] = Field(
        None,
        description="Timestamp of Colab benchmark execution",
    )
    baseline_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Comparison metrics from TF-IDF + Logistic Regression",
    )
    per_class_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-class breakdown of precision, recall, and F1",
    )

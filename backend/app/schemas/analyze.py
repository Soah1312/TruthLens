"""
TruthLens Schemas for /api/analyze endpoint.
Preserves canonical 6-class LIAR mapping and academic integrity.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class ClaimMetadata(BaseModel):
    """
    Optional metadata accompanying the statement.
    Supports all six canonical metadata fields from the LIAR benchmark:
    speaker, context, subject, state, party, job.

    Note on RoBERTa Model Inference:
    During Phase 2 fine-tuning, RoBERTa was trained strictly on the sentence-pair:
        `Speaker: {speaker} | Context: {context}`
    To preserve exact training fidelity and prevent out-of-distribution shifts,
    the neural classifier formats speaker and context into the tokenized pair.
    The remaining attributes (subject, state, party, job) are validated and preserved
    for downstream context retrieval (Phase 3) and robustness perturbation (Phase 4).
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
    include_evidence: bool = Field(
        False,
        description="Whether to retrieve relevant external web evidence via Tavily Search API.",
        json_schema_extra={"example": True},
    )


class EvidenceResult(BaseModel):
    """Structured external web evidence item retrieved from search."""
    title: str = Field(
        ...,
        description="Title of the source webpage or article",
        json_schema_extra={"example": "Government Announces New Fiscal Stimulus Plan"},
    )
    url: str = Field(
        ...,
        description="Direct URL to the external source",
        json_schema_extra={"example": "https://example.com/economy-news"},
    )
    domain: str = Field(
        ...,
        description="Host domain of the source (e.g., reuters.com, bbc.com)",
        json_schema_extra={"example": "example.com"},
    )
    snippet: str = Field(
        ...,
        description="Relevant text excerpt retrieved from the webpage",
        json_schema_extra={"example": "The government announced a major new economic policy package today."},
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tavily search relevance score (indicates topical similarity, not factual correctness)",
        json_schema_extra={"example": 0.89},
    )
    published_date: Optional[str] = Field(
        None,
        description="Publication date if reported by the source metadata (not fabricated)",
        json_schema_extra={"example": "2026-10-15"},
    )
    classification: str = Field(
        default="unclassified",
        description=(
            "Evidence relationship to the claim. Presented as 'unclassified' because reliable "
            "classification into supporting/contradicting requires an additional reasoning model."
        ),
        json_schema_extra={"example": "unclassified"},
    )


class EvidencePayload(BaseModel):
    """Container for external web evidence retrieved from Tavily."""
    status: str = Field(
        ...,
        description="Retrieval status: 'success', 'unavailable', 'no_results', or 'error'",
        json_schema_extra={"example": "success"},
    )
    query: Optional[str] = Field(
        None,
        description="The normalized search query dispatched to Tavily",
        json_schema_extra={"example": "government new economic policy"},
    )
    results: List[EvidenceResult] = Field(
        default_factory=list,
        description="List of deduplicated external evidence items (max 5)",
    )
    summary: Optional[str] = Field(
        None,
        description="Neutral summary of retrieval status and academic caveats",
        json_schema_extra={
            "example": "External search results retrieved. Their factual relevance has not been independently verified."
        },
    )
    error: Optional[str] = Field(
        None,
        description="Diagnostic error message when evidence retrieval is unavailable",
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
    """Schema for the /api/analyze response combining neural prediction and optional web evidence."""
    prediction: PredictionDetail
    model: str = Field(
        ...,
        description="Name of the model architecture used for inference",
        json_schema_extra={"example": "RoBERTa + Metadata"},
    )
    evidence: Optional[EvidencePayload] = Field(
        None,
        description="External web evidence retrieved via Tavily Search API (strictly separate from model prediction)",
    )
    disclaimer: str = Field(
        default=(
            "The model prediction is not independent factual verification. "
            "Retrieved search results should be evaluated against their original sources."
        ),
        description="Mandatory academic disclaimer",
        json_schema_extra={
            "example": (
                "The model prediction is not independent factual verification. "
                "Retrieved search results should be evaluated against their original sources."
            )
        },
    )

from fastapi import APIRouter, HTTPException, status
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.classifier import classifier_service, ModelNotAvailableError
from app.core.config import settings

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post(
    "",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify News Claim Credibility",
    description=(
        "Analyzes a news claim or political statement using fine-tuned DistilBERT "
        "and returns a 6-class probability distribution."
    ),
    responses={
        503: {
            "description": "Model weights are not yet installed or loaded.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model is not yet installed or loaded. Please train the model on Google Colab first."
                    }
                }
            },
        },
        422: {
            "description": "Validation error (e.g., text too short or too long)."
        },
    },
)
async def predict_claim(request: PredictRequest) -> PredictResponse:
    """Classifies a claim into one of the 6 truthfulness classes."""
    cleaned_text = request.text.strip()
    if len(cleaned_text) < settings.MIN_INPUT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Statement text must be at least {settings.MIN_INPUT_LENGTH} characters.",
        )
    if len(cleaned_text) > settings.MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Statement text exceeds maximum allowed length of {settings.MAX_INPUT_LENGTH} characters.",
        )

    try:
        result = classifier_service.predict(cleaned_text)
        return PredictResponse(**result)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"DistilBERT model is not available for inference: {str(exc)} "
                "Execute the training notebook in Google Colab, download the model, and place it in 'models/distilbert-liar', "
                "or upload to Hugging Face and configure HF_MODEL_REPO in .env."
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected inference error occurred: {str(exc)}",
        )

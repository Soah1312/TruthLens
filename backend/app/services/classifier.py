"""
TruthLens Model Inference Service
Provides cached, singleton inference for fine-tuned RoBERTa + Metadata model.
Supports loading from Hugging Face Hub (with private repository authentication) or local checkpoints.
Preserves exact training input representation (statement + metadata sentence pair) and canonical 6-class labels.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, Union

from app.core.config import settings

logger = logging.getLogger("truthlens.classifier")

# Canonical 6-class truthfulness labels in increasing order of veracity
LABEL_LIST = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

MODEL_IDENTIFIER = "RoBERTa + Metadata"
DEFAULT_DISCLAIMER = "This is a model prediction, not independent factual verification."


class ModelNotAvailableError(Exception):
    """Raised when the trained model weights or tokenizer cannot be loaded."""
    pass


class ModelInferenceError(Exception):
    """Raised when tokenization or forward inference execution fails."""
    pass


class RoBERTaInferenceService:
    """
    Singleton inference service that initializes the RoBERTa model once at application
    startup and reuses cached weights across all incoming requests.
    """

    _instance: Optional["RoBERTaInferenceService"] = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.model_identifier = MODEL_IDENTIFIER
        self.model_repo = settings.HF_MODEL_REPO

    @classmethod
    def get_instance(cls) -> "RoBERTaInferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def format_metadata(metadata: Optional[Any] = None) -> str:
        """
        Formats statement metadata into the exact representation used during model training:
            'Speaker: {speaker} | Context: {context}'
        
        When metadata is omitted or unavailable, defaults to 'unknown' rather than
        fabricating speaker identities or political affiliations.
        """
        if metadata is None:
            return "Speaker: unknown | Context: unknown"

        if isinstance(metadata, dict):
            speaker = metadata.get("speaker") or "unknown"
            context = metadata.get("context") or "unknown"
        else:
            speaker = getattr(metadata, "speaker", None) or "unknown"
            context = getattr(metadata, "context", None) or "unknown"

        speaker_clean = str(speaker).strip() or "unknown"
        context_clean = str(context).strip() or "unknown"

        return f"Speaker: {speaker_clean} | Context: {context_clean}"

    def _resolve_model_location(self) -> Tuple[str, Optional[str]]:
        """
        Resolves the model repository or local directory path and authentication token.
        """
        if settings.MODEL_SOURCE.lower() == "huggingface":
            repo = settings.HF_MODEL_REPO.strip()
            if not repo or "your-username" in repo:
                raise ModelNotAvailableError(
                    "MODEL_SOURCE is set to 'huggingface', but HF_MODEL_REPO is not configured. "
                    "Please set HF_MODEL_REPO in your environment or .env file (e.g., 'soah-13/truthlens-roberta-liar-meta')."
                )
            token = settings.HF_TOKEN.strip() if settings.HF_TOKEN else None
            return repo, token
        else:
            local_path = settings.LOCAL_MODEL_PATH
            if not os.path.isabs(local_path):
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                candidate = os.path.join(base_dir, local_path)
                if os.path.exists(candidate):
                    local_path = candidate

            if not os.path.exists(local_path):
                raise ModelNotAvailableError(
                    f"Local model directory not found at: '{local_path}'. "
                    "Ensure trained weights are extracted or set MODEL_SOURCE='huggingface'."
                )

            return local_path, None

    def load_model(self) -> None:
        """
        Loads the pre-trained RoBERTa tokenizer and classification head into memory.
        Enables GPU acceleration automatically when CUDA is present.
        """
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_location, token = self._resolve_model_location()
            self.model_repo = model_location

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Initializing %s on %s from: %s", self.model_identifier, self.device, model_location)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_location,
                token=token,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_location,
                token=token,
            )
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.load_error = None
            logger.info(
                "%s successfully loaded and cached on device: %s (%d output classes)",
                self.model_identifier,
                self.device,
                len(LABEL_LIST),
            )
        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            logger.error("Failed to load model from %s: %s", getattr(self, "model_repo", "unknown"), exc)
            raise ModelNotAvailableError(f"Model initialization failed: {exc}") from exc

    def predict(
        self,
        text: str,
        metadata: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes RoBERTa neural sequence classification on a text claim with metadata.

        Args:
            text: Raw input claim text.
            metadata: Optional ClaimMetadata or dict with speaker/context.

        Returns:
            Structured dictionary conforming to AnalyzeResponse specification.
        """
        cleaned_text = text.strip() if text else ""
        if not cleaned_text:
            raise ValueError("Input statement text cannot be empty.")

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load_model()

        import torch

        metadata_str = self.format_metadata(metadata)

        try:
            # Tokenize statement and metadata as sentence-pair (exact training format)
            inputs = self.tokenizer(
                text=cleaned_text,
                text_pair=metadata_str,
                truncation=True,
                max_length=160,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities_tensor = torch.softmax(logits, dim=-1).squeeze(0)

            probs_list = probabilities_tensor.cpu().numpy().tolist()

            # Construct canonical 6-class score dictionary
            class_scores: Dict[str, float] = {}
            for idx, label in enumerate(LABEL_LIST):
                class_scores[label] = round(float(probs_list[idx]), 4)

            # Determine predicted class
            predicted_idx = int(torch.argmax(probabilities_tensor).item())
            predicted_label = ID2LABEL.get(predicted_idx, LABEL_LIST[predicted_idx])
            confidence = round(float(probs_list[predicted_idx]), 4)

            return {
                "prediction": {
                    "label": predicted_label,
                    "class_id": predicted_idx,
                    "confidence": confidence,
                    "class_scores": class_scores,
                },
                "model": self.model_identifier,
                "disclaimer": DEFAULT_DISCLAIMER,
                # Backwards compatibility fields for Phase 1 /predict callers
                "probabilities": class_scores,
            }
        except Exception as exc:
            logger.error("Inference execution failed on statement: %s", exc, exc_info=True)
            raise ModelInferenceError(f"Neural inference failed: {str(exc)}") from exc


# Global singleton instance
classifier_service = RoBERTaInferenceService.get_instance()

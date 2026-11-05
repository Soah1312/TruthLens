# TruthLens — Model Training & Dataset Processing

## Overview
This directory is reserved for offline training scripts, data preparation pipelines, and model evaluation routines, scheduled for implementation in **Phase 2**.

## Planned Scope (Phase 2)
- **Dataset**: LIAR benchmark dataset (6-way and collapsed binary classification: Real vs. Fake).
- **Core Architecture**: DistilBERT (`distilbert-base-uncased`) via Hugging Face `transformers` and `pytorch`.
- **Pipeline Components**:
  - `dataset_loader.py`: Download, parse, and split the LIAR dataset.
  - `preprocess.py`: Tokenization, cleaning, and input tensor preparation.
  - `train.py`: Fine-tuning loop with learning rate schedules, AdamW optimizer, and validation checkpoints.
  - `evaluate.py`: Calculation of performance metrics (Accuracy, Precision, Recall, Macro F1-score, and confusion matrix).
  - Export fine-tuned model artifacts to `/models/` for consumption by the backend inference engine.

*Note: Phase 1 establishes project structure and health monitoring. No model training components are active in Phase 1.*

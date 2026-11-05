# TruthLens — Robustness Evaluation Engine

## Overview
This directory is reserved for the adversarial perturbation and robustness assessment suite, scheduled for implementation in **Phase 4**.

## Planned Scope (Phase 4)
- **Objective**: Measure how stably the fine-tuned DistilBERT model performs under adversarial text noise and natural linguistic variations.
- **Perturbation Attacks**:
  - Typographical noise (character swap, deletion, insertion).
  - Synonym replacement (context-preserving word substitution).
  - Paraphrasing / back-translation.
  - Negation insertion and polarity stress testing.
- **Evaluation Metrics**:
  - Original Performance vs. Perturbed Performance (Accuracy/F1 drop).
  - Prediction Consistency Score (% of predictions that remain unchanged under perturbation).
  - Robustness Degradation Index.
- Output metrics will be exposed through backend endpoints and visualized in the frontend dashboard.

*Note: Phase 1 establishes project structure and health monitoring. No robustness evaluation components are active in Phase 1.*

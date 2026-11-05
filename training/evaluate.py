"""
TruthLens Phase 2 - DistilBERT Evaluation & Benchmarking Suite
Evaluates fine-tuned DistilBERT checkpoint on the unperturbed LIAR test split,
generating full classification metrics, per-class breakdowns, and confusion matrix data.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from training.preprocess import load_liar_dataset, LABEL_LIST, LABEL2ID, ID2LABEL
from training.train import LiarDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("truthlens.evaluate")


def evaluate_checkpoint(
    model_dir: str = "models/distilbert-liar",
    dataset_dir: str = "datasets/liar",
    max_length: int = 128,
    batch_size: int = 16,
    output_report_file: str = "models/distilbert-liar/evaluation_report.json",
) -> Dict[str, Any]:
    """
    Evaluates a saved DistilBERT model against the test set of the LIAR dataset.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {model_dir}. "
            "Please train the model using Google Colab first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating model using device: %s", device)

    # Load tokenizer and model
    logger.info("Loading model and tokenizer from: %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load test split
    _, _, test_samples = load_liar_dataset(dataset_dir)
    test_dataset = LiarDataset(test_samples, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    logger.info("Running inference on %d test samples...", len(test_samples))
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().tolist())

    # Compute comprehensive metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    per_class_metrics = {}
    for idx, label in enumerate(LABEL_LIST):
        per_class_metrics[label] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx]),
        }

    cm = confusion_matrix(all_targets, all_preds).tolist()
    report = classification_report(
        all_targets,
        all_preds,
        target_names=LABEL_LIST,
        output_dict=True,
        zero_division=0,
    )

    evaluation_data = {
        "model_name": "distilbert-base-uncased (Fine-Tuned)",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    # Save report
    os.makedirs(os.path.dirname(output_report_file), exist_ok=True)
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)

    logger.info("Evaluation report saved to: %s", output_report_file)
    logger.info("Benchmark Results:")
    logger.info("  Accuracy:        %.4f", evaluation_data["accuracy"])
    logger.info("  Macro Precision: %.4f", evaluation_data["macro_precision"])
    logger.info("  Macro Recall:    %.4f", evaluation_data["macro_recall"])
    logger.info("  Macro F1:        %.4f", evaluation_data["macro_f1"])
    logger.info("  Weighted F1:     %.4f", evaluation_data["weighted_f1"])

    return evaluation_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens DistilBERT Evaluation Suite")
    parser.add_argument("--model_dir", type=str, default="models/distilbert-liar")
    parser.add_argument("--dataset_dir", type=str, default="datasets/liar")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="models/distilbert-liar/evaluation_report.json")
    args = parser.parse_args()

    evaluate_checkpoint(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        output_report_file=args.output_file,
    )

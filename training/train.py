"""
TruthLens Phase 2 - DistilBERT Fine-Tuning Pipeline
Trains distilbert-base-uncased on the LIAR dataset for 6-class claim classification.
Standard hyperparameters:
  - Model: distilbert-base-uncased
  - Learning Rate: 2e-5
  - Batch Size: 16
  - Max Epochs: 3
  - Max Length: 128
  - Weight Decay: 0.01
  - Optimizer: AdamW
  - Best model checkpoint selected via validation Macro F1
"""

import os
import sys
import json
import random
import logging
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from training.preprocess import load_liar_dataset, LABEL_LIST, LABEL2ID, ID2LABEL, StatementSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("truthlens.train")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LiarDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized LIAR statements."""

    def __init__(self, samples: List[StatementSample], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample.statement,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample.label_id, dtype=torch.long),
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs evaluation over a dataset split and returns loss and macro F1 metrics."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def train(
    dataset_dir: str = "datasets/liar",
    output_dir: str = "models/distilbert-liar",
    model_name: str = "distilbert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    max_length: int = 128,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main training execution function.
    Trains DistilBERT and saves best checkpoint based on validation macro F1.
    """
    set_seed(seed)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("Using GPU: %s (Total VRAM: %.2f GB)", gpu_name, gpu_mem)
    else:
        logger.warning(
            "CUDA is NOT available. Model training is running on CPU. "
            "For production Phase 2 training, execute on Google Colab GPU as required."
        )

    # Load dataset
    logger.info("Loading LIAR dataset splits from: %s", dataset_dir)
    train_samples, valid_samples, test_samples = load_liar_dataset(dataset_dir)

    # Initialize tokenizer and model
    logger.info("Loading tokenizer & pre-trained model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Prepare DataLoaders
    train_dataset = LiarDataset(train_samples, tokenizer, max_length=max_length)
    valid_dataset = LiarDataset(valid_samples, tokenizer, max_length=max_length)
    test_dataset = LiarDataset(test_samples, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info("Starting training loop: %d epochs, %d batches/epoch, total_steps=%d", epochs, len(train_loader), total_steps)

    best_val_macro_f1 = -1.0
    training_history: List[Dict[str, Any]] = []

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 100 == 0 or step == len(train_loader):
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Loss: %.4f",
                    epoch,
                    epochs,
                    step,
                    len(train_loader),
                    epoch_loss / step,
                )

        train_loss = epoch_loss / len(train_loader)

        # Validation evaluation after every epoch
        val_metrics = evaluate_model(model, valid_loader, device)
        logger.info(
            "Validation Epoch %d | Loss: %.4f | Acc: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
            epoch,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["weighted_f1"],
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_weighted_f1": round(val_metrics["weighted_f1"], 4),
        }
        training_history.append(record)

        # Save best model based on validation macro-F1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            logger.info("--> New best validation Macro F1 (%.4f). Saving model to %s", best_val_macro_f1, output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    # Final evaluation on the unseen test set using the best saved model
    logger.info("Loading best model checkpoint for final test set benchmark...")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    best_model.to(device)

    test_metrics = evaluate_model(best_model, test_loader, device)
    logger.info(
        "Final Test Benchmark | Acc: %.4f | Macro Precision: %.4f | Macro Recall: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
        test_metrics["accuracy"],
        test_metrics["macro_precision"],
        test_metrics["macro_recall"],
        test_metrics["macro_f1"],
        test_metrics["weighted_f1"],
    )

    final_results = {
        "model_name": model_name,
        "num_classes": len(LABEL_LIST),
        "labels": LABEL_LIST,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "seed": seed,
            "optimizer": "AdamW",
        },
        "training_history": training_history,
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "macro_precision": round(test_metrics["macro_precision"], 4),
            "macro_recall": round(test_metrics["macro_recall"], 4),
            "macro_f1": round(test_metrics["macro_f1"], 4),
            "weighted_f1": round(test_metrics["weighted_f1"], 4),
        },
    }

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Evaluation metrics saved to: %s", metrics_file)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens DistilBERT Training Pipeline")
    parser.add_argument("--dataset_dir", type=str, default="datasets/liar")
    parser.add_argument("--output_dir", type=str, default="models/distilbert-liar")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

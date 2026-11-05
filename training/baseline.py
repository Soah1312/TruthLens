"""
TruthLens Phase 2 - Baseline Model Pipeline (TF-IDF + Logistic Regression)
Serves as an empirical traditional ML benchmark against DistilBERT for 6-class claim classification.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from training.preprocess import load_liar_dataset, LABEL_LIST, LABEL2ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("truthlens.baseline")


def train_and_evaluate_baseline(
    dataset_dir: str = "datasets/liar",
    max_features: int = 5000,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any, Any]:
    """
    Trains a TF-IDF + Logistic Regression baseline classifier on the original LIAR training split
    and benchmarks it on the unperturbed test split.
    
    Returns:
        Tuple of (metrics_dict, trained_model, fitted_vectorizer)
    """
    logger.info("Loading LIAR dataset splits for baseline evaluation...")
    train_samples, _, test_samples = load_liar_dataset(dataset_dir)

    train_texts = [s.statement for s in train_samples]
    train_labels = [s.label_id for s in train_samples]

    test_texts = [s.statement for s in test_samples]
    test_labels = [s.label_id for s in test_samples]

    logger.info("Fitting TF-IDF vectorizer (max_features=%d, ngram_range=(1,2))...", max_features)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    logger.info("Training multi-class Logistic Regression (random_state=%d, max_iter=1000)...", random_state)
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, train_labels)

    logger.info("Evaluating baseline on test split (%d samples)...", len(test_labels))
    predictions = clf.predict(X_test)

    # Compute aggregate metrics
    acc = accuracy_score(test_labels, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="weighted", zero_division=0
    )

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        test_labels, predictions, average=None, zero_division=0
    )

    per_class_metrics = {}
    for idx, label in enumerate(LABEL_LIST):
        per_class_metrics[label] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx]),
        }

    report = classification_report(
        test_labels,
        predictions,
        target_names=LABEL_LIST,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, predictions).tolist()

    results = {
        "model_name": "TF-IDF + Logistic Regression (Baseline)",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    logger.info("Baseline Evaluation Summary:")
    logger.info("  Accuracy:    %.4f", results["accuracy"])
    logger.info("  Macro F1:    %.4f", results["macro_f1"])
    logger.info("  Weighted F1: %.4f", results["weighted_f1"])

    return results, clf, vectorizer


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(project_root, "datasets", "liar")
    results, _, _ = train_and_evaluate_baseline(dataset_path)

    out_file = os.path.join(project_root, "training", "baseline_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline results saved to {out_file}")

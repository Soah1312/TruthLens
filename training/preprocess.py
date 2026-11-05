"""
TruthLens Phase 2 - Data Preprocessing & Loading Pipeline for LIAR Dataset
Preserves all six fine-grained truthfulness classes and standard train/valid/test splits.
No traditional NLP destructive cleaning (stemming/lemmatization/stopword removal) is applied,
preserving natural sentence semantics for transformer contextual embeddings.
"""

import os
import csv
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("truthlens.preprocess")

# Canonical 6-class truthfulness labels in order of increasing veracity
LABEL_LIST: List[str] = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABEL_LIST)}

# Standard 14-column schema for LIAR TSV files (Wang, ACL 2017)
COLUMN_NAMES: List[str] = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


@dataclass
class StatementSample:
    """Represents a single statement data point from LIAR."""
    sample_id: str
    label_str: str
    label_id: int
    statement: str
    speaker: str = ""
    context: str = ""


def load_tsv_split(file_path: str) -> List[StatementSample]:
    """
    Loads and validates a single LIAR TSV file split (train, valid, or test).
    
    Args:
        file_path: Path to the TSV file.
        
    Returns:
        List of StatementSample instances with valid statements and mapped label IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LIAR dataset split not found at: {file_path}")

    samples: List[StatementSample] = []
    skipped_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if len(row) < 3:
                skipped_count += 1
                continue

            sample_id = row[0].strip()
            raw_label = row[1].strip()
            statement = row[2].strip()

            # Speaker and context if available
            speaker = row[4].strip() if len(row) > 4 else ""
            context = row[13].strip() if len(row) > 13 else ""

            # Standardize label mapping
            normalized_label = raw_label.lower()
            if normalized_label not in LABEL2ID:
                skipped_count += 1
                continue

            if not statement:
                skipped_count += 1
                continue

            samples.append(
                StatementSample(
                    sample_id=sample_id,
                    label_str=normalized_label,
                    label_id=LABEL2ID[normalized_label],
                    statement=statement,
                    speaker=speaker,
                    context=context,
                )
            )

    logger.info(
        "Loaded %d valid samples from %s (skipped %d malformed rows)",
        len(samples),
        os.path.basename(file_path),
        skipped_count,
    )
    return samples


def get_label_distribution(samples: List[StatementSample]) -> Dict[str, Dict[str, Any]]:
    """Calculates counts and percentages for each of the 6 truthfulness labels."""
    total = len(samples)
    counts = {label: 0 for label in LABEL_LIST}
    for s in samples:
        counts[s.label_str] += 1

    distribution = {}
    for label in LABEL_LIST:
        cnt = counts[label]
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        distribution[label] = {
            "id": LABEL2ID[label],
            "count": cnt,
            "percentage": round(pct, 2),
        }
    return distribution


def load_liar_dataset(
    dataset_dir: str = "datasets/liar",
) -> Tuple[List[StatementSample], List[StatementSample], List[StatementSample]]:
    """
    Loads all three splits of the LIAR dataset.
    
    Returns:
        Tuple of (train_samples, valid_samples, test_samples)
    """
    train_path = os.path.join(dataset_dir, "train.tsv")
    valid_path = os.path.join(dataset_dir, "valid.tsv")
    test_path = os.path.join(dataset_dir, "test.tsv")

    train_samples = load_tsv_split(train_path)
    valid_samples = load_tsv_split(valid_path)
    test_samples = load_tsv_split(test_path)

    logger.info("Total LIAR dataset: train=%d, valid=%d, test=%d", len(train_samples), len(valid_samples), len(test_samples))
    return train_samples, valid_samples, test_samples


if __name__ == "__main__":
    # Self-test when executed directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(project_root, "datasets", "liar")
    train, val, test = load_liar_dataset(dataset_path)

    print("\n--- LIAR Dataset Statistics ---")
    print(f"Train split: {len(train)} samples")
    print(f"Validation split: {len(val)} samples")
    print(f"Test split: {len(test)} samples")

    print("\n--- Training Label Distribution ---")
    dist = get_label_distribution(train)
    for label, stats in dist.items():
        print(f"  [{stats['id']}] {label:<12}: {stats['count']:>5} ({stats['percentage']}%)")

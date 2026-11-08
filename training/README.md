# TruthLens — Model Training & Experimental Evaluation

## Overview
This directory houses the dataset ingestion, baseline modeling, and training scripts for the **TruthLens** misinformation classification benchmark.

All neural network fine-tuning was executed using GPU-accelerated cloud infrastructure (**Google Colab NVIDIA T4 GPU**), preserving local compute while ensuring reproducible, hardware-accelerated convergence.

---

## 1. Benchmark Dataset: LIAR (ACL 2017)
- **Source**: William Yang Wang, *"Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection"*, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017).
- **Location**: `datasets/liar/` (`train.tsv`, `valid.tsv`, `test.tsv`).
- **Data Splits**:
  - **Train**: 10,240 statements
  - **Validation**: 1,284 statements
  - **Test**: 1,267 statements
- **6 Fine-Grained Veracity Classes**:
  0. `pants-fire`
  1. `false`
  2. `barely-true`
  3. `half-true`
  4. `mostly-true`
  5. `true`
  *(All 6 classes are strictly preserved without lossy binary collapsing).*
- **Text Representation**:
  - **Sentence-Pair Formulation**:
    $$\text{Sequence A} = \text{Statement}$$
    $$\text{Sequence B} = \text{"Speaker: } \langle\text{speaker}\rangle \text{ | Context: } \langle\text{context}\rangle\text{"}$$
- **Zero Destructive Cleaning Policy**: Traditional aggressive stemming, lemmatization, and stopword pruning are avoided to preserve syntax, sentiment cues, and contextual self-attention dynamics.

---

## 2. Experimental Progression & Benchmarking Summary

Across four structured experimental iterations, we evaluated progressively sophisticated architectures and feature representations:

| Experiment | Model Architecture | Input Features | Objective / Regularization | Test Accuracy | Test Macro-F1 |
|:---|:---|:---|:---|:---:|:---:|
| **Exp 1: Baseline** | TF-IDF + Logistic Regression | Statement unigrams/bigrams | Standard Cross-Entropy | 24.39% | 23.82% |
| **Exp 2: DistilBERT** | `distilbert-base-uncased` | Statement only | AdamW, lr 2e-5, CE | 27.52% | 27.18% |
| **Exp 3: RoBERTa** | `roberta-base` | Statement + Speaker/Context Metadata | Sentence-pair + Label Smoothing (0.1) | 28.41% | 28.39% |
| **Exp 4: DeBERTa-v3 (Latest)** | `microsoft/deberta-v3-base` | Statement + Metadata (Seq A + B) | Disentangled Attention + RTD + AdamW | **29.52%** | **29.62%** |

*Note: On the 6-class LIAR benchmark, human agreement averages 25%–30%, establishing DeBERTa-v3 as a highly competitive statistical prior.*

---

## 3. Standalone Training & Evaluation Scripts

| Script | Purpose | Execution |
|:---|:---|:---|
| [`preprocess.py`](preprocess.py) | Ingests, validates, and reports class distributions for the LIAR TSV splits. | `python training/preprocess.py` |
| [`baseline.py`](baseline.py) | Trains a TF-IDF + Logistic Regression benchmark on train split and evaluates on test. | `python training/baseline.py` |
| [`train.py`](train.py) | Standalone PyTorch fine-tuning script with AdamW, learning rate warmup, and checkpointing. | `python training/train.py --epochs 3 --lr 2e-5` |
| [`evaluate.py`](evaluate.py) | Computes accuracy, macro/weighted F1, confusion matrix, and classification report. | `python training/evaluate.py` |

---

## 4. Latest Google Colab Training Notebook

The primary, end-to-end reproducible training workflow for the latest and highest-performing model is:

**[`TruthLens_Experiment_4_DeBERTa_v3_Base.ipynb`](../TruthLens_Experiment_4_DeBERTa_v3_Base.ipynb)**

### Step-by-Step Execution Guide:
1. **Open Google Colab**: Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Upload Notebook**: Click **Upload** and select `TruthLens_Experiment_4_DeBERTa_v3_Base.ipynb` from the root directory of this repository.
3. **Select GPU Accelerator**:
   - In Colab menu: **Runtime** &rarr; **Change runtime type** &rarr; select **T4 GPU** (or any available CUDA accelerator).
4. **Execute Top to Bottom**:
   - Run **Runtime** &rarr; **Run all** (or step through Section 1 to Section 10).
   - Training 4 epochs on an NVIDIA T4 GPU takes approximately **10 to 14 minutes**.
5. **Inspect Benchmark Results**:
   - Section 6 computes test set accuracy, per-class metrics, and generates the seaborn confusion matrix.
   - Section 7 renders the cross-experiment comparative analysis.
6. **Export & Model Deployment**:
   - Section 8 exports `evaluation_results.json` into `models/deberta-v3-liar/`.
   - Section 9 provides direct, authenticated Hugging Face Hub publication to `soah-13/truthlens-deberta-v3-liar`.

---

## 5. Integrating the Trained Checkpoint with Backend

The FastAPI backend automatically pulls the latest fine-tuned checkpoint from Hugging Face Hub:

```env
MODEL_SOURCE="huggingface"
HF_MODEL_REPO="soah-13/truthlens-deberta-v3-liar"
METRICS_FILE_PATH="models/deberta-v3-liar/evaluation_results.json"
```

For offline local execution, unpack the trained model directory into `models/deberta-v3-liar` and configure `MODEL_SOURCE="local"`.

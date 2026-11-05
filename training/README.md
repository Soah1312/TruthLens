# TruthLens — Phase 2: DistilBERT Model Training & Evaluation

## Overview
Phase 2 implements the core Natural Language Processing (NLP) claim-classification baseline using `distilbert-base-uncased` fine-tuned on the benchmark LIAR dataset.

All actual neural network training takes place in the cloud using **Google Colab's GPU** (e.g., NVIDIA T4), preserving local computing resources while ensuring rapid, reproducible model convergence.

---

## 1. Dataset Specification: LIAR Benchmark
- **Source**: William Yang Wang, *"Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection"*, ACL 2017.
- **Location**: `datasets/liar/` (`train.tsv`, `valid.tsv`, `test.tsv`).
- **Splits Preserved**:
  - **Train**: 10,240 statements
  - **Validation**: 1,284 statements
  - **Test**: 1,267 statements
- **6 Fine-Grained Veracity Labels**:
  0. `pants-fire`
  1. `false`
  2. `barely-true`
  3. `half-true`
  4. `mostly-true`
  5. `true`
  *(All 6 classes are preserved without binary collapsing).*
- **Tokenization**: `DistilBertTokenizerFast` (`max_length=128`).
- **Zero Destructive Cleaning Policy**: Traditional stemming, lemmatization, and stopword pruning are explicitly avoided to retain complete contextual semantics for self-attention.

---

## 2. Reusable Training & Evaluation Scripts

| Script | Purpose | Execution |
|:---|:---|:---|
| [`training/preprocess.py`](file:///c:/Users/soahf/Desktop/NLP_NEWS/training/preprocess.py) | Ingests, validates, and reports class distributions for the LIAR TSV splits. | `python training/preprocess.py` |
| [`training/baseline.py`](file:///c:/Users/soahf/Desktop/NLP_NEWS/training/baseline.py) | Trains a TF-IDF + Logistic Regression benchmark on train split and evaluates on test. | `python training/baseline.py` |
| [`training/train.py`](file:///c:/Users/soahf/Desktop/NLP_NEWS/training/train.py) | Standalone PyTorch fine-tuning script with AdamW, learning rate warmup, and checkpointing. | `python training/train.py --epochs 3 --lr 2e-5` |
| [`training/evaluate.py`](file:///c:/Users/soahf/Desktop/NLP_NEWS/training/evaluate.py) | Computes accuracy, macro/weighted F1, confusion matrix, and classification report. | `python training/evaluate.py` |

---

## 3. Google Colab Training Workflow

The primary training interface is the complete self-contained notebook:
[`training/TruthLens_Phase2_Colab.ipynb`](file:///c:/Users/soahf/Desktop/NLP_NEWS/training/TruthLens_Phase2_Colab.ipynb)

### Step-by-Step Execution Guide:
1. **Open Google Colab**: Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Upload Notebook**: Click **Upload** and select `training/TruthLens_Phase2_Colab.ipynb` from your local project repository.
3. **Select GPU Hardware Accelerator**:
   - In Colab menu: **Runtime** &rarr; **Change runtime type** &rarr; select **T4 GPU** (or any available GPU).
4. **Execute Top to Bottom**:
   - Run **Runtime** &rarr; **Run all** (or step through Section 1 to Section 10).
   - Training 3 epochs on a T4 GPU takes approximately **4 to 6 minutes**.
5. **Inspect Benchmark Results**:
   - Section 4 outputs the TF-IDF + Logistic Regression baseline results.
   - Section 6 outputs the DistilBERT test set accuracy, per-class metrics, and confusion matrix.
   - Section 7 renders the empirical comparison table.
6. **Export & Download Trained Model**:
   - Section 8 exports the weights, tokenizer, and `metrics.json` into `truthlens-distilbert-liar.zip` and triggers automatic browser download.
7. **(Optional) Hugging Face Hub Upload**:
   - Section 9 allows automatic authenticated upload to your Hugging Face profile using your `HF_TOKEN` from Colab Secrets or a prompt.

---

## 4. Connecting Trained Model to Local Backend

After Colab training completes, you can enable local inference using either method:

### Option A: Local Model Directory
1. Unzip `truthlens-distilbert-liar.zip`.
2. Move the extracted folder contents (`config.json`, `model.safetensors` or `pytorch_model.bin`, `tokenizer.json`, `metrics.json`) into:
   ```
   models/distilbert-liar/
   ```
3. In `backend/.env`, ensure:
   ```bash
   MODEL_SOURCE="local"
   LOCAL_MODEL_PATH="models/distilbert-liar"
   METRICS_FILE_PATH="models/distilbert-liar/metrics.json"
   ```

### Option B: Hugging Face Hub
1. If you ran Section 9 in Colab, your model is published on Hugging Face (e.g. `username/truthlens-distilbert-liar`).
2. In `backend/.env`, set:
   ```bash
   MODEL_SOURCE="huggingface"
   HF_MODEL_REPO="your-username/truthlens-distilbert-liar"
   ```

---

## 5. Backend Endpoints (Phase 2)
- **POST `/api/predict`**: Accepts `{"text": "Political claim..."}` and returns the predicted label, confidence, and full 6-class probability distribution. Returns `HTTP 503` if model weights are not loaded.
- **GET `/api/model/metrics`**: Returns evaluated test accuracy, macro precision, recall, macro F1, and baseline comparison. Connected directly to the frontend Model Performance dashboard.

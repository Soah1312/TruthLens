# TruthLens: An NLP-Based Fake News Detection and Robustness Evaluation System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_18_%2B_Vite-61DAFB.svg?style=flat&logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/Language-TypeScript_5-3178C6.svg?style=flat&logo=typescript)](https://www.typescriptlang.org)
[![Python](https://img.shields.io/badge/Language-Python_3.10%2F3.11-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![Transformers](https://img.shields.io/badge/Model-DistilBERT-FF6F00.svg?style=flat&logo=huggingface)](https://huggingface.co/distilbert-base-uncased)
[![Colab GPU](https://img.shields.io/badge/Training-Google_Colab_GPU-F9AB00.svg?style=flat&logo=googlecolab)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview

**TruthLens** is an academic engineering and research platform designed to automate fake news detection and evaluate the robustness of transformer-based language models against adversarial textual noise.

The end-to-end system allows users to submit news articles (via raw text or web URLs) through an asynchronous **Telegram bot**. The system extracts clean article content, runs semantic classification through a fine-tuned **DistilBERT** transformer model, validates facts against live web evidence gathered via the **Tavily Search API**, stress-tests predictions against textual perturbations, and visualizes comprehensive model performance and robustness metrics via an interactive web dashboard.

---

## 2. Problem Statement

The rapid proliferation of digital misinformation presents a major threat to democratic discourse, public health, and societal cohesion. While automated natural language processing (NLP) models have shown promise in classifying deceptive text, existing solutions suffer from three fundamental limitations:

1. **Vulnerability to Textual Noise & Paraphrasing**: State-of-the-art neural classifiers frequently fail or flip predictions when subjected to simple typos, character swaps, synonym substitutions, or slight grammatical alterations that do not change semantic truth.
2. **Lack of Grounded Evidence**: Black-box classification scores provide zero real-world factual context, leaving users unable to verify *why* a statement is deemed false.
3. **Accessibility Barriers**: Advanced verification tools are rarely accessible where misinformation spreads fastest — messaging apps and mobile communication channels.

**TruthLens** addresses these gaps by coupling fine-tuned transformer classification with dynamic external web search evidence and formal adversarial perturbation benchmarking.

---

## 3. Project Objectives

- **Develop a High-Accuracy NLP Classifier**: Fine-tune a lightweight transformer architecture (`distilbert-base-uncased`) on the benchmark LIAR dataset for multi-class credibility scoring across 6 truthfulness levels (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`).
- **Leverage Cloud GPU Acceleration**: Execute all compute-intensive model training on Google Colab GPU (T4), producing portable model weights and evaluation artifacts.
- **Provide Real-Time Fact Grounding**: Integrate automated article extraction (Trafilatura) and live web evidence retrieval (Tavily API) in subsequent phases to accompany predictions with corroborating sources.
- **Empirically Benchmark Model Robustness**: Subject the model to targeted adversarial perturbations (typos, synonym replacement, paraphrasing, negation injection) in Phase 4 to quantify prediction consistency.
- **Deliver an Omnichannel User Experience**: Provide an accessible Telegram intake bot alongside a professional research dashboard for evaluating model performance and robustness.

---

## 4. Technology Stack

### Backend
- **Language**: Python 3.10 / 3.11
- **API Framework**: FastAPI
- **ASGI Server**: Uvicorn
- **Data Validation & Settings**: Pydantic v2 & Pydantic-Settings
- **ORM & Database**: SQLAlchemy 2.0 with SQLite (for initial development)
- **Unit & Integration Testing**: Pytest, HTTPX

### Frontend
- **Library & Build Tool**: React 18 with Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS (Tailored dark theme)
- **Icons**: Lucide React
- **Data Visualization**: Recharts

### Machine Learning & Data (Phase 2)
- **Framework**: PyTorch
- **Transformers**: Hugging Face Transformers (`distilbert-base-uncased`)
- **Evaluation**: Scikit-Learn (Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Confusion Matrix)
- **Dataset**: LIAR benchmark dataset (William Yang Wang, ACL 2017)
- **Training Environment**: Google Colab GPU (T4 accelerator)
- **Model Hosting & Distribution**: Hugging Face Hub / Local ZIP export

---

## 5. Project Directory Structure

```
truthlens/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── health.py          # GET /api/health (System health)
│   │   │       ├── predict.py         # POST /api/predict (6-class claim inference)
│   │   │       └── metrics.py         # GET /api/model/metrics (Colab benchmark metrics)
│   │   ├── core/
│   │   │   └── config.py              # Pydantic v2 environment settings
│   │   ├── database/
│   │   │   └── database.py            # SQLAlchemy SQLite connection setup
│   │   ├── services/
│   │   │   └── classifier.py          # Singleton DistilBERT inference service
│   │   ├── schemas/
│   │   │   ├── health.py              # Health response contract
│   │   │   ├── predict.py             # Prediction request/response schemas
│   │   │   └── metrics.py             # Evaluation metrics response schema
│   │   └── main.py                    # FastAPI application initialization & CORS
│   ├── tests/
│   │   ├── test_health.py             # Phase 1 health check tests
│   │   └── test_phase2.py             # Phase 2 inference, validation & metrics tests
│   ├── requirements.txt               # Backend dependencies
│   ├── .env.example                   # Backend environment template
│   └── .env                           # Local environment config (ignored)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.tsx        # Navigation sidebar with Phase status badges
│   │   │   │   └── Header.tsx         # System status header & health badge
│   │   │   ├── dashboard/
│   │   │   │   ├── SummaryCards.tsx   # 4 metric cards ("Not available yet")
│   │   │   │   ├── RecentAnalysis.tsx # Empty state prompting Telegram intake
│   │   │   │   ├── ModelPerformance.tsx # Live metrics synced from /api/model/metrics
│   │   │   │   └── RobustnessEvaluation.tsx # Robustness evaluation placeholders
│   │   │   └── common/
│   │   │       └── StatusBadge.tsx    # Live backend connection indicator
│   │   ├── pages/
│   │   │   └── DashboardPage.tsx      # Main dashboard view
│   │   ├── services/
│   │   │   └── api.ts                 # Central API service client
│   │   ├── hooks/
│   │   │   └── useHealth.ts           # Dynamic backend health & polling hook
│   │   ├── types/
│   │   │   └── index.ts               # TypeScript data definitions
│   │   ├── App.tsx                    # Root application container
│   │   ├── main.tsx                   # React DOM entry point
│   │   └── index.css                  # Tailwind styles and custom dark theme
│   ├── package.json                   # Frontend dependencies and scripts
│   ├── tailwind.config.js             # Tailwind CSS tokens
│   ├── vite.config.ts                 # Vite bundler configuration
│   └── .env.example                   # Frontend environment template
│
├── training/
│   ├── TruthLens_Phase2_Colab.ipynb   # Complete 10-section Google Colab training notebook
│   ├── preprocess.py                  # LIAR dataset loading & split validation
│   ├── baseline.py                    # TF-IDF + Logistic Regression benchmark
│   ├── train.py                       # Standalone PyTorch DistilBERT fine-tuning
│   ├── evaluate.py                    # Standalone checkpoint benchmarking suite
│   └── README.md                      # Detailed Phase 2 Colab training instructions
│
├── datasets/
│   └── liar/
│       ├── train.tsv                  # 10,240 training statements
│       ├── valid.tsv                  # 1,284 validation statements
│       └── test.tsv                   # 1,267 test statements
│
├── models/
│   └── .gitkeep                       # Destination for trained model artifacts
│
├── docs/
│   └── ARCHITECTURE.md                # System architecture documentation
│
├── .gitignore                         # Project-wide Git ignore rules
└── README.md                          # Project documentation
```

---

## 6. Training Pipeline (Google Colab)

To execute the DistilBERT training pipeline:
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload [`training/TruthLens_Phase2_Colab.ipynb`](file:///c:/Users/soahf/Desktop/NLP_NEWS/training/TruthLens_Phase2_Colab.ipynb).
3. Set runtime to **GPU** (`T4 GPU`).
4. Run all cells:
   - Evaluates TF-IDF + Logistic Regression baseline on LIAR test set.
   - Fine-tunes `distilbert-base-uncased` across 3 epochs (batch size 16, lr 2e-5, AdamW).
   - Generates confusion matrix and learning curves.
   - Exports `truthlens-distilbert-liar.zip` containing weights, tokenizer, and `metrics.json`.
   - Optionally uploads model to your Hugging Face account.

---

## 7. Local Installation & Startup Instructions

### Backend Setup
```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
- API Root: [http://localhost:8000/](http://localhost:8000/)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Model Metrics: [http://localhost:8000/api/model/metrics](http://localhost:8000/api/model/metrics)
- Claim Prediction: `POST http://localhost:8000/api/predict`

### Frontend Setup
```powershell
cd frontend
npm run dev
```
- Dashboard: [http://localhost:5173/](http://localhost:5173/)

---

## 8. Current Development Status

| Phase | Milestone | Status |
|:---|:---|:---:|
| **Phase 1** | Foundation, Health Monitoring & Dashboard Skeleton | **COMPLETED** |
| **Phase 2** | DistilBERT Colab Pipeline, LIAR Dataset & Inference API | **CODE PREPARED** (Awaiting Colab Run) |
| **Phase 3** | Telegram Bot, Trafilatura Extraction & Tavily Web Search | *Upcoming* |
| **Phase 4** | Textual Perturbations & Adversarial Robustness Suite | *Upcoming* |
| **Phase 5** | Analytics Dashboard & Cloud Deployment | *Upcoming* |

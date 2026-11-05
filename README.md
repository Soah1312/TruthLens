# TruthLens: An NLP-Based Fake News Detection and Robustness Evaluation System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_18_%2B_Vite-61DAFB.svg?style=flat&logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/Language-TypeScript_5-3178C6.svg?style=flat&logo=typescript)](https://www.typescriptlang.org)
[![Python](https://img.shields.io/badge/Language-Python_3.10%2F3.11-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind_CSS_3-38B2AC.svg?style=flat&logo=tailwind-css)](https://tailwindcss.com)
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

- **Develop a High-Accuracy NLP Classifier**: Fine-tune a lightweight transformer architecture (`distilbert-base-uncased`) on the benchmark LIAR dataset for multi-class and binary credibility scoring.
- **Provide Real-Time Fact Grounding**: Integrate automated article extraction (Trafilatura) and live web evidence retrieval (Tavily API) to accompany every prediction with corroborating sources.
- **Empirically Benchmark Model Robustness**: Subject the model to targeted adversarial perturbations (typos, synonym replacement, paraphrasing, negation injection) to quantify prediction consistency and degradation.
- **Deliver an Omnichannel User Experience**: Provide an accessible Telegram intake bot for everyday readers alongside a professional research dashboard for evaluating model performance and robustness.

---

## 4. Technology Stack

### Backend
- **Language**: Python 3.10 / 3.11
- **API Framework**: FastAPI (High-performance ASGI framework)
- **ASGI Server**: Uvicorn
- **Data Validation & Settings**: Pydantic v2 & Pydantic-Settings
- **ORM & Database**: SQLAlchemy 2.0 with SQLite (for initial development)
- **Unit & Integration Testing**: Pytest, HTTPX

### Frontend
- **Library & Build Tool**: React 18 with Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS (Tailored dark theme)
- **Icons**: Lucide React
- **Data Visualization (Future)**: Recharts

### Machine Learning & Data (Future Phases)
- **Framework**: PyTorch
- **Transformers**: Hugging Face Transformers (`distilbert-base-uncased`)
- **Evaluation**: Scikit-Learn (Precision, Recall, Macro F1, Confusion Matrix)
- **Dataset**: LIAR benchmark dataset

### External Integrations (Future Phases)
- **Telegram Bot**: `python-telegram-bot`
- **Article Extraction**: Trafilatura
- **Web Evidence Search**: Tavily API

### Target Deployment (Future Phases)
- **Backend**: Render
- **Frontend**: Vercel
- **Version Control**: GitHub

---

## 5. Project Architecture & Directory Structure

```
truthlens/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       └── health.py          # GET /api/health endpoint
│   │   ├── core/
│   │   │   └── config.py              # Pydantic v2 environment settings
│   │   ├── database/
│   │   │   └── database.py            # SQLAlchemy SQLite connection setup
│   │   ├── services/                  # Business logic (intake, inference)
│   │   ├── schemas/                   # Pydantic data transfer schemas
│   │   │   └── health.py              # Health response contract schema
│   │   └── main.py                    # FastAPI application initialization & CORS
│   ├── tests/
│   │   └── test_health.py             # Pytest test suite for health & discovery
│   ├── requirements.txt               # Phase 1 backend dependencies
│   ├── .env.example                   # Backend environment template
│   └── .env                           # Local environment config (ignored)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.tsx        # Navigation sidebar with future placeholders
│   │   │   │   └── Header.tsx         # System status header & health badge
│   │   │   ├── dashboard/
│   │   │   │   ├── SummaryCards.tsx   # 4 cards displaying "Not available yet"
│   │   │   │   ├── RecentAnalysis.tsx # Empty state with required prompt
│   │   │   │   ├── ModelPerformance.tsx # Unpopulated metric containers
│   │   │   │   └── RobustnessEvaluation.tsx # Robustness evaluation placeholders
│   │   │   └── common/
│   │   │       └── StatusBadge.tsx    # Live backend connection indicator
│   │   ├── pages/
│   │   │   └── DashboardPage.tsx      # Main dashboard view assembling sections
│   │   ├── services/
│   │   │   └── api.ts                 # Central API service client
│   │   ├── hooks/
│   │   │   └── useHealth.ts           # Dynamic backend health & polling hook
│   │   ├── types/
│   │   │   └── index.ts               # TypeScript data definitions
│   │   ├── App.tsx                    # Root application container
│   │   ├── main.tsx                   # React DOM entry point
│   │   └── index.css                  # Tailwind styles and custom dark theme
│   ├── public/
│   ├── index.html                     # HTML5 entry with Inter & JetBrains Mono
│   ├── package.json                   # Frontend dependencies and scripts
│   ├── tsconfig.json                  # TypeScript compiler settings
│   ├── tailwind.config.js             # Tailwind CSS tokens
│   ├── vite.config.ts                 # Vite bundler configuration
│   └── .env.example                   # Frontend environment template
│
├── bot/
│   └── README.md                      # Phase 3 Telegram bot documentation
│
├── training/
│   └── README.md                      # Phase 2 model training documentation
│
├── robustness/
│   └── README.md                      # Phase 4 robustness evaluation suite
│
├── datasets/
│   └── .gitkeep                       # Destination for LIAR dataset in Phase 2
│
├── models/
│   └── .gitkeep                       # Destination for fine-tuned weights in Phase 2
│
├── docs/
│   └── ARCHITECTURE.md                # Comprehensive architectural design document
│
├── .gitignore                         # Project-wide Git ignore rules
└── README.md                          # Project documentation (this file)
```

---

## 6. Local Installation & Startup Instructions

### Prerequisites
- **Python**: 3.10 or 3.11 installed
- **Node.js**: v18+ or v20+ with `npm`

---

### Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

5. **Run the backend test suite**:
   ```bash
   pytest -v tests
   ```

6. **Start the FastAPI backend**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   *Alternative command directly from backend directory:*
   ```bash
   uvicorn app.main --reload --port 8000
   ```

7. **Verify backend endpoints**:
   - Root discovery: [http://localhost:8000/](http://localhost:8000/)
   - Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
   - Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Frontend Setup

1. **Open a new terminal and navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   *(Ensure `VITE_API_BASE_URL=http://localhost:8000` is present)*

4. **Start the Vite development server**:
   ```bash
   npm run dev
   ```

5. **Access the dashboard**:
   - Open your browser at [http://localhost:5173/](http://localhost:5173/)
   - Confirm that the top navigation header displays **"Backend Connected"** with a green pulse indicator.

---

## 7. Current Development Status

| Component | Status | Notes |
|:---|:---:|:---|
| **Phase 1: Foundation & Health** | **COMPLETED** | FastAPI backend, SQLite initialization, React dashboard skeleton, live health monitoring |
| **Phase 2: DistilBERT Model** | *Upcoming* | LIAR dataset processing, PyTorch training, evaluation pipeline |
| **Phase 3: Bot & Web Evidence** | *Upcoming* | python-telegram-bot, Trafilatura extraction, Tavily API search |
| **Phase 4: Robustness Suite** | *Upcoming* | Adversarial perturbation generator and consistency evaluation |
| **Phase 5: Analytics & Deploy** | *Upcoming* | Interactive Recharts dashboard, Render backend, Vercel frontend |

---

## 8. Five Project Development Phases

### Phase 1: Foundation, Health Monitoring & Dashboard Skeleton (Current)
- Establish canonical project architecture and directory layout.
- Implement FastAPI application with `/api/health`, `/docs`, CORS, and SQLite connectivity.
- Build modern React + TypeScript + Tailwind CSS dark SaaS dashboard skeleton.
- Connect frontend health polling to backend API with graceful disconnection handling.
- Author system architecture and baseline testing specifications.

### Phase 2: NLP Model Fine-Tuning & Evaluation Pipeline
- Ingest and preprocess the LIAR dataset.
- Fine-tune `distilbert-base-uncased` using PyTorch and Hugging Face `transformers`.
- Implement evaluation pipeline for Accuracy, Precision, Recall, and Macro F1-score.
- Export trained model weights to `/models/` and build backend inference routes.

### Phase 3: Article Intake, Telegram Bot & Web Evidence Retrieval
- Build asynchronous Telegram bot via `python-telegram-bot` accepting text and article URLs.
- Integrate Trafilatura for automated web page body extraction.
- Implement Tavily Search API client for retrieving real-time fact-checking evidence.
- Store submissions, verdicts, and evidence records in SQLite database.

### Phase 4: Adversarial Perturbation & Robustness Evaluation Suite
- Implement text perturbation suite: typographical noise, synonym substitution, paraphrasing, negation insertion.
- Benchmark fine-tuned DistilBERT performance degradation under adversarial input.
- Compute Prediction Consistency Scores and invariance rates across perturbation categories.

### Phase 5: Web Dashboard Analytics, Visualizations & Deployment
- Populate frontend dashboard with live data feeds and Recharts interactive graphs.
- Deploy FastAPI backend to Render.
- Deploy React frontend to Vercel.
- Conduct end-to-end integration and demonstration testing for academic review.

---

## 9. Academic Review & Viva Notes

- **Separation of Concerns**: Intake, classification, evidence synthesis, and presentation layers operate independently to allow modular testing.
- **Reproducibility**: All configurations are externalized into environment variables; random seeds will be fixed during Phase 2 training.
- **Zero Hallucination Policy**: Phase 1 explicitly avoids fabricated model metrics or hardcoded fake news outputs. All metric cards display "Not available yet" until true model evaluation is conducted.

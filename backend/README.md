# TruthLens Backend: Architecture & API Service

TruthLens provides an NLP-based fake news claim classification and external evidence assessment backend built with **FastAPI**, **DeBERTa-v3**, **Tavily Search API**, and **Groq LLM**.

---

## 1. Tripartite Verification Architecture

TruthLens employs three independent, decoupled verification stages:

```
                      User Submitted Claim
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [1] DeBERTa-v3                  [2] Tavily Search
    (Claim Classification)         (Information Retrieval)
               │                               │
       6-Class Probability                     ▼ Up to 5 Web Passages
          Distribution                 [3] Groq LLM
               │                    (openai/gpt-oss-120b)
               │                   (Relationship Analysis)
               │                               │
               │                   Supports / Contradicts / Neutral
               │                   + Verbatim Quote Verification
               │                   + Deterministic Overall Verdict
               │                               │
               └───────────────┬───────────────┘
                               ▼
                    Unified API Response
```

1. **DeBERTa-v3 Classifier (`soah-13/truthlens-deberta-v3-liar`)**:
   - Neural 6-class truthfulness classification fine-tuned on the LIAR benchmark (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`).
   - Evaluates linguistic and stylistic statement semantics with optional speaker/context metadata.
   - Evaluated test performance: Accuracy 29.52%, Macro-F1 29.61%.

2. **Tavily Search API**:
   - Retrieves up to 5 real-time, deduplicated external news sources and passage snippets.
   - Preserves title, host domain, URL, and published dates.
   - Does not perform automated credibility scoring.

3. **Groq LLM Evidence Analyzer (`openai/gpt-oss-120b`)**:
   - Performs source-grounded relationship classification: `supports`, `contradicts`, or `neutral`.
   - **Verbatim Quote Verification**: Extracts exact textual evidence substrings from source passages. Any hallucinated or fabricated quotation is rejected.
   - **Deterministic Backend Verdict**: Computes `overall_assessment` (`supports`, `contradicts`, `mixed`, or `insufficient_evidence`) via a deterministic backend rule matrix, preventing blind trust in LLM summaries.
   - **Crucial Caveat**: Groq performs automated LLM-assisted evidence reasoning across retrieved passages; it does *not* provide independently guaranteed fact verification.

---

## 2. Environment Variables

Configure the following variables in `.env`:

```env
# Server
PROJECT_NAME="TruthLens"
VERSION="1.0.0"
ENVIRONMENT="development"
HOST="127.0.0.1"
PORT=8000

# Model Settings
MODEL_SOURCE="huggingface"
HF_MODEL_REPO="soah-13/truthlens-deberta-v3-liar"
HF_TOKEN="your_huggingface_token"

# Tavily Search API
TAVILY_API_KEY="your_tavily_api_key"
TAVILY_SEARCH_DEPTH="basic"
TAVILY_MAX_RESULTS=5

# Groq LLM Evidence Analysis
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL="openai/gpt-oss-120b"
GROQ_TIMEOUT_SECONDS=30
GROQ_MAX_RETRIES=2
GROQ_MAX_SOURCES=5

# Telegram Bot
TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
```

---

## 3. Endpoints

### `POST /api/analyze`
Evaluates a text statement and optionally retrieves and analyzes external evidence.

**Request:**
```json
{
  "text": "India gained independence from British rule on 15 August 1947.",
  "metadata": {
    "speaker": "Historian",
    "context": "Public Lecture"
  },
  "include_evidence": true
}
```

**Response:**
```json
{
  "prediction": {
    "label": "true",
    "class_id": 5,
    "confidence": 0.35,
    "class_scores": {
      "pants-fire": 0.02,
      "false": 0.05,
      "barely-true": 0.08,
      "half-true": 0.20,
      "mostly-true": 0.30,
      "true": 0.35
    }
  },
  "model": "DeBERTa-v3 + Metadata",
  "evidence": {
    "status": "success",
    "query": "India gained independence British rule 15 August 1947",
    "overall_assessment": "supports",
    "analysis_status": "completed",
    "analysis_summary": "Retrieved external news articles explicitly confirm the claim.",
    "limitations": [
      "Source passages are excerpts and may omit surrounding context."
    ],
    "results": [
      {
        "source_id": "source_1",
        "title": "Independence Day (India)",
        "url": "https://en.wikipedia.org/wiki/Independence_Day_(India)",
        "domain": "en.wikipedia.org",
        "snippet": "...nation's independence from the United Kingdom on 15 August 1947...",
        "relevance_score": 0.95,
        "classification": "supports",
        "relevance": "high",
        "reasoning": "The passage confirms Indian independence on 15 August 1947.",
        "evidence_quote": "independence from the United Kingdom on 15 August 1947"
      }
    ]
  },
  "disclaimer": "The model prediction is not independent factual verification. Retrieved search results should be evaluated against their original sources."
}
```

---

## 4. Running the Backend & Tests

```bash
# Activate virtual environment
.\backend\venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run all backend tests (57 tests)
pytest backend/tests/ -v

# Run Groq evidence analyzer tests specifically
pytest backend/tests/test_evidence_analyzer.py -v

# Start FastAPI server
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

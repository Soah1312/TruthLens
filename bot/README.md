# TruthLens — Telegram Bot Service

## Overview
The TruthLens Telegram bot allows users to submit statements and claims directly through Telegram, executing fine-tuned RoBERTa-base credibility classification and retrieving external web evidence via Tavily.

## Architecture
- **Library**: `python-telegram-bot` (v20+, asynchronous long polling)
- **Backend Communication**: Strictly interfaces with FastAPI `POST /api/analyze` via `httpx.AsyncClient`
- **Decoupled Design**: The bot does not load neural model weights or Tavily SDK directly.

## Commands
- `/start` — Welcome message and instructions to begin.
- `/help` — Overview of claim submission, 6 truthfulness labels, and system limitations.
- `/about` — Technical architecture, LIAR benchmark provenance, and model details.

## Running the Bot
Ensure the FastAPI backend is running first on `http://127.0.0.1:8000`:

```bash
# Terminal 1: Launch FastAPI Backend
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# Terminal 2: Launch Telegram Bot
.\backend\venv\Scripts\python.exe bot/run_bot.py
# or
.\backend\venv\Scripts\python.exe -m app.telegram.bot
```

## Bot Handle
- **Bot Username**: `@TruthLensNewsBot`

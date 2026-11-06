# TruthLens — Telegram Bot Service

## Overview
The TruthLens Telegram bot allows users to submit statements and claims directly through Telegram, executing fine-tuned RoBERTa-base credibility classification with sentence-pair metadata inputs and retrieving real-time external web evidence via the Tavily Search API.

- **Bot Username**: `@TruthLensNewsBot`
- **Supported Commands**: `/start`, `/help`, `/about`
- **Claim Intake**: Plain-text news statements and political assertions (3 to 5000 characters).

---

## Operating Modes

TruthLens supports two operating modes for the Telegram bot:

### Mode 1: Local Development (Long Polling)
During local development, the bot can poll the Telegram Bot API directly without requiring a public HTTPS domain:

```powershell
# Ensure any existing webhook is removed before polling
python backend/scripts/register_webhook.py --delete

# Terminal 1: Launch FastAPI Backend
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# Terminal 2: Run Telegram Polling Daemon
.\backend\venv\Scripts\python.exe bot/run_bot.py
```

### Mode 2: Production Deployment (HTTPS Webhooks)
In production, long polling is replaced with an event-driven webhook endpoint:
- **Endpoint**: `POST /api/telegram/webhook`
- **Security**: Validates incoming updates using the `X-Telegram-Bot-Api-Secret-Token` header matching `TELEGRAM_WEBHOOK_SECRET`.
- **Handling**: Directly invokes internal FastAPI services (`analyze_claim`) without external HTTP roundtrips.

#### Registering the Webhook:
```bash
python backend/scripts/register_webhook.py --register \
  --url https://your-backend.onrender.com/api/telegram/webhook \
  --secret your-secret-token \
  --drop-pending
```

#### Checking Webhook Status:
```bash
python backend/scripts/register_webhook.py --status
```

#### Removing Webhook (Reverting to Polling):
```bash
python backend/scripts/register_webhook.py --delete
```

---

## Response Structure
Responses strictly conform to the 6-class LIAR benchmark format:
1. Submitted claim.
2. RoBERTa veracity prediction (e.g., `MOSTLY-TRUE`).
3. Model confidence percentage (distinguished from factual certainty).
4. All six class scores (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`).
5. External web evidence (up to 5 sources from Tavily with clickable URLs and relevance scores).
6. Mandatory academic disclaimer.

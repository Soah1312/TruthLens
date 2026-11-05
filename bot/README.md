# TruthLens — Telegram Bot Service

## Overview
This directory is reserved for the Telegram bot interface to TruthLens, scheduled for implementation in **Phase 3**.

## Planned Scope (Phase 3)
- **Library**: `python-telegram-bot`
- **Features**:
  - Accept raw text input or article URLs directly from users.
  - Integration with Trafilatura for extracting clean body text from web URLs.
  - Forward extracted text to the TruthLens backend classification and evidence retrieval pipeline.
  - Return clear, formatted verdict cards directly in chat, with links to the web dashboard for detailed evidence and model analysis.

*Note: Phase 1 establishes project structure and health monitoring. No Telegram bot components are active in Phase 1.*

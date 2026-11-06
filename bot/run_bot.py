"""
TruthLens Telegram Bot Runner Script
Launches the TruthLens Telegram bot using the backend app.telegram module.
"""

import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.telegram.bot import run_bot

if __name__ == "__main__":
    run_bot()

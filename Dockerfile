FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    ENVIRONMENT=production \
    MODEL_SOURCE=huggingface \
    HF_MODEL_REPO=soah-13/truthlens-roberta-liar-meta

# Install runtime utilities and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CPU-only PyTorch using --extra-index-url to resolve PyPI dependencies
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY backend /app/backend

WORKDIR /app/backend

EXPOSE 8000

# Health check using FastAPI /api/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

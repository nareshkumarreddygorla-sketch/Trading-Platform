# ================================================================
# AlphaForge Trading Platform — Development Dockerfile
# ================================================================
# Lightweight single-worker image for local development.
# For production, use deploy/docker/Dockerfile.api (referenced by
# docker-compose.yml) which includes gunicorn, tini, security
# scanning, and the full entrypoint flow.
#
# Usage:
#   docker build -t alphaforge-dev .
#   docker run -p 8000:8000 -v ./data:/app/data alphaforge-dev

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -s /bin/bash trader && \
    mkdir -p /app/models /app/data /app/logs && \
    chown -R trader:trader /app
USER trader

# Environment defaults (development)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info \
    PAPER_MODE=true \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:${PORT}/health || exit 1

EXPOSE ${PORT}

# Single-worker uvicorn for development (no gunicorn)
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--reload"]

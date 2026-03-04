# AlphaForge Trading Platform
# Multi-stage build for production deployment
FROM python:3.11-slim AS base

# System dependencies for scientific computing + TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -s /bin/bash trader && \
    chown -R trader:trader /app
USER trader

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Environment defaults
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    PAPER_MODE=true \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

EXPOSE ${PORT}

# Run via uvicorn
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

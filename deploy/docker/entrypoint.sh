#!/bin/bash
# ============================================================
# AlphaForge Trading Platform — Production Entrypoint
# Handles: migrations -> auto-training -> API startup
# ============================================================
set -euo pipefail

# ── Configuration ──
MODEL_DIR="${MODEL_DIR:-/app/models}"
MODEL_FILE="${MODEL_DIR}/alpha_xgb.joblib"
META_FILE="${MODEL_DIR}/alpha_xgb_meta.json"
TRAIN_SYMBOLS="${TRAIN_SYMBOLS:-all}"
TRAIN_INTERVAL="${TRAIN_INTERVAL:-5m}"
TRAIN_PERIOD="${TRAIN_PERIOD:-60d}"
RETRAIN_CRON="${RETRAIN_CRON:-0 2 * * 0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-30}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# ── Helpers ──
log_info()  { echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warn()  { echo "[WARN]  $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }

die() {
    log_error "$@"
    exit 1
}

echo "============================================================"
echo "  AlphaForge Trading Platform — Starting Up"
echo "============================================================"
echo "  Model dir:       ${MODEL_DIR}"
echo "  Train symbols:   ${TRAIN_SYMBOLS}"
echo "  Retrain cron:    ${RETRAIN_CRON}"
echo "  Workers:         ${WORKERS}"
echo "  Port:            ${PORT}"
echo "  Log level:       ${LOG_LEVEL}"
echo "============================================================"

# ── Step 0: Create required directories ──
mkdir -p "${MODEL_DIR}" /app/logs /app/data

# ── Step 1: Run database migrations ──
log_info "Running database migrations..."
if [ -n "${DATABASE_URL:-}" ]; then
    # Wait for the database to be ready (up to 30 seconds)
    DB_RETRIES=0
    DB_MAX_RETRIES=15
    while [ $DB_RETRIES -lt $DB_MAX_RETRIES ]; do
        if python -c "
import sys
from sqlalchemy import create_engine, text
try:
    engine = create_engine('${DATABASE_URL}', connect_args={'connect_timeout': 3})
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            log_info "Database connection verified."
            break
        fi
        DB_RETRIES=$((DB_RETRIES + 1))
        log_warn "Database not ready yet (attempt ${DB_RETRIES}/${DB_MAX_RETRIES}), waiting 2s..."
        sleep 2
    done

    if [ $DB_RETRIES -ge $DB_MAX_RETRIES ]; then
        die "Database is not reachable after ${DB_MAX_RETRIES} attempts. Aborting."
    fi

    # Run Alembic migrations
    if [ -d "/app/alembic" ] && [ -f "/app/alembic.ini" ]; then
        log_info "Applying Alembic migrations (upgrade head)..."
        if PYTHONPATH=/app alembic -c /app/alembic.ini upgrade head; then
            log_info "Database migrations applied successfully."
        else
            log_error "Alembic migration failed!"
            # In production, a migration failure is fatal -- do not start with
            # a schema that may be out of date.
            die "Migration failure is fatal. Fix migrations and redeploy."
        fi
    else
        log_warn "No alembic directory found at /app/alembic -- skipping migrations."
    fi
else
    log_warn "DATABASE_URL not set -- skipping migrations (SQLite dev mode)."
fi

# ── Step 2: Auto-train if no model exists ──
if [ ! -f "${MODEL_FILE}" ]; then
    log_info "No trained model found. Training now (first-time setup)..."
    log_info "This may take 5-15 minutes for the full NIFTY 500 universe."

    if PYTHONPATH=/app TRAIN_SYMBOLS="${TRAIN_SYMBOLS}" \
        TRAIN_INTERVAL="${TRAIN_INTERVAL}" \
        TRAIN_PERIOD="${TRAIN_PERIOD}" \
        MODEL_DIR="${MODEL_DIR}" \
        python scripts/train_alpha_model.py 2>&1 | tee /app/logs/initial_training.log; then
        log_info "Model trained successfully."
    else
        log_warn "Model training failed. API will start without a pre-trained model."
        log_warn "The platform can still operate; models can be trained via the API."
    fi
else
    log_info "Existing model found at ${MODEL_FILE} -- skipping training."
    if [ -f "${META_FILE}" ]; then
        META_SUMMARY=$(python -c '
import sys, json
try:
    d = json.load(open("'"${META_FILE}"'"))
    print(f"features={d.get(\"n_features\",\"?\")}, trained={d.get(\"trained_at\",\"?\")}")
except Exception:
    print("available")
' 2>/dev/null || echo "available")
        log_info "Model metadata: ${META_SUMMARY}"
    fi
fi

# ── Step 3: Setup weekly auto-retrain cron (if cron available) ──
if command -v crontab &> /dev/null; then
    log_info "Setting up weekly auto-retrain (cron: ${RETRAIN_CRON})..."
    RETRAIN_CMD="cd /app && PYTHONPATH=/app TRAIN_SYMBOLS=${TRAIN_SYMBOLS} MODEL_DIR=${MODEL_DIR} python scripts/train_alpha_model.py >> /app/logs/retrain.log 2>&1"
    # Replace existing retrain cron entry (if any) and add the new one
    (crontab -l 2>/dev/null | grep -v "train_alpha_model" ; echo "${RETRAIN_CRON} ${RETRAIN_CMD}") | crontab - 2>/dev/null || true
    # Start cron daemon in background
    if command -v crond &> /dev/null; then
        crond 2>/dev/null || true
    elif command -v cron &> /dev/null; then
        cron 2>/dev/null || true
    fi
    log_info "Auto-retrain scheduled."
else
    log_warn "crontab not available -- skipping auto-retrain scheduling."
fi

# ── Step 4: Start the API server ──
log_info "Starting AlphaForge API server..."
echo "   URL:              http://0.0.0.0:${PORT}"
echo "   Workers:          ${WORKERS}"
echo "   Mode:             ${TRADING_MODE:-paper}"
echo "   Graceful timeout: ${GRACEFUL_TIMEOUT}s"
echo "============================================================"

# Use gunicorn with uvicorn workers for multi-process production serving.
# Falls back to single-worker uvicorn for dev or when WORKERS=1.
if command -v gunicorn &> /dev/null && [ "${WORKERS}" -gt 1 ]; then
    exec gunicorn src.api.app:app \
        -c deploy/uvicorn_config.py
else
    exec uvicorn src.api.app:app \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --workers "${WORKERS}" \
        --log-level "${LOG_LEVEL}" \
        --timeout-graceful-shutdown "${GRACEFUL_TIMEOUT}"
fi

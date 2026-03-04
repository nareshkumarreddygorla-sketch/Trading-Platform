#!/bin/bash
# ============================================================
# Trading Platform — Auto Entrypoint
# Handles: auto-training → API startup → frontend
# Zero-touch production deployment
# ============================================================
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
MODEL_FILE="${MODEL_DIR}/alpha_xgb.joblib"
META_FILE="${MODEL_DIR}/alpha_xgb_meta.json"
TRAIN_SYMBOLS="${TRAIN_SYMBOLS:-all}"
TRAIN_INTERVAL="${TRAIN_INTERVAL:-5m}"
TRAIN_PERIOD="${TRAIN_PERIOD:-60d}"
RETRAIN_CRON="${RETRAIN_CRON:-0 2 * * 0}"  # Sunday 2am IST

echo "============================================================"
echo "  Trading Platform — Starting Up"
echo "============================================================"
echo "  Model dir:     $MODEL_DIR"
echo "  Train symbols: $TRAIN_SYMBOLS"
echo "  Retrain cron:  $RETRAIN_CRON"
echo "============================================================"

mkdir -p "$MODEL_DIR"

# ── Step 1: Auto-train if no model exists ──
if [ ! -f "$MODEL_FILE" ]; then
    echo ""
    echo "📊 No trained model found. Training now (first-time setup)..."
    echo "   This may take 5-15 minutes for full NIFTY 500 universe."
    echo ""
    PYTHONPATH=/app TRAIN_SYMBOLS="$TRAIN_SYMBOLS" \
        TRAIN_INTERVAL="$TRAIN_INTERVAL" \
        TRAIN_PERIOD="$TRAIN_PERIOD" \
        MODEL_DIR="$MODEL_DIR" \
        python scripts/train_alpha_model.py
    echo ""
    echo "✅ Model trained successfully!"
else
    echo "✅ Existing model found at $MODEL_FILE — skipping training."
    # Show model age
    if [ -f "$META_FILE" ]; then
        echo "   Model metadata: $(cat $META_FILE | python -c 'import sys,json; d=json.load(sys.stdin); print(f"features={d.get(\"n_features\",\"?\")}, trained={d.get(\"trained_at\",\"?\")}")' 2>/dev/null || echo 'available')"
    fi
fi

# ── Step 2: Setup weekly auto-retrain cron ──
if command -v crontab &> /dev/null; then
    echo ""
    echo "⏰ Setting up weekly auto-retrain (cron: $RETRAIN_CRON)..."
    RETRAIN_CMD="cd /app && PYTHONPATH=/app TRAIN_SYMBOLS=$TRAIN_SYMBOLS MODEL_DIR=$MODEL_DIR python scripts/train_alpha_model.py >> /app/logs/retrain.log 2>&1"
    # Add cron job (replace existing if any)
    (crontab -l 2>/dev/null | grep -v "train_alpha_model" ; echo "$RETRAIN_CRON $RETRAIN_CMD") | crontab -
    # Start cron daemon
    if command -v crond &> /dev/null; then
        crond
    elif command -v cron &> /dev/null; then
        cron
    fi
    echo "   ✅ Auto-retrain scheduled"
fi

# ── Step 3: Create log directory ──
mkdir -p /app/logs

# ── Step 4: Start the API server ──
echo ""
echo "🚀 Starting Trading Platform API server..."
echo "   URL: http://0.0.0.0:${PORT:-8000}"
echo "   Mode: ${TRADING_MODE:-paper}"
echo "============================================================"

exec uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}"

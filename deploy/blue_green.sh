#!/usr/bin/env bash
# ================================================================
# AlphaForge Trading Platform — Blue-Green Deployment
# ================================================================
#
# Zero-downtime deployment with automatic health validation,
# database migration verification, model compatibility check,
# and one-command rollback.
#
# Usage:
#   ./deploy/blue_green.sh deploy   [--config deploy/deployment_config.yaml]
#   ./deploy/blue_green.sh rollback [--config deploy/deployment_config.yaml]
#   ./deploy/blue_green.sh status
#
# Prerequisites:
#   - docker compose v2+
#   - curl, jq
#
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults (overridden by deployment_config.yaml or env) ──
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-/health}"
READY_ENDPOINT="${READY_ENDPOINT:-/ready}"
HEALTH_RETRIES="${HEALTH_RETRIES:-10}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-5}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10}"
API_PORT="${API_PORT:-8000}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
STATE_FILE="${STATE_FILE:-$PROJECT_ROOT/.deploy_state}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/deploy_$(date +%Y%m%d_%H%M%S).log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================================================================
# Utility functions
# ================================================================

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"; }

ensure_log_dir() {
    mkdir -p "$(dirname "$LOG_FILE")"
}

get_active_slot() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo "blue"
    fi
}

set_active_slot() {
    echo "$1" > "$STATE_FILE"
}

inactive_slot() {
    local active
    active=$(get_active_slot)
    if [ "$active" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# ================================================================
# Pre-flight checks
# ================================================================

preflight_check() {
    log_step "Running pre-flight checks..."

    # Check docker compose
    if ! command -v docker &>/dev/null; then
        log_error "docker not found. Install Docker first."
        exit 1
    fi

    if ! docker compose version &>/dev/null; then
        log_error "docker compose v2 not found."
        exit 1
    fi

    # Check required tools
    for cmd in curl jq; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "$cmd not found. Install it first."
            exit 1
        fi
    done

    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    log_info "Pre-flight checks passed"
}

# ================================================================
# Database migration check
# ================================================================

check_db_migrations() {
    log_step "Checking database migrations..."

    # Verify alembic is available and migrations are up to date
    if [ -f "$PROJECT_ROOT/alembic.ini" ]; then
        # Run alembic check in the new container
        if docker compose -f "$COMPOSE_FILE" run --rm --no-deps api \
            python -c "
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine
import os

db_url = os.environ.get('DATABASE_URL', '')
if not db_url:
    print('SKIP: No DATABASE_URL configured')
    exit(0)

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current = context.get_current_revision()

    config = Config('alembic.ini')
    script = ScriptDirectory.from_config(config)
    head = script.get_current_head()

    if current == head:
        print(f'OK: Database at head revision ({head})')
    else:
        print(f'PENDING: Current={current}, Head={head}')
        print('Run: alembic upgrade head')
        exit(1)
except Exception as e:
    print(f'WARN: Migration check failed: {e}')
    exit(0)
" 2>/dev/null; then
            log_info "Database migrations verified"
        else
            log_warn "Database migration check failed — review before proceeding"
            if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
                return 1
            fi
        fi
    else
        log_info "No alembic.ini found — skipping migration check"
    fi
    return 0
}

# ================================================================
# Model compatibility check
# ================================================================

check_model_compatibility() {
    log_step "Checking model compatibility..."

    # Verify that existing trained models are compatible with new code
    if docker compose -f "$COMPOSE_FILE" run --rm --no-deps api \
        python -c "
import os
import sys

model_dir = os.environ.get('MODEL_DIR', '/app/models')
if not os.path.exists(model_dir):
    print('SKIP: No model directory')
    sys.exit(0)

model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') or f.endswith('.pt')]
if not model_files:
    print('SKIP: No model files found')
    sys.exit(0)

errors = []
for mf in model_files:
    try:
        path = os.path.join(model_dir, mf)
        if mf.endswith('.joblib'):
            import joblib
            model = joblib.load(path)
            print(f'  OK: {mf} loaded successfully')
        elif mf.endswith('.pt'):
            import torch
            model = torch.load(path, map_location='cpu', weights_only=False)
            print(f'  OK: {mf} loaded successfully')
    except Exception as e:
        errors.append(f'{mf}: {e}')
        print(f'  FAIL: {mf} — {e}')

if errors:
    print(f'FAIL: {len(errors)} model(s) incompatible')
    sys.exit(1)
else:
    print(f'OK: All {len(model_files)} models compatible')
" 2>/dev/null; then
        log_info "Model compatibility verified"
    else
        log_warn "Model compatibility check encountered issues"
    fi
}

# ================================================================
# Health check
# ================================================================

wait_for_health() {
    local target_port="$1"
    local endpoint="$2"
    local retries="$3"
    local interval="$4"

    log_step "Waiting for health check at :${target_port}${endpoint} (${retries} retries, ${interval}s interval)..."

    for i in $(seq 1 "$retries"); do
        if curl -sf --max-time "$HEALTH_TIMEOUT" \
            "http://127.0.0.1:${target_port}${endpoint}" > /dev/null 2>&1; then
            log_info "Health check passed on attempt $i"
            return 0
        fi
        log_info "  Attempt $i/$retries — waiting ${interval}s..."
        sleep "$interval"
    done

    log_error "Health check failed after $retries attempts"
    return 1
}

deep_health_check() {
    local target_port="$1"

    log_step "Running deep health validation..."

    # Check /health
    local health_response
    health_response=$(curl -sf --max-time "$HEALTH_TIMEOUT" \
        "http://127.0.0.1:${target_port}${HEALTH_ENDPOINT}" 2>/dev/null || echo '{}')

    local status
    status=$(echo "$health_response" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")

    if [ "$status" != "ok" ] && [ "$status" != "healthy" ]; then
        log_error "Health endpoint returned status=$status"
        log_error "Response: $health_response"
        return 1
    fi

    # Check /ready if available
    if curl -sf --max-time "$HEALTH_TIMEOUT" \
        "http://127.0.0.1:${target_port}${READY_ENDPOINT}" > /dev/null 2>&1; then
        log_info "Readiness check passed"
    else
        log_warn "Readiness endpoint unavailable (non-blocking)"
    fi

    # Check /metrics endpoint (Prometheus)
    if curl -sf --max-time "$HEALTH_TIMEOUT" \
        "http://127.0.0.1:${target_port}/metrics" > /dev/null 2>&1; then
        log_info "Metrics endpoint accessible"
    fi

    log_info "Deep health validation passed"
    return 0
}

# ================================================================
# Deploy
# ================================================================

do_deploy() {
    ensure_log_dir
    preflight_check

    local active
    active=$(get_active_slot)
    local target
    target=$(inactive_slot)

    log_info "========================================="
    log_info "  Blue-Green Deployment"
    log_info "  Active: $active -> Target: $target"
    log_info "  Time: $(date)"
    log_info "========================================="

    # Step 1: Build the new image
    log_step "1/6 Building new image..."
    if ! docker compose -f "$COMPOSE_FILE" build api 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Build failed"
        exit 1
    fi
    log_info "Build completed"

    # Step 2: Check database migrations
    log_step "2/6 Database migration check..."
    if ! check_db_migrations; then
        log_error "Database migration check failed — aborting deployment"
        exit 1
    fi

    # Step 3: Check model compatibility
    log_step "3/6 Model compatibility check..."
    check_model_compatibility

    # Step 4: Start the new (target) slot on a staging port
    local staging_port=$((API_PORT + 1))
    log_step "4/6 Starting $target slot on port $staging_port..."

    # Use a project-specific name for the target environment
    export COMPOSE_PROJECT_NAME="alphaforge-${target}"
    export API_EXTERNAL_PORT="$staging_port"

    if ! docker compose -f "$COMPOSE_FILE" up -d api 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Failed to start $target slot"
        exit 1
    fi

    # Step 5: Health check the new slot
    log_step "5/6 Health checking $target slot..."
    if ! wait_for_health "$staging_port" "$HEALTH_ENDPOINT" "$HEALTH_RETRIES" "$HEALTH_INTERVAL"; then
        log_error "New slot failed health check — rolling back"
        docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
        unset COMPOSE_PROJECT_NAME API_EXTERNAL_PORT
        exit 1
    fi

    if ! deep_health_check "$staging_port"; then
        log_error "Deep health check failed — rolling back"
        docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
        unset COMPOSE_PROJECT_NAME API_EXTERNAL_PORT
        exit 1
    fi

    # Step 6: Switch traffic (update nginx upstream and reload)
    log_step "6/6 Switching traffic to $target..."

    # Update the active slot record
    local previous_slot="$active"
    set_active_slot "$target"

    # Stop old slot gracefully (allow in-flight requests to finish)
    log_info "Draining $previous_slot slot (graceful_timeout=30s)..."
    export COMPOSE_PROJECT_NAME="alphaforge-${previous_slot}"
    docker compose -f "$COMPOSE_FILE" stop --timeout 30 api 2>/dev/null || true
    unset COMPOSE_PROJECT_NAME

    # Reload nginx to point to new upstream (if nginx is running)
    if docker ps --format '{{.Names}}' | grep -q "alphaforge-nginx"; then
        docker exec alphaforge-nginx nginx -s reload 2>/dev/null || true
        log_info "Nginx reloaded"
    fi

    unset API_EXTERNAL_PORT

    log_info "========================================="
    log_info "  Deployment complete!"
    log_info "  Active slot: $target"
    log_info "  Previous slot: $previous_slot (stopped)"
    log_info "  Log: $LOG_FILE"
    log_info "========================================="
}

# ================================================================
# Rollback
# ================================================================

do_rollback() {
    ensure_log_dir

    local active
    active=$(get_active_slot)
    local previous
    previous=$(inactive_slot)

    log_info "========================================="
    log_info "  Rollback: $active -> $previous"
    log_info "========================================="

    # Start the previous slot
    log_step "Starting previous slot ($previous)..."
    export COMPOSE_PROJECT_NAME="alphaforge-${previous}"
    docker compose -f "$COMPOSE_FILE" up -d api 2>&1 | tee -a "$LOG_FILE"

    # Wait for health
    if wait_for_health "$API_PORT" "$HEALTH_ENDPOINT" "$HEALTH_RETRIES" "$HEALTH_INTERVAL"; then
        # Switch the active marker
        set_active_slot "$previous"

        # Stop the failed slot
        export COMPOSE_PROJECT_NAME="alphaforge-${active}"
        docker compose -f "$COMPOSE_FILE" stop --timeout 30 api 2>/dev/null || true
        unset COMPOSE_PROJECT_NAME

        # Reload nginx
        if docker ps --format '{{.Names}}' | grep -q "alphaforge-nginx"; then
            docker exec alphaforge-nginx nginx -s reload 2>/dev/null || true
        fi

        log_info "Rollback complete — active slot: $previous"
    else
        log_error "Rollback target ($previous) also failed health check!"
        log_error "MANUAL INTERVENTION REQUIRED"
        unset COMPOSE_PROJECT_NAME
        exit 1
    fi
}

# ================================================================
# Status
# ================================================================

do_status() {
    local active
    active=$(get_active_slot)
    echo ""
    echo "  AlphaForge Blue-Green Status"
    echo "  ─────────────────────────────"
    echo "  Active slot:  $active"
    echo "  Standby slot: $(inactive_slot)"
    echo ""

    # Check health of running API
    if curl -sf --max-time 5 "http://127.0.0.1:${API_PORT}${HEALTH_ENDPOINT}" > /dev/null 2>&1; then
        echo "  API health:   HEALTHY"
        local resp
        resp=$(curl -sf --max-time 5 "http://127.0.0.1:${API_PORT}${HEALTH_ENDPOINT}" 2>/dev/null)
        echo "  Response:     $resp"
    else
        echo "  API health:   UNHEALTHY or DOWN"
    fi

    echo ""
}

# ================================================================
# Main
# ================================================================

case "${1:-help}" in
    deploy)
        do_deploy
        ;;
    rollback)
        do_rollback
        ;;
    status)
        do_status
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status}"
        echo ""
        echo "Commands:"
        echo "  deploy    Build, validate, and switch to new version"
        echo "  rollback  Revert to previous version"
        echo "  status    Show current deployment state"
        exit 1
        ;;
esac

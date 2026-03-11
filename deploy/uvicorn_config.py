"""
Production Uvicorn / Gunicorn configuration for the AlphaForge Trading Platform.

Usage with gunicorn:
  gunicorn src.api.app:app -c deploy/uvicorn_config.py

Environment variables:
  WORKERS          Override worker count (default: min(2*CPU+1, 8))
  PORT             API port (default: 8000)
  BIND_HOST        Bind address (default: 0.0.0.0)
  LOG_LEVEL        Logging level (default: info)
  GRACEFUL_TIMEOUT Seconds for in-flight requests / open orders to complete (default: 30)
  KEEP_ALIVE       Keep-alive timeout in seconds (default: 5)
  ACCESS_LOG       Enable access logging (default: true)
  MAX_REQUESTS     Restart worker after N requests to prevent memory leaks (default: 10000)
  MAX_REQUESTS_JITTER  Random jitter to prevent all workers restarting simultaneously (default: 1000)
"""

import multiprocessing
import os

# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------


def _worker_count() -> int:
    """Calculate worker count: 2*CPU + 1, capped at 8 for trading systems.

    Trading platforms need headroom for market data processing, model inference,
    and order management. Capping at 8 prevents resource contention that could
    introduce latency in order submission.
    """
    override = os.environ.get("WORKERS")
    if override:
        return max(1, int(override))
    cpu_count = multiprocessing.cpu_count()
    calculated = 2 * cpu_count + 1
    # Cap at 8 for trading — too many workers cause GIL contention on shared
    # model inference and risk checks
    return min(calculated, 8)


workers = _worker_count()
worker_class = "uvicorn.workers.UvicornWorker"

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

bind = f"{os.environ.get('BIND_HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}"

# Keep-alive: time to wait for new requests on an idle connection.
# 5s is reasonable for internal services behind a reverse proxy.
# Increase if clients send bursts with short gaps.
keepalive = int(os.environ.get("KEEP_ALIVE", "5"))

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

# Seconds to finish in-flight requests when receiving SIGTERM.
# 30s allows open orders to complete execution before the process exits.
# Aligns with Kubernetes terminationGracePeriodSeconds.
graceful_timeout = int(os.environ.get("GRACEFUL_TIMEOUT", "30"))
timeout = graceful_timeout + 5  # Hard kill after graceful + 5s buffer

# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

# Restart workers periodically to reclaim leaked memory from ML inference
max_requests = int(os.environ.get("MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.environ.get("MAX_REQUESTS_JITTER", "1000"))

# Preload the app so models are loaded once, then forked to workers
# (saves memory via copy-on-write for PyTorch/XGBoost model weights)
preload_app = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

loglevel = os.environ.get("LOG_LEVEL", "info").lower()

_access_log_enabled = os.environ.get("ACCESS_LOG", "true").lower() in ("true", "1", "yes")
accesslog = "-" if _access_log_enabled else None  # "-" means stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" %(D)sms'
errorlog = "-"

# ---------------------------------------------------------------------------
# Process naming
# ---------------------------------------------------------------------------

proc_name = "alphaforge-api"

# ---------------------------------------------------------------------------
# Server hooks
# ---------------------------------------------------------------------------


def on_starting(server):
    """Log startup configuration."""
    server.log.info(
        "AlphaForge API starting — workers=%d, graceful_timeout=%ds, bind=%s",
        workers,
        graceful_timeout,
        bind,
    )


def post_fork(server, worker):
    """After fork: re-seed random and reset per-worker state."""
    import random

    random.seed()
    server.log.info("Worker %s spawned (pid=%s)", worker.age, worker.pid)


def worker_exit(server, worker):
    """Clean up worker resources on exit."""
    server.log.info("Worker %s exiting (pid=%s)", worker.age, worker.pid)


def on_exit(server):
    """Final cleanup."""
    server.log.info("AlphaForge API server shutting down")

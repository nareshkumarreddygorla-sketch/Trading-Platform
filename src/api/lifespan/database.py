"""
Lifespan: database engine, session factory, repos, trade_store, Redis health check,
Sentry init, production env validation, and WebSocket manager setup.
"""

import logging
import os
import time

from fastapi import FastAPI

logger = logging.getLogger(__name__)


async def init_database(app: FastAPI) -> None:
    """Initialize Sentry, validate production env, set up WebSocket manager,
    persistence layer (DB repos), and Redis health check."""
    # ── Error tracking & observability ──
    from src.api.sentry_setup import init_sentry

    init_sentry()

    # Cold start: trading not allowed until recovery completes
    # Production env validation: require DATABASE_URL and JWT_SECRET when ENV=production
    if os.environ.get("ENV", "").lower() == "production" or os.environ.get("PRODUCTION", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        if not os.environ.get("DATABASE_URL"):
            logger.error("Production: DATABASE_URL is required")
            raise RuntimeError("DATABASE_URL is required in production")
        if not (os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET")):
            logger.error("Production: JWT_SECRET or AUTH_SECRET is REQUIRED")
            raise RuntimeError("JWT_SECRET or AUTH_SECRET is required in production")

    app.state.startup_lock = True
    app.state.recovery_complete = False
    app.state.safe_mode = False

    app.state.order_entry_service = None
    app.state.kill_switch = None
    app.state.persistence_service = None
    app.state.fill_handler = None
    app.state.order_repo = None
    app.state.position_repo = None

    from src.api.ws_manager import ConnectionManager, set_ws_manager

    ws_manager = ConnectionManager()
    set_ws_manager(ws_manager)
    app.state.ws_manager = ws_manager

    persistence_service = None
    order_repo = None
    position_repo = None
    audit_repo = None

    if os.environ.get("DATABASE_URL"):
        try:
            from src.persistence import (
                AuditRepository,
                OrderRepository,
                PersistenceService,
                PositionRepository,
                RiskSnapshotRepository,
                UserRepository,
                get_engine,
            )
            from src.persistence.models import Base

            engine = get_engine()
            Base.metadata.create_all(engine)
            order_repo = OrderRepository()
            position_repo = PositionRepository()
            risk_snapshot_repo = RiskSnapshotRepository()
            audit_repo = AuditRepository()
            user_repo = UserRepository()
            persistence_service = PersistenceService(
                order_repo=order_repo,
                position_repo=position_repo,
                risk_snapshot_repo=risk_snapshot_repo,
            )
            app.state.persistence_service = persistence_service
            app.state.order_repo = order_repo
            app.state.position_repo = position_repo
            app.state.risk_snapshot_repo = risk_snapshot_repo
            app.state.audit_repo = audit_repo
            app.state.user_repo = user_repo
            logger.info("Persistence (orders/positions/risk_snapshot/audit/users) configured")
        except Exception as e:
            logger.error("Persistence required (DATABASE_URL set) but failed: %s", e)
            raise

    # ── Redis startup health check (Sprint 5.5) ──
    app.state.redis_healthy = False
    try:
        from src.core.config import get_settings as _gs

        _redis_url_check = _gs().market_data.redis_url
    except Exception:
        _redis_url_check = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis as _redis_sync

        _rc = _redis_sync.from_url(_redis_url_check, socket_timeout=5)
        _rc.ping()
        _rc.close()
        app.state.redis_healthy = True
        logger.info("Redis health check PASSED (url=%s)", _redis_url_check)
    except Exception as _redis_err:
        _require_redis = os.environ.get("REQUIRE_REDIS", "").lower() in ("true", "1", "yes")
        _paper_mode_env = os.environ.get("PAPER_MODE", "true").lower() in ("true", "1", "yes")
        if _require_redis:
            logger.error("Redis health check FAILED (REQUIRE_REDIS=true): %s", _redis_err)
            raise RuntimeError(f"Redis required but unavailable: {_redis_err}") from _redis_err
        elif _paper_mode_env:
            logger.warning(
                "Redis health check FAILED (paper mode — continuing with in-memory fallback): %s", _redis_err
            )
        else:
            logger.warning("Redis health check FAILED: %s — fill dedup will use in-memory only", _redis_err)


async def shutdown_database(app: FastAPI) -> None:
    """Shutdown: close Redis, flush TradeStore, snapshot positions, stop token blacklist."""
    import json

    # ── Close Redis connection (app.state.redis) ──
    try:
        _redis = getattr(app.state, "redis", None)
        if _redis:
            await _redis.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.debug("Redis close: %s", e)

    # Redis connection pool close
    try:
        _redis_url = os.environ.get("REDIS_URL") or os.environ.get("RATE_LIMIT_REDIS_URL")
        if _redis_url:
            _rpool = getattr(app.state, "_redis_pool", None)
            if _rpool is not None:
                _rpool.close()
                logger.info("Redis connection pool closed")
    except Exception as ex:
        logger.warning("Redis pool close: %s", ex)

    # TradeStore flush
    _ts = getattr(app.state, "trade_store", None)
    if _ts is not None:
        try:
            _flush = getattr(_ts, "flush", None)
            if _flush is not None:
                _flush()
                logger.info("TradeStore flushed")
        except Exception as ex:
            logger.warning("TradeStore flush: %s", ex)

    # Position snapshot — persist all open positions before shutdown
    _rm = getattr(app.state, "risk_manager", None)
    if _rm is not None:
        try:
            positions = getattr(_rm, "positions", [])
            if positions:
                snapshot = {
                    "timestamp": time.time(),
                    "positions": {
                        getattr(pos, "symbol", "unknown"): {
                            "qty": getattr(pos, "qty", getattr(pos, "quantity", 0)),
                            "avg_price": getattr(pos, "avg_price", getattr(pos, "entry_price", 0)),
                            "side": getattr(getattr(pos, "side", None), "value", str(getattr(pos, "side", "unknown"))),
                        }
                        for pos in positions
                    },
                }
                snapshot_path = os.path.join(os.environ.get("DATA_DIR", "data"), "position_snapshot.json")
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                with open(snapshot_path, "w") as f:
                    json.dump(snapshot, f, indent=2)
                logger.info("Position snapshot saved: %d positions -> %s", len(positions), snapshot_path)
        except Exception as ex:
            logger.warning("Position snapshot: %s", ex)

    # Stop token blacklist cleanup thread
    try:
        from src.api.token_blacklist import stop_cleanup_thread

        stop_cleanup_thread()
    except Exception:
        pass

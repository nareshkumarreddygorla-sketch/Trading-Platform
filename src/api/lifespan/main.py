"""
Lifespan orchestrator: calls each init function in order and handles shutdown.
Includes:
- Startup position recovery from DB
- Pre-market readiness gate (all systems must report ready before trading starts)
- Graceful shutdown: persist all state, cancel open orders, flush audit trail
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import init_database, shutdown_database
from .risk import init_risk
from .execution import init_execution, shutdown_execution
from .ai import init_ai
from .market_data import init_market_data, shutdown_market_data
from .trading import init_trading, shutdown_trading

logger = logging.getLogger(__name__)

_GRACEFUL_SHUTDOWN_TIMEOUT = 10  # seconds to wait for pending orders to settle

# Readiness gate: only True after all initialization is complete
_ready = False


def is_ready() -> bool:
    """Check if the application has completed all initialization."""
    return _ready


async def _startup_position_recovery(app: FastAPI) -> None:
    """
    Startup position recovery: load positions from DB, reconcile with broker,
    restore idempotency keys, and initialize risk manager with recovered state.
    """
    logger.info("Starting position recovery from DB...")

    try:
        from src.persistence.position_recovery import PositionRecoveryManager

        recovery_mgr = PositionRecoveryManager()
        app.state.position_recovery_manager = recovery_mgr

        # Determine if we're in live mode
        gateway = getattr(app.state, "gateway", None)
        is_live = gateway is not None and not getattr(gateway, "paper", True)
        risk_manager = getattr(app.state, "risk_manager", None)

        # Build broker position getter for live mode
        broker_get_positions = None
        if is_live and gateway and hasattr(gateway, "get_positions"):
            broker_get_positions = gateway.get_positions

        # Run full recovery workflow
        report = recovery_mgr.run_full_recovery(
            broker_get_positions=broker_get_positions,
            risk_manager=risk_manager,
            is_live=is_live,
        )

        if report.get("safe_mode_recommended"):
            app.state.safe_mode = True
            logger.warning(
                "SAFE MODE activated by position recovery: %s",
                report.get("errors", ["phantom positions detected"]),
            )

        logger.info(
            "Position recovery complete: %d positions, %d idem_keys, safe_mode=%s",
            report.get("positions_recovered", 0),
            report.get("idempotency_keys_loaded", 0),
            report.get("safe_mode_recommended", False),
        )

        # Wire recovery manager to autonomous loop if available
        autonomous_loop = getattr(app.state, "autonomous_loop", None)
        if autonomous_loop is not None:
            autonomous_loop.set_position_recovery_manager(recovery_mgr)

    except Exception as e:
        logger.warning("Position recovery failed (non-fatal): %s", e)


async def _pre_market_readiness_gate(app: FastAPI) -> None:
    """
    Pre-market readiness gate: all critical systems must report ready
    before trading is allowed. Non-blocking in paper mode.
    """
    logger.info("Running pre-market readiness gate...")

    try:
        from src.core.operational_runbook import OperationalRunbook

        runbook = OperationalRunbook(
            app_state=app.state,
            audit_repo=getattr(app.state, "audit_repo", None),
        )
        app.state.operational_runbook = runbook

        readiness = await runbook.pre_market_readiness_check()

        if readiness["ready"]:
            logger.info("Pre-market readiness gate: PASSED — all systems ready")
        else:
            blockers = readiness.get("blockers", [])
            # Determine if we should block trading
            gateway = getattr(app.state, "gateway", None)
            is_live = gateway is not None and not getattr(gateway, "paper", True)

            if is_live:
                logger.critical(
                    "PRE-MARKET READINESS GATE FAILED (LIVE MODE): %d blockers — "
                    "entering safe_mode: %s",
                    len(blockers),
                    ", ".join(blockers),
                )
                app.state.safe_mode = True
            else:
                logger.warning(
                    "Pre-market readiness gate: %d warnings in paper mode (continuing): %s",
                    len(blockers),
                    ", ".join(blockers),
                )

    except Exception as e:
        logger.warning("Pre-market readiness gate failed (non-fatal): %s", e)


async def _graceful_shutdown(app: FastAPI) -> None:
    """Graceful shutdown: persist all state, cancel open orders, flush audit trail."""
    logger.info("Initiating graceful shutdown")

    # 1. Stop the autonomous loop so no new orders are placed
    autonomous_loop = getattr(app.state, "autonomous_loop", None)
    if autonomous_loop is not None:
        try:
            is_running = getattr(autonomous_loop, "_running", False)
            if is_running:
                await autonomous_loop.stop()
                logger.info("Autonomous loop stopped during graceful shutdown")
        except Exception as e:
            logger.warning("Autonomous loop stop failed during graceful shutdown: %s", e)

    # 1b. Cancel all pending orders at broker (prevent phantom fills after shutdown)
    oes = getattr(app.state, "order_entry_service", None)
    if oes is not None:
        lifecycle = getattr(oes, "lifecycle", None)
        gateway = getattr(app.state, "gateway", None)
        if lifecycle is not None and gateway is not None:
            try:
                pending_fn = getattr(lifecycle, "get_pending_orders", None)
                if pending_fn is not None:
                    pending = pending_fn()
                    if pending:
                        logger.warning("Cancelling %d pending orders at broker on shutdown", len(pending))
                        for order in pending:
                            broker_id = getattr(order, "broker_order_id", None)
                            if broker_id:
                                try:
                                    cancel_fn = getattr(gateway, "cancel_order", None)
                                    if cancel_fn:
                                        await asyncio.wait_for(cancel_fn(broker_id), timeout=5.0)
                                        logger.info("Cancelled order %s at broker", broker_id)
                                except (asyncio.TimeoutError, Exception) as e:
                                    logger.warning("Failed to cancel order %s at broker: %s", broker_id, e)
            except Exception as e:
                logger.warning("Pending order cancellation on shutdown failed: %s", e)

    # 2. Persist all open positions to DB (crash-safe state snapshot)
    try:
        recovery_mgr = getattr(app.state, "position_recovery_manager", None)
        if recovery_mgr and autonomous_loop:
            open_trades = getattr(autonomous_loop, "_open_trades", {})
            persisted = 0
            for trade_key, trade in open_trades.items():
                try:
                    recovery_mgr.persist_position_change(
                        symbol=trade.get("symbol", trade_key.split(":")[0]),
                        exchange=trade.get("exchange", "NSE"),
                        side=trade.get("side", "BUY"),
                        quantity=trade.get("qty", 0),
                        avg_price=trade.get("entry_price", 0),
                        strategy_id=trade.get("strategy_id"),
                        change_type="update",
                        reason="graceful_shutdown_snapshot",
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning("Failed to persist position %s on shutdown: %s", trade_key, e)
            if persisted:
                logger.info("Persisted %d positions to DB during graceful shutdown", persisted)
    except Exception as e:
        logger.warning("Position persistence during shutdown: %s", e)

    # 3. Wait up to 10 seconds for pending orders to settle
    oes = getattr(app.state, "order_entry_service", None)
    if oes is not None:
        lifecycle = getattr(oes, "lifecycle", None)
        if lifecycle is not None:
            pending_fn = getattr(lifecycle, "get_pending_orders", None)
            if pending_fn is not None:
                elapsed = 0.0
                while elapsed < _GRACEFUL_SHUTDOWN_TIMEOUT:
                    try:
                        pending = pending_fn()
                        if not pending:
                            break
                        logger.info(
                            "Waiting for %d pending orders to settle (%.0fs remaining)",
                            len(pending),
                            _GRACEFUL_SHUTDOWN_TIMEOUT - elapsed,
                        )
                    except Exception:
                        break
                    await asyncio.sleep(1.0)
                    elapsed += 1.0

    # 4. Flush audit trail
    try:
        audit_repo = getattr(app.state, "audit_repo", None)
        if audit_repo:
            audit_repo.append_sync(
                event_type="graceful_shutdown",
                actor="system",
                payload={
                    "open_trades": len(getattr(autonomous_loop, "_open_trades", {})) if autonomous_loop else 0,
                    "daily_pnl": getattr(autonomous_loop, "_daily_pnl", 0.0) if autonomous_loop else 0.0,
                },
            )
            logger.info("Audit trail flushed during graceful shutdown")
    except Exception as e:
        logger.debug("Audit trail flush: %s", e)

    # 5. Close database connections
    try:
        from src.persistence import get_engine
        engine = get_engine()
        if engine is not None:
            engine.dispose()
            logger.info("Database engine disposed")
    except Exception as e:
        logger.debug("Database engine dispose: %s", e)

    # 6. Cancel background AI tasks
    for task_name in ("_drift_task", "_walk_forward_task", "_retrain_task"):
        task = getattr(app.state, task_name, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            logger.info("Cancelled background task: %s", task_name)

    # 7. Close Redis connections
    try:
        import redis as _redis_sync
        import os
        _redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _rc = _redis_sync.from_url(_redis_url, socket_timeout=2)
        _rc.close()
        logger.info("Redis connection closed during graceful shutdown")
    except Exception as e:
        logger.debug("Redis close during graceful shutdown: %s", e)


def _wire_ws_snapshot_provider(app: FastAPI) -> None:
    """Register a snapshot provider on the WebSocket manager.

    The provider is a closure over ``app.state`` that collects the current
    dashboard state from risk_manager, kill_switch, circuit_breaker, and
    market_data_service.  The ConnectionManager calls it every 5 seconds and
    broadcasts the result as a ``snapshot`` event to all connected clients.
    """
    ws_mgr = getattr(app.state, "ws_manager", None)
    if ws_mgr is None:
        logger.warning("WebSocket manager not available; snapshot provider not wired")
        return

    import time as _time

    def _build_snapshot() -> dict:
        rm = getattr(app.state, "risk_manager", None)
        ks = getattr(app.state, "kill_switch", None)
        cb = getattr(app.state, "circuit_breaker", None)
        al = getattr(app.state, "autonomous_loop", None)

        # Equity and PnL
        equity = rm.equity if rm else 0.0
        daily_pnl = rm.daily_pnl if rm else 0.0

        # Open positions count
        open_positions_count = len(rm.positions) if rm else 0

        # Circuit breaker state
        circuit_open = False
        if rm:
            circuit_open = rm.is_circuit_open()

        # Kill switch state
        kill_switch_armed = False
        if ks:
            # KillSwitch.is_armed() is async but we need sync here;
            # read the internal state directly for the snapshot.
            ks_state = getattr(ks, "_state", None)
            if ks_state is not None:
                kill_switch_armed = getattr(ks_state, "armed", False)

        # Market data feed health
        market_feed_healthy = True
        last_tick_ts = None
        mds = getattr(app.state, "market_data_service", None)
        if mds is not None:
            try:
                market_feed_healthy = mds.is_healthy()
            except Exception:
                market_feed_healthy = False
        else:
            # Fallback: check bar cache freshness
            bc = getattr(app.state, "bar_cache", None)
            if bc is not None:
                last_ts_fn = getattr(bc, "last_bar_timestamp", None)
                if last_ts_fn is not None:
                    try:
                        last_ts = last_ts_fn()
                        if last_ts:
                            last_tick_ts = last_ts
                            if (_time.time() - last_ts) > 120:
                                market_feed_healthy = False
                    except Exception:
                        pass

        # Autonomous loop state
        open_trades = len(getattr(al, "_open_trades", {})) if al else 0
        tick_count = getattr(al, "_tick_count", 0) if al else 0

        # Safe mode
        safe_mode = getattr(app.state, "safe_mode", False)

        return {
            "equity": equity,
            "daily_pnl": daily_pnl,
            "open_positions_count": open_positions_count,
            "circuit_open": circuit_open,
            "kill_switch_armed": kill_switch_armed,
            "market_feed_healthy": market_feed_healthy,
            "last_tick_ts": last_tick_ts,
            "open_trades": open_trades,
            "tick_count": tick_count,
            "safe_mode": safe_mode,
        }

    ws_mgr.set_snapshot_provider(_build_snapshot)
    logger.info("WebSocket snapshot provider wired (broadcasts every 5s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_database(app)

    # ── Pre-flight database health check (Sprint 11.3) ──
    try:
        from src.persistence.database import check_database_health
        db_health = check_database_health()
        if not db_health["healthy"]:
            logger.critical("DATABASE HEALTH CHECK FAILED: %s — trading will be disabled", db_health["error"])
        else:
            logger.info("Database health OK (size=%.1fGB, pool=%d)", db_health["size_gb"], db_health["pool_size"])
    except Exception as e:
        logger.warning("Database health check unavailable: %s", e)

    await init_execution(app)
    await init_risk(app)
    await init_market_data(app)
    await init_ai(app)
    await init_trading(app)

    # ── Startup position recovery from DB ──
    await _startup_position_recovery(app)

    # ── Pre-market readiness gate ──
    await _pre_market_readiness_gate(app)

    # ── Wire WebSocket snapshot provider (broadcasts dashboard state every 5s) ──
    _wire_ws_snapshot_provider(app)

    global _ready
    _ready = True
    logger.info("Application startup complete — all systems initialized")
    yield
    _ready = False
    await _graceful_shutdown(app)
    await shutdown_trading(app)
    await shutdown_market_data(app)
    await shutdown_execution(app)
    await shutdown_database(app)
    logger.info("API shutdown complete")

"""
FastAPI application: health, market, strategies, risk, orders, backtest, metrics.
"""
from contextlib import asynccontextmanager
import logging
import time

# Load .env before anything reads os.environ
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import HTMLResponse, Response

from .routers import health, strategies, risk, orders, market, backtest, trading, alpha_research, reconciliation, auth, capital, audit, performance, agents, training, broker

logger = logging.getLogger(__name__)


def _prometheus_metrics() -> Response:
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter, CollectorRegistry, REGISTRY

        # Register trading gauges (idempotent via try/except)
        def _get_or_create_gauge(name, doc):
            try:
                return Gauge(name, doc)
            except ValueError:
                # Already registered
                c = REGISTRY._names_to_collectors.get(name)
                if c is not None:
                    return c
                return Gauge(name, doc)

        try:
            # Get app state from module-level app
            _app = globals().get("app")
            if _app:
                rm = getattr(_app.state, "risk_manager", None)
                if rm:
                    _get_or_create_gauge("trading_equity", "Current portfolio equity").set(rm.equity)
                    _get_or_create_gauge("trading_daily_pnl", "Daily PnL").set(rm.daily_pnl)
                    _get_or_create_gauge("trading_open_positions", "Number of open positions").set(len(rm.positions))
                    _get_or_create_gauge("trading_circuit_open", "Circuit breaker state (1=open)").set(1 if rm.is_circuit_open() else 0)

                cb = getattr(_app.state, "circuit_breaker", None)
                if cb:
                    peak = getattr(cb, "_peak_equity", 0)
                    _get_or_create_gauge("trading_peak_equity", "Peak equity").set(peak)

                ks = getattr(_app.state, "kill_switch", None)
                if ks:
                    _get_or_create_gauge("trading_kill_switch_armed", "Kill switch armed (1=yes)").set(1 if getattr(ks, "_armed", False) else 0)

                sm = getattr(_app.state, "safe_mode", False)
                _get_or_create_gauge("trading_safe_mode", "Safe mode active (1=yes)").set(1 if sm else 0)

                al = getattr(_app.state, "autonomous_loop", None)
                if al:
                    _get_or_create_gauge("trading_autonomous_ticks", "Autonomous loop tick count").set(getattr(al, "_tick_count", 0))
                    _get_or_create_gauge("trading_open_trades", "Autonomous loop tracked trades").set(len(getattr(al, "_open_trades", {})))

                # Sprint 10.10: Additional Prometheus metrics
                mr = getattr(_app.state, "model_registry", None)
                if mr:
                    _get_or_create_gauge("trading_models_loaded", "Number of loaded ML models").set(
                        sum(1 for m in mr.list_all() if getattr(m, "_loaded", True))
                    )
                ws_mgr = getattr(_app.state, "ws_manager", None)
                if ws_mgr:
                    _get_or_create_gauge("trading_ws_connections", "Active WebSocket connections").set(
                        len(getattr(ws_mgr, "_active_connections", []))
                    )
        except Exception:
            pass

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(
            content="prometheus_client not installed",
            media_type="text/plain",
            status_code=503,
        )



@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    import os
    import sys

    _snapshot_task = None
    _heartbeat_task = None
    # Cold start: trading not allowed until recovery completes
    # Production env validation: require DATABASE_URL and JWT_SECRET when ENV=production
    if os.environ.get("ENV", "").lower() == "production" or os.environ.get("PRODUCTION", "").lower() in ("1", "true", "yes"):
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

    from .ws_manager import ConnectionManager, set_ws_manager
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
                get_engine,
                PersistenceService,
                OrderRepository,
                PositionRepository,
                RiskSnapshotRepository,
                AuditRepository,
                UserRepository,
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
            raise RuntimeError(f"Redis required but unavailable: {_redis_err}")
        elif _paper_mode_env:
            logger.warning("Redis health check FAILED (paper mode — continuing with in-memory fallback): %s", _redis_err)
        else:
            logger.warning("Redis health check FAILED: %s — fill dedup will use in-memory only", _redis_err)

    safe_mode = False
    try:
        from src.risk_engine import RiskManager
        from src.risk_engine.circuit_breaker import CircuitBreaker
        from src.execution import OrderRouter, OrderLifecycle, AngelOneExecutionGateway
        from src.execution.order_entry import OrderEntryService, IdempotencyStore, KillSwitch, ExposureReservation
        from src.execution.order_entry.redis_distributed_lock import RedisDistributedLock
        from src.execution.order_entry.redis_cluster_reservation import RedisClusterReservation
        from src.core.events import OrderStatus
        from src.core.config import get_settings
        from src.risk_engine.limits import RiskLimits
        _settings = get_settings()
        _exec = _settings.execution
        _risk_cfg = _settings.risk
        _limits = RiskLimits(
            max_position_pct=_risk_cfg.max_position_pct,
            max_daily_loss_pct=_risk_cfg.max_daily_loss_pct,
            max_open_positions=_risk_cfg.max_open_positions,
            circuit_breaker_drawdown_pct=_risk_cfg.circuit_breaker_drawdown_pct,
            max_sector_concentration_pct=_risk_cfg.max_sector_concentration_pct,
            var_limit_pct=_risk_cfg.var_limit_pct,
            max_consecutive_losses=_risk_cfg.max_consecutive_losses,
            max_per_symbol_pct=_risk_cfg.max_per_symbol_pct,
        )

        def _set_safe_mode_cb():
            app.state.safe_mode = True
            app.state.safe_mode_since = time.time()
            if getattr(app.state, "audit_repo", None):
                try:
                    app.state.audit_repo.append_sync("broker_failure", "system", {"reason": "heartbeat_failure"})
                except Exception as e:
                    logger.warning("Audit append failed (broker_failure): %s", e)

        risk_manager = RiskManager(equity=100_000.0, limits=_limits)
        # Paper execution mode: never place real broker orders (live market data can still be used)
        paper_mode = getattr(_settings, "paper_execution_mode", None)
        if paper_mode is None:
            paper_mode = getattr(_exec, "paper_execution_mode", _exec.paper)
        paper_mode = bool(paper_mode)
        gateway = AngelOneExecutionGateway(
            api_key=_exec.angel_one_api_key or "",
            api_secret=_exec.angel_one_api_secret or "",
            access_token=_exec.angel_one_token or "",
            paper=paper_mode,
            refresh_token=_exec.angel_one_refresh_token,
            client_code=_exec.angel_one_client_code,
            password=_exec.angel_one_password,
            totp=_exec.angel_one_totp,
            totp_secret=_exec.angel_one_totp_secret,
            request_timeout=_exec.angel_one_request_timeout,
            on_health_failure=_set_safe_mode_cb,
        )
        app.state.gateway = gateway
        if paper_mode:
            logger.info("PAPER EXECUTION MODE ACTIVE: no real broker orders will be placed")
        circuit_breaker = CircuitBreaker(risk_manager)
        app.state.circuit_breaker = circuit_breaker
        router = OrderRouter(gateway)
        lifecycle = OrderLifecycle()
        _redis_url = _settings.market_data.redis_url
        idempotency = IdempotencyStore(redis_url=_redis_url)
        kill_switch = KillSwitch()
        reservation = ExposureReservation()
        distributed_lock = RedisDistributedLock(redis_url=_redis_url)
        cluster_reservation = RedisClusterReservation(redis_url=_redis_url)

        # Cold start recovery: load DB state and reconcile with broker before allowing trading
        if order_repo is not None and position_repo is not None:
            from src.startup.recovery import run_cold_start_recovery
            is_live = not getattr(gateway, "paper", True)
            get_risk_snapshot = (persistence_service.get_risk_snapshot_sync if persistence_service else None)
            safe_mode, _ = await run_cold_start_recovery(
                order_repo,
                position_repo,
                risk_manager,
                lifecycle,
                get_broker_positions=gateway.get_positions if is_live else None,
                is_live_mode=is_live,
                get_risk_snapshot=get_risk_snapshot,
            )
            app.state.safe_mode = safe_mode

        async def persist_order_cb(order):
            if persistence_service:
                await persistence_service.persist_order(order)

        async def persist_order_submitting_cb(order, idempotency_key=None):
            if persistence_service:
                await persistence_service.persist_order_submitting(order, idempotency_key=idempotency_key)

        def update_order_after_broker_ack_sync(oid, broker_order_id):
            if persistence_service:
                return persistence_service.update_order_after_broker_ack_sync(oid, broker_order_id)
            return False

        def reject_order_submitting_sync(oid):
            if persistence_service:
                from src.core.events import OrderStatus as OS
                persistence_service.update_order_status_sync(oid, OS.REJECTED)

        def _on_risk_rejected():
            ckc = getattr(app.state, "circuit_kill_controller", None)
            if ckc is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(ckc.record_rejection())
                except RuntimeError:
                    logger.warning("No running event loop for record_rejection")

        async def _broadcast_order_created(o):
            from .ws_manager import get_ws_manager
            mgr = get_ws_manager()
            if mgr:
                await mgr.broadcast({
                    "type": "order_created",
                    "order_id": getattr(o, "order_id", None),
                    "broker_order_id": getattr(o, "broker_order_id", None),
                    "symbol": getattr(o, "symbol", ""),
                    "exchange": getattr(getattr(o, "exchange", None), "value", str(getattr(o, "exchange", ""))),
                    "side": getattr(o, "side", ""),
                    "quantity": getattr(o, "quantity", 0),
                    "status": getattr(getattr(o, "status", None), "value", str(getattr(o, "status", ""))),
                })

        # GAP-18: Create rate limiter for order flood protection
        from src.execution.order_entry.rate_limiter import OrderRateLimiter, RateLimitConfig
        _rate_limiter = OrderRateLimiter(RateLimitConfig(max_orders_per_minute=60))

        # GAP-17: Wire Kafka publisher when KAFKA_BROKER_URL is configured
        _kafka_publish_fn = None
        _kafka_broker_url = os.environ.get("KAFKA_BROKER_URL") or os.environ.get("MD_KAFKA_BROKERS")
        if _kafka_broker_url:
            try:
                from aiokafka import AIOKafkaProducer
                import json as _json
                _kafka_producer = AIOKafkaProducer(
                    bootstrap_servers=_kafka_broker_url,
                    value_serializer=lambda v: _json.dumps(v, default=str).encode("utf-8"),
                )

                async def _kafka_publish_order(order):
                    try:
                        if not _kafka_producer._sender or not _kafka_producer._sender.sender_task:
                            await _kafka_producer.start()
                        await _kafka_producer.send_and_wait(
                            "trading.orders",
                            value={
                                "order_id": getattr(order, "order_id", None),
                                "symbol": getattr(order, "symbol", ""),
                                "side": getattr(getattr(order, "side", None), "value", ""),
                                "quantity": getattr(order, "quantity", 0),
                                "status": getattr(getattr(order, "status", None), "value", ""),
                                "broker_order_id": getattr(order, "broker_order_id", None),
                            },
                        )
                    except Exception as e:
                        logger.warning("Kafka publish failed: %s", e)

                _kafka_publish_fn = _kafka_publish_order
                app.state._kafka_producer = _kafka_producer  # Store for shutdown (Sprint 10.9)
                logger.info("Kafka order publishing enabled (broker=%s)", _kafka_broker_url)
            except ImportError:
                logger.info("aiokafka not installed — Kafka publishing disabled")
            except Exception as e:
                logger.warning("Kafka setup failed: %s", e)

        app.state.order_entry_service = OrderEntryService(
            risk_manager=risk_manager,
            order_router=router,
            lifecycle=lifecycle,
            idempotency_store=idempotency,
            kill_switch=kill_switch,
            reservation=reservation,
            persist_order=persist_order_cb if persistence_service else None,
            persist_order_submitting=persist_order_submitting_cb if persistence_service else None,
            update_order_after_broker_ack=update_order_after_broker_ack_sync if persistence_service else None,
            reject_order_submitting=reject_order_submitting_sync if persistence_service else None,
            publish_order_event=_kafka_publish_fn,
            distributed_lock=distributed_lock,
            cluster_reservation=cluster_reservation,
            rate_limiter=_rate_limiter,
            on_risk_rejected=_on_risk_rejected,
            on_order_created=_broadcast_order_created,
        )
        app.state.kill_switch = kill_switch
        app.state.risk_manager = risk_manager

        from src.execution.order_entry.circuit_and_kill import CircuitAndKillController, CircuitKillConfig
        circuit_kill_config = CircuitKillConfig(
            max_daily_loss_pct=_risk_cfg.max_daily_loss_pct,
            max_drawdown_pct=_risk_cfg.circuit_breaker_drawdown_pct,
        )
        app.state.circuit_kill_controller = CircuitAndKillController(
            risk_manager=risk_manager,
            kill_switch=kill_switch,
            config=circuit_kill_config,
        )

        from src.execution.fill_handler import FillHandler
        from src.execution.fill_handler.events import FillType

        async def on_fill_persist(event):
            if persistence_service:
                status = OrderStatus.REJECTED if event.fill_type == FillType.REJECT else (
                    OrderStatus.CANCELLED if event.fill_type == FillType.CANCEL else (
                        OrderStatus.FILLED if event.fill_type == FillType.FILL else OrderStatus.PARTIALLY_FILLED
                    ))
                await persistence_service.persist_fill(
                    order_id=event.order_id,
                    status=status,
                    filled_qty=event.filled_qty,
                    avg_price=event.avg_price,
                    symbol=event.symbol,
                    exchange=getattr(event.exchange, "value", str(event.exchange)),
                    side=event.side,
                    strategy_id=event.strategy_id or None,
                )
            from .ws_manager import get_ws_manager
            mgr = get_ws_manager()
            if mgr:
                await mgr.broadcast({
                    "type": "order_filled",
                    "order_id": event.order_id,
                    "symbol": event.symbol,
                    "side": event.side,
                    "filled_qty": event.filled_qty,
                    "avg_price": event.avg_price,
                    "fill_type": getattr(event.fill_type, "value", str(event.fill_type)),
                })
                rm = getattr(app.state, "risk_manager", None)
                if rm:
                    positions = [{"symbol": p.symbol, "exchange": getattr(p.exchange, "value", str(p.exchange)), "side": getattr(p.side, "value", str(p.side)), "quantity": p.quantity, "avg_price": getattr(p, "avg_price", 0.0)} for p in rm.positions]
                    await mgr.broadcast({"type": "position_updated", "positions": positions})
                    await mgr.broadcast({"type": "equity_updated", "equity": rm.equity, "daily_pnl": rm.daily_pnl})

        from src.ai.performance_tracker import PerformanceTracker
        from .ws_manager import get_ws_manager
        def _broadcast_strategy_disabled(sid: str, reason: str):
            mgr = get_ws_manager()
            if mgr:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                loop.create_task(mgr.broadcast({"type": "strategy_disabled", "strategy_id": sid, "reason": reason}))
        def _broadcast_exposure_multiplier(sid: str, mult: float):
            mgr = get_ws_manager()
            if mgr:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                loop.create_task(mgr.broadcast({"type": "exposure_multiplier_changed", "strategy_id": sid, "multiplier": mult}))
        performance_tracker = PerformanceTracker(on_strategy_disabled=_broadcast_strategy_disabled, on_exposure_multiplier_changed=_broadcast_exposure_multiplier)
        app.state.performance_tracker = performance_tracker
        def _on_fill_callback(ev):
            pnl = getattr(ev, "realized_pnl", None) or getattr(ev, "pnl", None) or 0.0
            performance_tracker.record_fill(ev.strategy_id or "unknown", pnl)
        app.state.fill_handler = FillHandler(
            risk_manager=risk_manager,
            lifecycle=lifecycle,
            on_fill_persist=on_fill_persist if persistence_service else None,
            order_lock=app.state.order_entry_service._global_lock,
            on_fill_callback=_on_fill_callback,
        )
        from src.execution.fill_handler import FillListener, PaperFillSimulator
        app.state.fill_listener = None
        app.state.paper_fill_simulator = None
        if not getattr(gateway, "paper", True):
            fill_listener = FillListener(gateway, app.state.fill_handler, poll_interval_seconds=10.0)
            fill_listener.start()
            app.state.fill_listener = fill_listener
        else:
            # Paper mode: auto-fill PENDING orders so positions/P&L work
            paper_sim = PaperFillSimulator(
                lifecycle=lifecycle,
                fill_handler=app.state.fill_handler,
                fill_delay_seconds=3.0,
                poll_interval_seconds=5.0,
            )
            paper_sim.start()
            app.state.paper_fill_simulator = paper_sim
            logger.info("Paper fill simulator started for paper trading mode")
        logger.info("OrderEntryService and KillSwitch configured")
    except Exception as e:
        logger.warning("OrderEntryService not configured: %s", e)

    # Recovery complete; release startup lock so /trading/ready can pass (unless safe_mode)
    app.state.recovery_complete = True
    app.state.startup_lock = False

    # ── Institutional Risk Modules ──
    try:
        from src.risk_engine.var import PortfolioVaR
        app.state.portfolio_var = PortfolioVaR(horizon_days=1)
        logger.info("PortfolioVaR initialized (95%%/99%% parametric VaR)")
    except Exception as e:
        logger.warning("PortfolioVaR not initialized: %s", e)

    try:
        from src.risk_engine.correlation import CorrelationGuard
        app.state.correlation_guard = CorrelationGuard(max_pairwise_correlation=0.70, max_portfolio_vol_pct=15.0)
        logger.info("CorrelationGuard initialized (max_corr=0.70, max_port_vol=15%%)")
    except Exception as e:
        logger.warning("CorrelationGuard not initialized: %s", e)

    try:
        from src.risk_engine.sector_map import SectorClassifier
        app.state.sector_classifier = SectorClassifier()
        logger.info("SectorClassifier initialized (%d sectors)", len(app.state.sector_classifier.list_sectors()))
    except Exception as e:
        logger.warning("SectorClassifier not initialized: %s", e)

    try:
        from src.risk_engine.gap_risk import GapRiskManager
        app.state.gap_risk_manager = GapRiskManager()
        logger.info("GapRiskManager initialized")
    except Exception as e:
        logger.warning("GapRiskManager not initialized: %s", e)

    try:
        from src.risk_engine.tail_risk import TailRiskProtector
        app.state.tail_risk_protector = TailRiskProtector()
        logger.info("TailRiskProtector initialized (VIX + rapid drawdown defense)")
    except Exception as e:
        logger.warning("TailRiskProtector not initialized: %s", e)

    try:
        from src.risk_engine.vol_targeting import VolatilityTargeter
        app.state.vol_targeter = VolatilityTargeter(target_vol_annual=12.0)
        logger.info("VolatilityTargeter initialized (target=12%% annualized)")
    except Exception as e:
        logger.warning("VolatilityTargeter not initialized: %s", e)

    try:
        from src.execution.freeze_qty import FreezeQuantityManager
        app.state.freeze_qty_manager = FreezeQuantityManager()
        logger.info("FreezeQuantityManager initialized")
    except Exception as e:
        logger.warning("FreezeQuantityManager not initialized: %s", e)

    try:
        from src.execution.market_impact import MarketImpactModel
        app.state.market_impact_model = MarketImpactModel(gamma=0.25)
        logger.info("MarketImpactModel initialized (Almgren-Chriss)")
    except Exception as e:
        logger.warning("MarketImpactModel not initialized: %s", e)

    try:
        from src.alerts.notifier import AlertNotifier
        app.state.alert_notifier = AlertNotifier()
        logger.info("AlertNotifier initialized (channels: %s)", app.state.alert_notifier.config.enabled_channels)
    except Exception as e:
        logger.warning("AlertNotifier not initialized: %s", e)

    # ── Wire institutional risk modules into RiskManager ──
    rm = getattr(app.state, "risk_manager", None)
    if rm is not None:
        rm._portfolio_var = getattr(app.state, "portfolio_var", None)
        rm._correlation_guard = getattr(app.state, "correlation_guard", None)
        rm._tail_risk_protector = getattr(app.state, "tail_risk_protector", None)
        wired = [n for n in ("portfolio_var", "correlation_guard", "tail_risk_protector") if getattr(rm, f"_{n}") is not None]
        if wired:
            logger.info("RiskManager: institutional modules wired: %s", wired)

    # ── Load equity from DB snapshot (GAP-6 fix) ──
    if rm is not None:
        try:
            snap_repo = getattr(app.state, "risk_snapshot_repo", None)
            if snap_repo is not None:
                snap = snap_repo.get_latest_sync()
                if snap and getattr(snap, "equity", 0) > 0:
                    rm.equity = snap.equity
                    rm.daily_pnl = getattr(snap, "daily_pnl", 0.0)
                    logger.info("RiskManager equity loaded from DB snapshot: %.2f", rm.equity)
        except Exception as e:
            logger.warning("Could not load equity from DB snapshot: %s", e)


    try:
        from src.compliance.audit_trail import SEBIAuditTrail
        app.state.sebi_audit = SEBIAuditTrail()
        logger.info("SEBI audit trail initialized (append-only)")
    except Exception as e:
        logger.debug("SEBI audit trail not initialized: %s", e)

    try:
        from src.reporting.daily_report import DailyReportGenerator
        app.state.daily_report_generator = DailyReportGenerator(
            risk_manager=getattr(app.state, "risk_manager", None),
            persistence_service=getattr(app.state, "persistence_service", None),
        )
        logger.info("DailyReportGenerator initialized")
    except Exception as e:
        logger.debug("DailyReportGenerator not initialized: %s", e)

    # ── ADV Cache (Sprint 7.7) ──
    try:
        from src.market_data.adv_cache import ADVCache
        _adv_cache = ADVCache()
        app.state.adv_cache = _adv_cache
        logger.info("ADVCache initialized")
    except Exception as e:
        logger.debug("ADVCache not initialized: %s", e)

    # ── Gap Risk Manager (Sprint 7.3) ──
    _gap_risk_task = None
    try:
        from src.risk_engine.gap_risk import GapRiskManager
        _gap_risk_mgr = GapRiskManager()
        app.state.gap_risk_manager = _gap_risk_mgr
        logger.info("GapRiskManager initialized")
    except Exception as e:
        logger.debug("GapRiskManager not initialized: %s", e)

    # Periodic risk snapshot (equity, daily_pnl) and circuit breaker equity update
    if getattr(app.state, "persistence_service", None) and getattr(app.state, "risk_manager", None):
        _loop = asyncio.get_running_loop()
        _cb = getattr(app.state, "circuit_breaker", None)

        async def _periodic_risk_snapshot():
            while True:
                await asyncio.sleep(60)
                try:
                    rm = getattr(app.state, "risk_manager", None)
                    ps = getattr(app.state, "persistence_service", None)
                    if rm and ps:
                        _rm, _ps = rm, ps
                        await _loop.run_in_executor(None, lambda: _ps.save_risk_snapshot_sync(_rm.equity, _rm.daily_pnl))
                    if _cb and rm:
                        _cb.update_equity(rm.equity)
                    # Auto circuit/kill on daily loss or drawdown
                    ckc = getattr(app.state, "circuit_kill_controller", None)
                    if ckc and rm:
                        await ckc.check_daily_loss_and_trip(rm.equity)
                        peak = getattr(_cb, "_peak_equity", rm.equity) if _cb else rm.equity
                        await ckc.check_drawdown_and_trip(peak, rm.equity)

                    # ── Tail Risk: VIX + rapid drawdown monitoring ──
                    _trp = getattr(app.state, "tail_risk_protector", None)
                    if _trp and rm:
                        try:
                            _trp.record_equity(rm.equity)
                            _vix_lvl = getattr(getattr(_trp, 'state', None), 'vix_level', None)
                            if _vix_lvl and hasattr(_vix_lvl, 'value'):
                                if _vix_lvl.value == "EXTREME":
                                    ks = getattr(app.state, "kill_switch", None)
                                    if ks:
                                        from src.execution.order_entry.kill_switch import KillReason
                                        await ks.arm(KillReason.INDIA_VIX_SPIKE, "India VIX EXTREME (>35)")
                                    _an = getattr(app.state, "alert_notifier", None)
                                    if _an:
                                        from src.alerts.notifier import AlertSeverity
                                        await _an.send(AlertSeverity.CRITICAL, "India VIX Extreme", "VIX > 35 — kill switch armed", source="tail_risk")
                                elif _vix_lvl.value == "HIGH":
                                    _an = getattr(app.state, "alert_notifier", None)
                                    if _an:
                                        from src.alerts.notifier import AlertSeverity
                                        await _an.send(AlertSeverity.WARNING, "India VIX High", "VIX 25-35 — exposure halved", source="tail_risk")
                        except Exception as e:
                            logger.debug("Tail risk check: %s", e)

                    # ── Portfolio VaR update ──
                    _pvar = getattr(app.state, "portfolio_var", None)
                    if _pvar and rm:
                        try:
                            positions = [{"symbol": p.symbol, "notional": abs(getattr(p, "avg_price", 0) * p.quantity)} for p in rm.positions]
                            var_result = _pvar.compute(positions, rm.equity)
                            app.state._last_var = var_result
                        except Exception as e:
                            logger.debug("VaR compute: %s", e)

                    # ── Vol Targeting daily return update ──
                    _vt = getattr(app.state, "vol_targeter", None)
                    if _vt and rm:
                        try:
                            daily_return = rm.daily_pnl / rm.equity if rm.equity > 0 else 0.0
                            _vt.record_daily_return(daily_return)
                        except Exception as e:
                            logger.debug("Vol targeting: %s", e)

                    # ── Cross-asset data refresh (VIX, USDINR) ──
                    try:
                        import yfinance as _yf
                        _trp_ca = getattr(app.state, "tail_risk_protector", None)
                        # India VIX
                        _vix_data = _yf.Ticker("^INDIAVIX").history(period="1d")
                        if not _vix_data.empty:
                            _vix_val = float(_vix_data["Close"].iloc[-1])
                            app.state._india_vix = _vix_val
                            if _trp_ca:
                                _trp_ca.update_vix(_vix_val)
                        # USDINR
                        _fx_data = _yf.Ticker("USDINR=X").history(period="2d")
                        if not _fx_data.empty and len(_fx_data) >= 2:
                            _fx_ret = float((_fx_data["Close"].iloc[-1] / _fx_data["Close"].iloc[-2]) - 1.0)
                            app.state._usdinr_return = _fx_ret
                        elif not _fx_data.empty:
                            app.state._usdinr_return = 0.0
                    except Exception as e:
                        logger.debug("Cross-asset data refresh: %s", e)

                    # ── Kill switch auto-disarm (Sprint 7.4) ──
                    ks = getattr(app.state, "kill_switch", None)
                    if ks and ks.is_armed():
                        try:
                            _last_broker_ok_ts = getattr(app.state, "_last_broker_ok_ts", 0)
                            _broker_ok = (time.time() - _last_broker_ok_ts) < 120 if _last_broker_ok_ts else False
                            _vix_val = getattr(app.state, "_india_vix", None)
                            disarmed = await ks.check_auto_disarm(broker_healthy=_broker_ok, vix_value=_vix_val)
                            if disarmed:
                                logger.info("Kill switch auto-disarmed (recovery conditions met)")
                        except Exception as e:
                            logger.debug("Kill switch auto-disarm check: %s", e)

                    # ── Safe mode auto-clear (Sprint 7.10) ──
                    if getattr(app.state, "safe_mode", False):
                        _safe_since = getattr(app.state, "safe_mode_since", 0)
                        _last_broker_ok_ts = getattr(app.state, "_last_broker_ok_ts", 0)
                        _broker_ok = (time.time() - _last_broker_ok_ts) < 120 if _last_broker_ok_ts else False
                        _feed_ok = True
                        _mds_sm = getattr(app.state, "market_data_service", None)
                        if _mds_sm:
                            _feed_ok = _mds_sm.is_healthy()
                        _circuit_closed = not rm.is_circuit_open() if rm else True
                        if _broker_ok and _feed_ok and _circuit_closed:
                            _cool_down = time.time() - _safe_since if _safe_since else 999
                            if _cool_down > 300:  # 5 min minimum cool-down
                                app.state.safe_mode = False
                                logger.info("Safe mode AUTO-CLEARED after %.0fs (broker OK, feed OK, circuit closed)", _cool_down)

                    # ── Record intraday PnL snapshot for rolling loss tracking (Sprint 7.11) ──
                    if rm:
                        rm.record_intraday_snapshot()

                    # ── Alert on circuit breaker events ──
                    _an = getattr(app.state, "alert_notifier", None)
                    if _an and rm and rm.is_circuit_open():
                        from src.alerts.notifier import AlertSeverity
                        await _an.send(AlertSeverity.CRITICAL, "Circuit Breaker Open", f"Daily PnL: {rm.daily_pnl:.2f}, Equity: {rm.equity:.2f}", source="risk_heartbeat")

                    from .ws_manager import get_ws_manager
                    mgr = get_ws_manager()
                    if mgr:
                        _risk_update = {
                            "type": "risk_updated",
                            "equity": rm.equity,
                            "daily_pnl": rm.daily_pnl,
                            "circuit_open": rm.is_circuit_open(),
                        }
                        # Append VaR data
                        _var_r = getattr(app.state, "_last_var", None)
                        if _var_r:
                            _risk_update["var_95"] = getattr(_var_r, "var_95", 0)
                            _risk_update["var_99"] = getattr(_var_r, "var_99", 0)
                            _risk_update["portfolio_vol"] = getattr(_var_r, "portfolio_vol", 0)
                        # Append vol targeting
                        _vt2 = getattr(app.state, "vol_targeter", None)
                        if _vt2:
                            _risk_update["vol_scale_factor"] = _vt2.get_scale_factor()
                            _risk_update["realized_vol"] = _vt2.state.realized_vol_annual
                        # Append tail risk
                        _trp2 = getattr(app.state, "tail_risk_protector", None)
                        if _trp2 and hasattr(_trp2, 'state'):
                            _risk_update["vix_level"] = getattr(getattr(_trp2.state, 'vix_level', None), 'value', 'UNKNOWN')
                            _risk_update["tail_risk_exposure_scale"] = getattr(_trp2.state, 'exposure_scale', 1.0)
                        await mgr.broadcast(_risk_update)

                        if rm.is_circuit_open():
                            await mgr.broadcast({"type": "circuit_open"})
                        ks = getattr(app.state, "kill_switch", None)
                        if ks:
                            state = await ks.get_state()
                            if state.armed:
                                await mgr.broadcast({"type": "kill_switch_armed", "reason": str(state.reason) if state.reason else "", "detail": state.detail or ""})
                        # Market data feed status
                        _mds = getattr(app.state, "market_data_service", None)
                        if _mds:
                            st = _mds.get_status()
                            await mgr.broadcast({
                                "type": "market_status_updated",
                                "connected": st.get("connected", False),
                                "healthy": st.get("healthy", False),
                                "last_tick_ts": st.get("last_tick_ts"),
                            })
                        elif getattr(app.state, "yf_feeder", None):
                            _yff = app.state.yf_feeder
                            _yf_healthy = getattr(_yff, "_running", False)
                            await mgr.broadcast({
                                "type": "market_status_updated",
                                "connected": _yf_healthy,
                                "healthy": _yf_healthy,
                                "last_tick_ts": None,
                            })
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception("Risk snapshot save failed: %s", e)

        _snapshot_task = asyncio.create_task(_periodic_risk_snapshot())

    # ── Gap Risk pre-market check (Sprint 7.3) ──
    _gap_risk_task = None
    _grm = getattr(app.state, "gap_risk_manager", None)
    if _grm is not None:
        async def _pre_market_gap_check():
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            _IST_gap = _tz(_td(hours=5, minutes=30))
            _last_gap_date = None
            while True:
                await asyncio.sleep(60)
                try:
                    now_ist = _dt.now(_IST_gap)
                    today = now_ist.strftime("%Y-%m-%d")
                    # Run at 09:10 IST (5 min before market open) on weekdays only once
                    if now_ist.hour != 9 or now_ist.minute < 10 or now_ist.minute > 15:
                        continue
                    if now_ist.weekday() >= 5:
                        continue
                    if _last_gap_date == today:
                        continue
                    _last_gap_date = today

                    grm = getattr(app.state, "gap_risk_manager", None)
                    oes = getattr(app.state, "order_entry_service", None)
                    rm = getattr(app.state, "risk_manager", None)
                    if grm and oes and rm and rm.positions:
                        result = await grm.execute_gap_action(
                            submit_order_fn=oes.submit_order,
                            positions=rm.positions,
                        )
                        if result:
                            logger.info("Gap risk pre-market check completed: %s", result)
                    else:
                        logger.debug("Gap risk check skipped (no positions or services unavailable)")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Gap risk check error: %s", e)

        _gap_risk_task = asyncio.create_task(_pre_market_gap_check())
        app.state._gap_risk_task = _gap_risk_task
        logger.info("Gap risk pre-market check scheduled (09:10 IST)")

    # ── Reconciliation periodic task (Sprint 7.5) ──
    _reconciliation_task = None
    _gateway = getattr(app.state, "gateway", None)
    _rm_recon = getattr(app.state, "risk_manager", None)
    if _gateway is not None and _rm_recon is not None:
        try:
            from src.execution.reconciliation.reconciler import ReconciliationJob

            async def _fetch_broker_positions():
                gw = getattr(app.state, "gateway", None)
                if gw and hasattr(gw, "get_positions"):
                    return await gw.get_positions()
                return []

            def _get_local_positions():
                rm = getattr(app.state, "risk_manager", None)
                return list(rm.positions) if rm else []

            _recon_job = ReconciliationJob(
                fetch_broker_positions=_fetch_broker_positions,
                get_local_positions=_get_local_positions,
                trigger_freeze_on_mismatch=True,
                auto_heal_threshold_pct=5.0,
            )
            app.state.reconciliation_job = _recon_job

            async def _periodic_reconciliation():
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    try:
                        result = await _recon_job.run()
                        if not result.in_sync:
                            logger.warning("Reconciliation mismatch: %s", result.mismatches)
                            if result.auto_healed:
                                logger.info("Reconciliation auto-healed: %s", result.auto_healed)
                        else:
                            logger.debug("Reconciliation: in sync (%d positions)", len(result.local_positions))
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error("Reconciliation failed: %s", e)

            _reconciliation_task = asyncio.create_task(_periodic_reconciliation())
            app.state._reconciliation_task = _reconciliation_task
            logger.info("Reconciliation periodic task started (every 5 minutes)")
        except Exception as e:
            logger.debug("Reconciliation not configured: %s", e)

    # Bar cache for autonomous loop (market data layer feeds this)
    from src.market_data.bar_cache import BarCache
    from src.market_data.bar_aggregator import TickToBarAggregator
    app.state.bar_cache = BarCache()
    app.state.bar_aggregator = TickToBarAggregator(app.state.bar_cache, interval_seconds=60)
    app.state.market_data_service = None
    app.state.angel_one_marketdata_enabled = False
    try:
        from src.core.config import get_settings
        _settings = get_settings()
        _exec = getattr(_settings, "execution", None)
        _md = getattr(_settings, "market_data", None)
        _feed_cfg = getattr(_settings, "angel_one_feed", None)
        angel_one_marketdata_enabled = getattr(_feed_cfg, "marketdata_enabled", False) if _feed_cfg else False
        app.state.angel_one_marketdata_enabled = bool(angel_one_marketdata_enabled)
        _symbols = (
            (getattr(_feed_cfg, "symbols", None) or []) if _feed_cfg and angel_one_marketdata_enabled
            else os.environ.get("MD_SYMBOLS") or (getattr(_exec, "market_data_symbols", None) if _exec else None) or ["RELIANCE", "INFY"]
        )
        if isinstance(_symbols, str):
            _symbols = [s.strip() for s in _symbols.split(",") if s.strip()]
        connector = None
        if angel_one_marketdata_enabled:
            _api_key = getattr(_exec, "angel_one_api_key", None) or os.environ.get("ANGEL_ONE_API_KEY") or ""
            _token = getattr(_exec, "angel_one_token", None) or os.environ.get("ANGEL_ONE_TOKEN") or ""
            _secret = getattr(_exec, "angel_one_api_secret", None) or os.environ.get("ANGEL_ONE_API_SECRET", "") or ""
            if _api_key and _token:
                from src.market_data.angel_one_ws_connector import AngelOneWsConnector
                _exchange = getattr(_feed_cfg, "exchange", "NSE") if _feed_cfg else "NSE"
                _backoff = getattr(_md, "marketdata_reconnect_backoff_seconds", 5) if _md else 5
                def _market_feed_unhealthy_cb():
                    app.state.safe_mode = True
                    app.state.safe_mode_since = time.time()
                    ckc = getattr(app.state, "circuit_kill_controller", None)
                    if ckc:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.get_event_loop()
                        loop.create_task(ckc.check_market_feed_and_trip(False))
                connector = AngelOneWsConnector(
                    api_key=_api_key,
                    api_secret=_secret,
                    token=_token,
                    exchange=_exchange,
                    on_feed_unhealthy=_market_feed_unhealthy_cb,
                )
        if connector is None and not angel_one_marketdata_enabled:
            _api_key = getattr(_exec, "angel_one_api_key", None) or os.environ.get("ANGEL_ONE_API_KEY") or ""
            _token = getattr(_exec, "angel_one_token", None) or os.environ.get("ANGEL_ONE_TOKEN") or ""
            if _api_key and _token:
                from src.market_data.connectors.angel_one import AngelOneConnector
                connector = AngelOneConnector(api_key=_api_key, api_secret=os.environ.get("ANGEL_ONE_API_SECRET", ""), token=_token)
        if connector is not None:
            from src.market_data.market_data_service import MarketDataService
            def _market_feed_unhealthy():
                app.state.safe_mode = True
                app.state.safe_mode_since = time.time()
                ckc = getattr(app.state, "circuit_kill_controller", None)
                if ckc:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.get_event_loop()
                    loop.create_task(ckc.check_market_feed_and_trip(False))
            _reconnect_delay = getattr(_md, "marketdata_reconnect_backoff_seconds", 5) if _md else 5
            app.state.market_data_service = MarketDataService(
                connector, app.state.bar_cache, app.state.bar_aggregator, _symbols,
                on_feed_unhealthy=_market_feed_unhealthy,
            )
            app.state.market_data_service.start()
            logger.info("MarketDataService started for symbols=%s (angel_one_feed=%s)", _symbols, angel_one_marketdata_enabled)
    except Exception as e:
        logger.debug("MarketDataService not started: %s", e)

# YFinance fallback feeder: populate bar cache when no Angel One WebSocket
    if app.state.market_data_service is None:
        try:
            from src.market_data.yfinance_fallback_feeder import YFinanceFallbackFeeder
            _yf_feeder = YFinanceFallbackFeeder(
                bar_cache=app.state.bar_cache,
                poll_interval_seconds=60.0,
            )
            _yf_feeder.start()
            app.state.yf_feeder = _yf_feeder
            logger.info("YFinance fallback feeder started (no Angel One keys configured)")
        except Exception as e:
            logger.debug("YFinance fallback feeder not started: %s", e)

# Autonomous loop: bars → strategy runner → allocator → OrderEntryService only
    _autonomous_loop = None
    if getattr(app.state, "order_entry_service", None) is not None:
        try:
            from src.execution.autonomous_loop import AutonomousLoop
            from src.strategy_engine.runner import StrategyRunner
            from src.strategy_engine.allocator import PortfolioAllocator, AllocatorConfig
            from src.api.routers.strategies import get_registry
            from src.core.events import Exchange
            from src.ai.feature_engine import FeatureEngine
            from src.ai.regime.classifier import RegimeClassifier
            from src.ai.alpha_model import AlphaModel, AlphaStrategy

            # Load trained XGBoost model if available
            _model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "alpha_xgb.joblib")
            _alpha_model = AlphaModel(strategy_id="ai_alpha")
            if os.path.exists(_model_path):
                if _alpha_model.load(_model_path):
                    logger.info("Loaded trained AI model from %s", _model_path)
                else:
                    logger.warning("Failed to load AI model from %s — using fallback heuristic", _model_path)
            else:
                logger.info("No trained AI model found at %s — using fallback heuristic. Run: PYTHONPATH=. python scripts/train_alpha_model.py", _model_path)

            def _get_safe_mode() -> bool:
                return bool(getattr(app.state, "safe_mode", True))

            bar_cache = app.state.bar_cache
            registry = get_registry()

            # ── Register ALL strategies (multi-strategy ensemble) ──
            # 1. AI Alpha (XGBoost + market context)
            try:
                registry.register(AlphaStrategy(alpha_model=_alpha_model))
                logger.info("Strategy registered: ai_alpha (XGBoost)")
            except Exception:
                pass

            # 2. Classical technical strategies
            try:
                from src.strategy_engine.classical import (
                    EMACrossoverStrategy,
                    MACDStrategy,
                    RSIStrategy,
                )
                registry.register(EMACrossoverStrategy(fast=9, slow=21))
                registry.register(MACDStrategy(fast=12, slow=26, signal=9))
                registry.register(RSIStrategy(period=14, oversold=30.0, overbought=70.0))
                logger.info("Strategies registered: ema_crossover, macd, rsi")
            except Exception as e:
                logger.warning("Classical strategies not registered: %s", e)

            # 3. Momentum Breakout (for trending markets)
            try:
                from src.strategy_engine.momentum_breakout import MomentumBreakoutStrategy
                registry.register(MomentumBreakoutStrategy())
                logger.info("Strategy registered: momentum_breakout")
            except Exception as e:
                logger.debug("MomentumBreakout not available: %s", e)

            # 4. Mean Reversion (for sideways markets)
            try:
                from src.strategy_engine.mean_reversion import MeanReversionStrategy
                registry.register(MeanReversionStrategy())
                logger.info("Strategy registered: mean_reversion")
            except Exception as e:
                logger.debug("MeanReversion not available: %s", e)

            logger.info("Total strategies registered: %d → %s",
                        len(registry.list_all()), registry.list_all())

            # ── Wire strategy registry disable into existing PerformanceTracker (Sprint 10.8: single instance) ──
            _performance_tracker = getattr(app.state, "performance_tracker", None)
            if _performance_tracker is not None:
                try:
                    # Wire registry.disable into the existing tracker's strategy disabled callback
                    _orig_disabled_cb = getattr(_performance_tracker, '_on_strategy_disabled', None)

                    def _enhanced_strategy_disabled(strategy_id: str, reason: str):
                        logger.warning("PerformanceTracker AUTO-DISABLED strategy '%s' (reason=%s)", strategy_id, reason)
                        registry.disable(strategy_id)
                        # Also fire the original WS broadcast callback
                        if _orig_disabled_cb:
                            try:
                                _orig_disabled_cb(strategy_id, reason)
                            except Exception:
                                pass

                    _performance_tracker._on_strategy_disabled = _enhanced_strategy_disabled
                    # Update config thresholds
                    _performance_tracker.max_consecutive_losses_disable = 5
                    _performance_tracker.min_win_rate_disable = 0.35
                    _performance_tracker.max_drawdown_pct_disable = 15.0
                    logger.info("PerformanceTracker enhanced with strategy registry disable (single instance)")
                except Exception as e:
                    logger.warning("PerformanceTracker registry wiring failed: %s", e)

            strategy_runner = StrategyRunner(registry)
            feature_engine = FeatureEngine()
            regime_classifier = RegimeClassifier()

            # Enhanced allocator: more concurrent signals, strategy-level caps
            allocator = PortfolioAllocator(AllocatorConfig(
                max_active_signals=10,              # allow 10 concurrent positions
                max_capital_pct_per_signal=8.0,     # 8% per position
                min_confidence=0.2,
                strategy_cap_pct={
                    "ai_alpha": 15.0,               # AI gets higher allocation
                    "ema_crossover": 8.0,
                    "macd": 8.0,
                    "rsi": 8.0,
                    "momentum_breakout": 10.0,
                    "mean_reversion": 10.0,
                },
            ))

            def _get_market_feed_healthy() -> bool:
                svc = getattr(app.state, "market_data_service", None)
                if svc is not None:
                    return svc.is_healthy()
                # No market data service — check bar cache freshness instead
                bc = getattr(app.state, "bar_cache", None)
                if bc is not None:
                    last_ts = getattr(bc, "last_bar_timestamp", lambda: None)()
                    if last_ts and (time.time() - last_ts) > 120:
                        return False  # No bars received in last 2 minutes
                return True  # No cache = paper mode, OK

            def _get_bar_ts() -> str:
                ts = bar_cache.get_current_bar_ts()
                if ts:
                    return ts
                from datetime import datetime, timezone
                return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            def _get_bars(symbol: str, exchange: Exchange, interval: str, n: int):
                return bar_cache.get_bars(symbol, exchange, interval, n)

            def _get_symbols():
                syms = bar_cache.symbols_with_bars(Exchange.NSE, "1m", min_bars=20)
                return [(s, Exchange.NSE) for s in syms]

            def _get_risk_state():
                rm = getattr(app.state, "risk_manager", None)
                if rm is None:
                    return {"equity": 0, "exposure_multiplier": 1.0, "max_position_pct": 5.0}
                cb = getattr(app.state, "circuit_breaker", None)
                peak = getattr(cb, "_peak_equity", rm.equity) if cb else rm.equity
                dd = (peak - rm.equity) / peak * 100 if peak > 0 else 0
                drawdown_scale = 0.5 if dd > 5 else (0.75 if dd > 3 else 1.0)

                # Regime-aware scaling: reduce exposure in crisis/high-vol
                regime_scale = 1.0
                try:
                    import numpy as np
                    recent_returns = np.array([0.0])
                    vol = getattr(rm, '_current_vol', 0.01)
                    regime_result = regime_classifier.classify(recent_returns, vol, 0.0)
                    regime_label = regime_result.label.value
                    if regime_label == "crisis":
                        regime_scale = 0.0
                    elif regime_label == "high_volatility":
                        regime_scale = 0.5
                except Exception:
                    pass

                # Vol targeting: scale positions by target_vol / realized_vol
                vol_scale = 1.0
                _vt = getattr(app.state, "vol_targeter", None)
                if _vt:
                    vol_scale = _vt.get_scale_factor()

                # Tail risk: reduce exposure based on VIX level
                tail_risk_scale = 1.0
                _trp = getattr(app.state, "tail_risk_protector", None)
                if _trp and hasattr(_trp, 'state'):
                    tail_risk_scale = getattr(_trp.state, 'exposure_scale', 1.0)

                # Composite exposure multiplier
                base_mult = getattr(rm, "_exposure_multiplier", 1.0)
                composite_mult = base_mult * vol_scale * tail_risk_scale

                return {
                    "equity": rm.equity,
                    "exposure_multiplier": composite_mult,
                    "max_position_pct": rm.limits.max_position_pct,
                    "drawdown_scale": drawdown_scale,
                    "regime_scale": regime_scale,
                    "vol_scale": vol_scale,
                    "tail_risk_scale": tail_risk_scale,
                }

            def _get_positions():
                rm = getattr(app.state, "risk_manager", None)
                return list(rm.positions) if rm else []

            async def _submit_order_async(req):
                return await app.state.order_entry_service.submit_order(req)

            # Full-market scanner: scores all NSE stocks via AlphaModel
            _market_scanner = None
            try:
                from src.scanner.market_scanner import MarketScanner
                _market_scanner = MarketScanner(
                    alpha_model=_alpha_model,
                    feature_engine=feature_engine,
                    top_n=20,
                    min_confidence=0.55,
                )
                logger.info("MarketScanner configured (full NSE universe)")
            except Exception as e:
                logger.warning("MarketScanner not configured: %s", e)

            # ── AI Ensemble Engine: LSTM + Transformer + RL + Sentiment + XGBoost ──
            _ensemble_engine = None
            _rl_predictor = None
            try:
                from src.ai.models.registry import ModelRegistry
                from src.ai.models.ensemble import EnsembleEngine
                from src.ai.models.lstm_predictor import LSTMPredictor
                from src.ai.models.transformer_predictor import TransformerPredictor
                from src.ai.models.rl_agent import RLPredictor
                from src.ai.models.sentiment_predictor import SentimentPredictor

                _models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
                model_registry = ModelRegistry()

                # LSTM
                _lstm_path = os.path.join(_models_dir, "lstm_predictor.pt")
                lstm_pred = LSTMPredictor(model_path=_lstm_path)
                model_registry.register(lstm_pred)
                logger.info("LSTM predictor registered (loaded=%s)", lstm_pred._loaded)

                # Transformer
                _tf_path = os.path.join(_models_dir, "transformer_predictor.pt")
                tf_pred = TransformerPredictor(model_path=_tf_path)
                model_registry.register(tf_pred)
                logger.info("Transformer predictor registered (loaded=%s)", tf_pred._loaded)

                # RL Agent
                _rl_path = os.path.join(_models_dir, "rl_agent.zip")
                _rl_predictor = RLPredictor(model_path=_rl_path)
                model_registry.register(_rl_predictor)
                logger.info("RL predictor registered (loaded=%s)", _rl_predictor._loaded)

                # Sentiment
                sentiment_pred = SentimentPredictor()
                model_registry.register(sentiment_pred)
                logger.info("Sentiment predictor registered")

                # XGBoost (via existing alpha model wrapper)
                try:
                    from src.ai.models.base import BasePredictor, PredictionOutput
                    class XGBoostPredictor(BasePredictor):
                        model_id = "xgboost_alpha"
                        version = "v1"
                        def __init__(self_xgb, alpha_model):
                            self_xgb._alpha = alpha_model
                            self_xgb.path = ""
                        def predict(self_xgb, features, context=None):
                            try:
                                score = self_xgb._alpha.score(features)
                                prob_up = 0.5 + score * 0.3
                                return PredictionOutput(
                                    prob_up=min(0.95, max(0.05, prob_up)),
                                    expected_return=score * 0.01,
                                    confidence=min(1.0, abs(score)),
                                    model_id="xgboost_alpha", version="v1",
                                    metadata={"raw_score": score},
                                )
                            except Exception:
                                return PredictionOutput(0.5, 0.0, 0.0, "xgboost_alpha", "v1", {})
                    if _alpha_model:
                        xgb_pred = XGBoostPredictor(_alpha_model)
                        model_registry.register(xgb_pred)
                        logger.info("XGBoost predictor registered")
                except Exception as e:
                    logger.debug("XGBoost predictor wrapper failed: %s", e)

                # RL agent: set weight=0 if model not loaded (Sprint 8.10)
                _rl_weight = 0.15
                if _rl_predictor and not _rl_predictor._loaded:
                    _rl_weight = 0.0
                    logger.warning("RL model not loaded — setting ensemble weight to 0")

                _ensemble_engine = EnsembleEngine(
                    registry=model_registry,
                    model_ids=["xgboost_alpha", "lstm_ts", "transformer_ts", "rl_ppo", "sentiment_finbert"],
                    weights={
                        "xgboost_alpha": 0.30,
                        "lstm_ts": 0.25,
                        "transformer_ts": 0.20,
                        "rl_ppo": _rl_weight,
                        "sentiment_finbert": 0.25 if _rl_weight == 0 else 0.10,
                    },
                )
                app.state.ensemble_engine = _ensemble_engine
                app.state.model_registry = model_registry
                logger.info("AI Ensemble Engine configured (5 models, rl_weight=%.2f)", _rl_weight)
            except Exception as e:
                logger.warning("AI Ensemble Engine not configured: %s", e)

            # ── Register ML strategies (connect to real models) ──
            try:
                from src.strategy_engine.ml_strategies import MLPredictorStrategy, RLAgentStrategy
                ml_strategy = MLPredictorStrategy(
                    ensemble_engine=_ensemble_engine,
                    feature_engine=feature_engine,
                    confidence_threshold=0.55,
                    prob_threshold=0.58,
                )
                registry.register(ml_strategy)
                logger.info("Strategy registered: ml_predictor (ensemble AI)")

                rl_strategy = RLAgentStrategy(
                    rl_predictor=_rl_predictor,
                    feature_engine=feature_engine,
                    confidence_threshold=0.5,
                )
                registry.register(rl_strategy)
                logger.info("Strategy registered: rl_agent (PPO)")
            except Exception as e:
                logger.warning("ML strategies not registered: %s", e)

            async def _ws_broadcast(message):
                from .ws_manager import get_ws_manager
                mgr = get_ws_manager()
                if mgr:
                    await mgr.broadcast(message)

            def _on_daily_reset():
                rm = getattr(app.state, "risk_manager", None)
                if rm and hasattr(rm, "reset_daily_pnl"):
                    rm.reset_daily_pnl()

            # GAP-5 fix: wire drift_gate and regime_gate into autonomous loop
            _drift_detector = getattr(app.state, "drift_detector", None)
            def _drift_gate():
                if _drift_detector is None:
                    return True
                try:
                    return not getattr(_drift_detector, "is_drifted", lambda: False)()
                except Exception:
                    return True  # Fail open on error

            _regime_cls = regime_classifier
            def _regime_gate():
                if _regime_cls is None:
                    return True
                try:
                    regime = getattr(_regime_cls, "current_regime", None)
                    if regime is not None and hasattr(regime, "value"):
                        return regime.value != "CRISIS"
                    return True
                except Exception:
                    return True

            _autonomous_loop = AutonomousLoop(
                _submit_order_async,
                get_safe_mode=_get_safe_mode,
                get_bar_ts=_get_bar_ts,
                get_bars=_get_bars,
                get_symbols=_get_symbols,
                strategy_runner=strategy_runner,
                allocator=allocator,
                get_risk_state=_get_risk_state,
                get_positions=_get_positions,
                get_market_feed_healthy=_get_market_feed_healthy,
                feature_engine=feature_engine,
                regime_classifier=regime_classifier,
                market_scanner=_market_scanner,
                performance_tracker=_performance_tracker,
                ws_broadcast=_ws_broadcast,
                on_daily_reset=_on_daily_reset,
                drift_gate=_drift_gate,
                regime_gate=_regime_gate,
                poll_interval_seconds=60.0,
                paper_mode=getattr(gateway, "paper", True),
            )
            # Wire open trade persistence (write-ahead for SL/TP tracking)
            try:
                from src.persistence.open_trade_repo import OpenTradeRepository
                _open_trade_repo = OpenTradeRepository()
                _autonomous_loop.set_open_trade_repo(_open_trade_repo)
                recovered = _autonomous_loop.load_open_trades_from_db()
                logger.info("Open trade persistence wired (recovered %d trades from DB)", recovered)
            except Exception as e:
                logger.warning("Open trade persistence not available: %s", e)

            _autonomous_loop.start()
            app.state.autonomous_loop = _autonomous_loop
            logger.info("Autonomous loop started (multi-strategy→regime→scanner→allocator→risk→execution)")

            # ── Wire kill switch into autonomous loop (Sprint 7.1) ──
            _ks_for_loop = getattr(app.state, "kill_switch", None)
            if _ks_for_loop is not None:
                async def _kill_switch_check():
                    ks = getattr(app.state, "kill_switch", None)
                    if ks is None:
                        return False
                    return ks.is_armed()
                _autonomous_loop.set_kill_switch(_kill_switch_check)
                logger.info("Kill switch wired to autonomous loop (auto-close on arm)")

            # ── Wire risk_manager for forced close symbols (Sprint 7.9) ──
            _autonomous_loop._risk_manager = getattr(app.state, "risk_manager", None)

            # ── Wire ensemble_engine for prediction metadata (Sprint 8.3) ──
            _autonomous_loop._ensemble_engine = getattr(app.state, "ensemble_engine", None)

            # ── Wire news sentiment into autonomous loop (Phase 1: news-aware trading) ──
            try:
                from src.ai.llm.client import LLMClient, LLMConfig
                from src.ai.llm.sentiment import NewsSentimentService
                _openai_key = os.environ.get("OPENAI_API_KEY", "")
                _anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if _openai_key:
                    _llm_config = LLMConfig(provider="openai", api_key=_openai_key, model="gpt-4o-mini")
                elif _anthropic_key:
                    _llm_config = LLMConfig(provider="anthropic", api_key=_anthropic_key, model="claude-3-haiku-20240307")
                else:
                    _llm_config = None
                if _llm_config:
                    _llm_client = LLMClient(_llm_config)
                    _sentiment_service = NewsSentimentService(_llm_client)
                    _autonomous_loop.set_sentiment_service(_sentiment_service)
                    logger.info("News sentiment wired to autonomous loop (provider=%s)", _llm_config.provider)
                else:
                    logger.info("No LLM API key set — sentiment integration skipped (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
            except Exception as e:
                logger.warning("Sentiment service not wired: %s", e)

# ── Multi-Agent Orchestrator ──
            try:
                from src.agents.base import AgentOrchestrator
                from src.agents.research_agent import ResearchAgent
                from src.agents.risk_agent import RiskMonitorAgent
                from src.agents.execution_agent import ExecutionAgent
                from src.agents.strategy_selector import StrategySelectorAgent

                orchestrator = AgentOrchestrator()

                # Pass MarketScanner for full-market discovery beyond BarCache
                _market_scanner_for_agent = None
                try:
                    from src.scanner.market_scanner import MarketScanner
                    _market_scanner_for_agent = MarketScanner(
                        alpha_model=_alpha_model,
                        feature_engine=feature_engine,
                        top_n=20,
                        min_confidence=0.55,
                    )
                except Exception:
                    pass

                research_agent = ResearchAgent(
                    get_symbols=_get_symbols,
                    get_bars=_get_bars,
                    feature_engine=feature_engine,
                    ensemble_engine=_ensemble_engine,
                    market_scanner=_market_scanner_for_agent,
                    top_n=10,
                    min_confidence=0.55,
                    scan_interval=300.0,
                )
                orchestrator.register(research_agent)

                risk_agent = RiskMonitorAgent(
                    risk_manager=risk_manager,
                    circuit_breaker=circuit_breaker,
                    kill_switch=kill_switch,
                    get_positions=_get_positions,
                    get_bars=_get_bars,
                    max_portfolio_drawdown_pct=5.0,
                    monitor_interval=30.0,
                )
                orchestrator.register(risk_agent)

                execution_agent = ExecutionAgent(
                    submit_order_fn=_submit_order_async,
                    get_bars=_get_bars,
                    get_positions=_get_positions,
                    max_concurrent_orders=5,
                    execution_interval=15.0,
                )
                orchestrator.register(execution_agent)

                strategy_selector = StrategySelectorAgent(
                    strategy_registry=registry,
                    regime_classifier=regime_classifier,
                    get_bars=_get_bars,
                    selection_interval=120.0,
                )
                orchestrator.register(strategy_selector)

                # Broadcast callback for WebSocket
                async def _agent_broadcast(msg):
                    from .ws_manager import get_ws_manager
                    mgr = get_ws_manager()
                    if mgr:
                        await mgr.broadcast({
                            "type": f"agent_{msg.msg_type}",
                            "source": msg.source,
                            "payload": msg.payload,
                            "timestamp": msg.timestamp,
                        })
                orchestrator.set_broadcast_callback(_agent_broadcast)

                orchestrator.start_all()
                app.state.agent_orchestrator = orchestrator
                logger.info("Agent Orchestrator started (research, risk, execution, strategy_selector)")
            except Exception as e:
                logger.warning("Agent Orchestrator not started: %s", e)

# ── Auto-Retrain Scheduler ──
            # Checks every 6 hours; retrains ALL AI models when stale (>7 days old)
            async def _auto_retrain_loop():
                """Check model age every 6 hours; retrain all models if >7 days old."""
                import json, subprocess, time as _time
                retrain_interval = 6 * 3600  # check every 6 hours
                model_max_age = 7 * 86400     # 7 days
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                while True:
                    await asyncio.sleep(retrain_interval)
                    try:
                        # Check training_meta.json (from auto_train_all.py) or XGBoost meta
                        meta_path = os.path.join(project_root, "models", "training_meta.json")
                        xgb_meta_path = os.path.join(project_root, "models", "alpha_xgb_meta.json")
                        should_retrain = False

                        if os.path.exists(meta_path):
                            age = _time.time() - os.path.getmtime(meta_path)
                            if age > model_max_age:
                                should_retrain = True
                                logger.info("Auto-retrain: models are %.1f days old (limit=7), triggering full retrain", age / 86400)
                        elif os.path.exists(xgb_meta_path):
                            age = _time.time() - os.path.getmtime(xgb_meta_path)
                            if age > model_max_age:
                                should_retrain = True
                                logger.info("Auto-retrain: XGBoost model is %.1f days old, triggering retrain", age / 86400)
                        else:
                            should_retrain = True
                            logger.info("Auto-retrain: no model metadata found, triggering training")

                        if should_retrain:
                            logger.info("Auto-retrain starting (all AI models)...")
                            env = os.environ.copy()
                            env["PYTHONPATH"] = project_root
                            # Use auto_train_all.py with --quick for scheduled retrains
                            train_script = os.path.join(project_root, "scripts", "auto_train_all.py")
                            if not os.path.exists(train_script):
                                train_script = os.path.join(project_root, "scripts", "train_alpha_model.py")
                            result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: subprocess.run(
                                    [sys.executable, train_script, "--quick"],
                                    env=env, capture_output=True, text=True, timeout=1800,
                                    cwd=project_root,
                                )
                            )
                            if result.returncode == 0:
                                logger.info("Auto-retrain completed successfully")
                                # Hot-reload ALL models (Sprint 8.9)
                                if _alpha_model.load(_model_path):
                                    logger.info("XGBoost model hot-reloaded")
                                _mr = getattr(app.state, "model_registry", None)
                                if _mr:
                                    for _mid, _suffix in [("lstm_ts", "lstm_predictor.pt"), ("transformer_ts", "transformer_predictor.pt"), ("rl_ppo", "rl_agent.zip")]:
                                        _mp = os.path.join(_models_dir, _suffix)
                                        if os.path.exists(_mp):
                                            try:
                                                _pred = _mr.get(_mid)
                                                if _pred and hasattr(_pred, 'load') and hasattr(_pred, '_loaded'):
                                                    if _pred.load(_mp):
                                                        _pred._loaded = True
                                                        logger.info("Hot-reloaded model: %s from %s", _mid, _suffix)
                                            except Exception as _re:
                                                logger.warning("Hot-reload failed for %s: %s", _mid, _re)
                                # Broadcast retrain event
                                try:
                                    ws_mgr = getattr(app.state, "ws_manager", None)
                                    if ws_mgr:
                                        await ws_mgr.broadcast({"type": "models_retrained", "status": "success"})
                                except Exception:
                                    pass
                            else:
                                logger.warning("Auto-retrain failed: %s", result.stderr[-500:] if result.stderr else "unknown error")
                    except Exception as e:
                        logger.exception("Auto-retrain error: %s", e)

            _retrain_task = asyncio.get_event_loop().create_task(_auto_retrain_loop())
            app.state._retrain_task = _retrain_task
            logger.info("Auto-retrain scheduler active (checks every 6h, retrains all AI models if >7 days old)")

# ── Drift Detection Scheduler (daily after market close) ──
            try:
                from src.ai.drift.multi_drift import MultiLayerDriftDetector
                _drift_detector = MultiLayerDriftDetector()
                app.state.drift_detector = _drift_detector
                logger.info("MultiLayerDriftDetector initialized")

                async def _daily_drift_check():
                    """Run drift detection daily at ~15:45 IST (after market close)."""
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    _IST = _tz(_td(hours=5, minutes=30))
                    while True:
                        await asyncio.sleep(300)  # check every 5 min
                        try:
                            now_ist = _dt.now(_IST)
                            # Only run at 15:45
                            if now_ist.hour != 15 or now_ist.minute < 45 or now_ist.minute > 50:
                                continue
                            if now_ist.weekday() >= 5:
                                continue

                            dd = getattr(app.state, "drift_detector", None)
                            if dd is None:
                                continue

                            signals = dd.check_all()
                            drifted_layers = [s for s in signals if s.drifted]
                            if len(drifted_layers) >= 2:
                                logger.warning("Model drift detected: %d layers flagged — %s",
                                               len(drifted_layers), [(s.drift_type.value, s.value) for s in drifted_layers])
                                _an = getattr(app.state, "alert_notifier", None)
                                if _an:
                                    from src.alerts.notifier import AlertSeverity
                                    await _an.send(AlertSeverity.CRITICAL, "Model Drift Detected",
                                                   f"{len(drifted_layers)} drift layers flagged: {[s.drift_type.value for s in drifted_layers]}",
                                                   source="drift_detector")
                                from .ws_manager import get_ws_manager
                                mgr = get_ws_manager()
                                if mgr:
                                    await mgr.broadcast({
                                        "type": "model_drift_detected",
                                        "layers": {s.drift_type.value: {"drifted": s.drifted, "value": s.value, "threshold": s.threshold} for s in signals},
                                    })
                            elif drifted_layers:
                                logger.info("Drift check: 1 layer flagged (warning only) — %s", drifted_layers[0].drift_type.value)
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.debug("Drift check error: %s", e)

                _drift_task = asyncio.get_event_loop().create_task(_daily_drift_check())
                app.state._drift_task = _drift_task
                logger.info("Drift detection scheduler active (daily check at 15:45 IST)")
            except Exception as e:
                logger.debug("Drift detection not configured: %s", e)

# ── Self-Learning Scheduler (Sprint 4.1 + 4.5) ──
            try:
                from src.ai.self_learning.scheduler import SelfLearningScheduler
                from src.ai.self_learning.drift import ConceptDriftDetector, DataDistributionMonitor

                _sl_drift = ConceptDriftDetector(threshold=0.3)
                _sl_dist_monitor = DataDistributionMonitor(window=100)

                def _sl_retrain_fn():
                    """Retrain all models via existing auto_train_all script."""
                    import subprocess, sys as _sys
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    env = os.environ.copy()
                    env["PYTHONPATH"] = project_root
                    train_script = os.path.join(project_root, "scripts", "auto_train_all.py")
                    if not os.path.exists(train_script):
                        train_script = os.path.join(project_root, "scripts", "train_alpha_model.py")
                    result = subprocess.run(
                        [_sys.executable, train_script, "--quick"],
                        env=env, capture_output=True, text=True, timeout=1800,
                        cwd=project_root,
                    )
                    if result.returncode == 0:
                        # Hot-reload ALL models (Sprint 8.9)
                        if _alpha_model and os.path.exists(_model_path):
                            _alpha_model.load(_model_path)
                        _mr = getattr(app.state, "model_registry", None)
                        if _mr:
                            for _mid, _suffix in [("lstm_ts", "lstm_predictor.pt"), ("transformer_ts", "transformer_predictor.pt"), ("rl_ppo", "rl_agent.zip")]:
                                _mp = os.path.join(_models_dir, _suffix)
                                if os.path.exists(_mp):
                                    try:
                                        _pred = _mr.get(_mid)
                                        if _pred and hasattr(_pred, 'load'):
                                            _pred.load(_mp)
                                            logger.info("SL hot-reloaded model: %s", _mid)
                                    except Exception:
                                        pass
                        return {"all_models": True}
                    return {"all_models": False}

                _sl_ic_fn = None
                if _ensemble_engine:
                    _sl_ic_fn = _ensemble_engine.update_weights_from_ic

                _sl_alert_fn = None
                _an_sl = getattr(app.state, "alert_notifier", None)
                if _an_sl:
                    _sl_alert_fn = _an_sl.send

                def _get_recent_features():
                    """Gather recent feature snapshots from bar cache."""
                    feats = []
                    try:
                        from src.core.events import Exchange
                        syms = bar_cache.symbols_with_bars(Exchange.NSE, "1m", min_bars=30)
                        for sym in syms[:5]:
                            bars_list = bar_cache.get_bars(sym, Exchange.NSE, "1m", 50)
                            if bars_list and len(bars_list) >= 20:
                                f = feature_engine.build_features(bars_list)
                                if f:
                                    feats.append(f)
                    except Exception:
                        pass
                    return feats

                _self_learning_scheduler = SelfLearningScheduler(
                    drift_detector=_sl_drift,
                    distribution_monitor=_sl_dist_monitor,
                    retrain_fn=_sl_retrain_fn,
                    ic_update_fn=_sl_ic_fn,
                    alert_fn=_sl_alert_fn,
                    get_recent_features_fn=_get_recent_features,
                    post_market_hour=15,
                    post_market_minute=45,
                    min_drift_layers_for_retrain=2,
                    weekly_revalidation_day=4,  # Friday
                )
                # Wire ensemble for calibrator fitting (Sprint 8.4)
                _self_learning_scheduler._ensemble = getattr(app.state, "ensemble_engine", None)
                _self_learning_scheduler.start()
                app.state.self_learning_scheduler = _self_learning_scheduler
                logger.info("SelfLearningScheduler started (post-market 15:45 IST, drift→retrain/IC-update)")
            except Exception as e:
                logger.warning("SelfLearningScheduler not started: %s", e)

# ── Daily Report Scheduler (16:00 IST after market close) ──
            try:
                _drg = getattr(app.state, "daily_report_generator", None)
                if _drg:
                    async def _daily_report_loop():
                        """Generate daily performance report at 16:00 IST."""
                        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                        _IST = _tz(_td(hours=5, minutes=30))
                        _last_report_date = None
                        while True:
                            await asyncio.sleep(300)  # check every 5 min
                            try:
                                now_ist = _dt.now(_IST)
                                today = now_ist.strftime("%Y-%m-%d")
                                if now_ist.hour != 16 or now_ist.minute > 5:
                                    continue
                                if now_ist.weekday() >= 5:
                                    continue
                                if _last_report_date == today:
                                    continue

                                _last_report_date = today
                                drg = getattr(app.state, "daily_report_generator", None)
                                if drg:
                                    report = drg.generate(date=today)
                                    try:
                                        drg.save_to_db(report)
                                    except Exception:
                                        pass
                                    logger.info("Daily report generated: %s trades, net_pnl=%.2f, sharpe=%.2f",
                                                report.total_trades, report.net_pnl, report.sharpe_ratio_20d)
                                    from .ws_manager import get_ws_manager
                                    mgr = get_ws_manager()
                                    if mgr:
                                        from src.reporting.daily_report import DailyReportGenerator
                                        await mgr.broadcast({"type": "daily_report", "report": DailyReportGenerator.to_dict(report)})
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                logger.debug("Daily report error: %s", e)

                    _report_task = asyncio.get_event_loop().create_task(_daily_report_loop())
                    app.state._report_task = _report_task
                    logger.info("Daily report scheduler active (generates at 16:00 IST)")
            except Exception as e:
                logger.debug("Daily report scheduler not configured: %s", e)

        except Exception as e:
            logger.warning("Autonomous loop not started: %s", e)

# Capital deployment gate: validate() must pass before enabling autonomous live mode
    try:
        from src.api.capital_gate import CapitalGate
        async def _check_redis():
            try:
                from src.core.config import get_settings
                url = get_settings().market_data.redis_url
            except Exception:
                url = "redis://localhost:6379/0"
            try:
                import redis.asyncio as redis
                r = redis.from_url(url, decode_responses=True)
                await r.ping()
                await r.aclose()
                return True
            except Exception:
                return False
        def _check_broker():
            oes = getattr(app.state, "order_entry_service", None)
            if not oes or not getattr(oes, "order_router", None):
                return True
            gw = getattr(oes.order_router, "default_gateway", None)
            return gw is not None
        def _check_market_data():
            svc = getattr(app.state, "market_data_service", None)
            if svc is None:
                return True
            return svc.is_healthy()
        app.state.capital_gate = CapitalGate(
            check_redis=_check_redis,
            check_broker=_check_broker,
            check_market_data=_check_market_data,
            stress_tests_passed=False,
            restart_simulation_passed=False,
        )
    except Exception as e:
        logger.warning("Capital gate not configured: %s", e)
        app.state.capital_gate = None

    # Broker heartbeat (live only): on 3 consecutive failures set safe_mode
    _gateway = getattr(app.state, "gateway", None)
    if _gateway and not getattr(_gateway, "paper", True):
        _heartbeat_failures = [0]  # mutable so closure can update

        async def _broker_heartbeat():
            while True:
                await asyncio.sleep(60)
                try:
                    await _gateway.get_orders(limit=1)
                    _heartbeat_failures[0] = 0
                    app.state._last_broker_ok_ts = time.time()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Broker heartbeat failed: %s", e)
                    _heartbeat_failures[0] = _heartbeat_failures[0] + 1
                    if _heartbeat_failures[0] >= 3:
                        logger.error("Broker heartbeat: 3 consecutive failures; entering safe_mode")
                        _set_safe_mode_cb()
                        _heartbeat_failures[0] = 0

        _heartbeat_task = asyncio.create_task(_broker_heartbeat())

# Alpha Research pipeline (optional)
    app.state.alpha_research_pipeline = None
    try:
        from src.ai.alpha_research import (
            AlphaHypothesisGenerator,
            StatisticalValidator,
            AlphaQualityScorer,
            SignalClustering,
            CapacityModel,
            DecayMonitor,
            EdgePreservationRules,
            ResearchPipeline,
            PipelineConfig,
        )
        from src.ai.alpha_research.scoring import AlphaQualityScoreConfig
        from src.ai.alpha_research.clustering import ClusterConfig
        from src.ai.alpha_research.decay import DecayConfig
        gen = AlphaHypothesisGenerator(min_univariate_ic=0.005, max_total_candidates=80)
        val = StatisticalValidator(min_sample_size=200, min_wf_positive_cycles=2, fdr_alpha=0.10)
        sco = AlphaQualityScorer(AlphaQualityScoreConfig(top_decile=True))
        clu = SignalClustering(ClusterConfig(max_correlation=0.85))
        cap = CapacityModel(target_capital=1e6, max_participation_pct=10.0)
        dec = DecayMonitor(DecayConfig(rolling_window=20))
        rules = EdgePreservationRules(min_wf_cycles=2, max_turnover=1.0)
        app.state.alpha_research_pipeline = ResearchPipeline(
            hypothesis_generator=gen,
            validator=val,
            scorer=sco,
            clustering=clu,
            capacity_model=cap,
            decay_monitor=dec,
            preservation_rules=rules,
            config=PipelineConfig(max_candidates_per_run=50, min_sample_size=200, fdr_alpha=0.10),
        )
        logger.info("Alpha research pipeline configured")
    except Exception as e:
        logger.warning("Alpha research pipeline not configured: %s", e)
    yield
    # Shutdown: stop agent orchestrator, autonomous loop, cancel periodic tasks
    _orch = getattr(app.state, "agent_orchestrator", None)
    if _orch is not None:
        try:
            await _orch.stop_all()
        except Exception as ex:
            logger.warning("Agent orchestrator stop: %s", ex)
    _al = getattr(app.state, "autonomous_loop", None)
    if _al is not None:
        try:
            await _al.stop()
        except Exception as ex:
            logger.warning("Autonomous loop stop: %s", ex)
    if _snapshot_task is not None:
        _snapshot_task.cancel()
        try:
            await _snapshot_task
        except asyncio.CancelledError:
            pass
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
    _retrain = getattr(app.state, "_retrain_task", None)
    if _retrain is not None:
        _retrain.cancel()
        try:
            await _retrain
        except asyncio.CancelledError:
            pass
    # Cancel gap risk, reconciliation, drift detection and daily report tasks
    for _task_name in ("_gap_risk_task", "_reconciliation_task", "_drift_task", "_report_task"):
        _t = getattr(app.state, _task_name, None)
        if _t is not None:
            _t.cancel()
            try:
                await _t
            except asyncio.CancelledError:
                pass
    # Kafka producer shutdown (Sprint 10.9)
    _kafka_prod = getattr(app.state, "_kafka_producer", None)
    if _kafka_prod is not None:
        try:
            await _kafka_prod.stop()
            logger.info("Kafka producer stopped")
        except Exception as ex:
            logger.warning("Kafka producer stop: %s", ex)

    _fill_listener = getattr(app.state, "fill_listener", None)
    if _fill_listener is not None:
        await _fill_listener.stop()
    _paper_sim = getattr(app.state, "paper_fill_simulator", None)
    if _paper_sim is not None:
        await _paper_sim.stop()
    _mds = getattr(app.state, "market_data_service", None)
    if _mds is not None:
        try:
            await _mds.stop()
        except Exception as ex:
            logger.warning("MarketDataService stop: %s", ex)
    logger.info("API shutdown")


def _get_allowed_origins() -> list[str]:
    """Return CORS allowed origins from env or sensible defaults."""
    import os
    custom = os.environ.get("CORS_ORIGINS")
    if custom:
        return [o.strip() for o in custom.split(",") if o.strip()]
    env = os.environ.get("ENV", "development").lower()
    if env == "production":
        # In production, set CORS_ORIGINS explicitly
        return [
            os.environ.get("FRONTEND_URL", "https://trading.example.com"),
        ]
    # Development: allow local origins
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def _setup_structured_logging():
    """Configure structured JSON logging (Sprint 9.2)."""
    import json as _json
    import os

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "ts": self.formatTime(record),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            return _json.dumps(log_entry)

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    if os.environ.get("LOG_FORMAT", "").lower() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logging.basicConfig(level=log_level, handlers=[handler], force=True)

        # Add file handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            os.makedirs("logs", exist_ok=True)
            fh = RotatingFileHandler("logs/trading.log", maxBytes=10 * 1024 * 1024, backupCount=5)
            fh.setFormatter(JSONFormatter())
            logging.getLogger().addHandler(fh)
        except Exception:
            pass
    else:
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# Initialize structured logging before app creation
_setup_structured_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Autonomous Trading Platform API",
        version="1.0.0",
        description="Institutional-grade multi-market trading engine",
        lifespan=lifespan,
    )

    # Security middleware (order matters: outermost first)
    from .middleware import SecurityHeadersMiddleware, RateLimitMiddleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=120, auth_requests_per_minute=20)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    # ── Global exception handlers (Sprint 9.1) ──
    import uuid
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        error_id = str(uuid.uuid4())[:8]
        logger.exception("Unhandled error [%s]: %s", error_id, exc)
        return JSONResponse(status_code=500, content={
            "error": "internal_error", "error_id": error_id,
            "message": "An unexpected error occurred",
        })

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(status_code=422, content={
            "error": "validation_error",
            "details": [{"field": str(e.get("loc", ["unknown"])[-1]), "message": e.get("msg", "")} for e in exc.errors()],
        })

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root():
        """Root: welcome page with links to docs and health."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Trading Platform API</title></head>
        <body style="font-family: system-ui; max-width: 600px; margin: 2rem auto; padding: 0 1rem;">
            <h1>Autonomous Trading Platform API</h1>
            <p>Version 1.0.0</p>
            <ul>
                <li><a href="/docs">API docs (Swagger)</a></li>
                <li><a href="/redoc">ReDoc</a></li>
                <li><a href="/health">Health</a></li>
                <li><a href="/ready">Ready</a></li>
            </ul>
        </body>
        </html>
        """

    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1", tags=["Auth"])
    app.include_router(market.router, prefix="/api/v1/market", tags=["Market"])
    app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
    app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk"])
    app.include_router(orders.router, prefix="/api/v1", tags=["Orders"])
    app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtest"])
    app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])
    app.include_router(alpha_research.router, prefix="/api/v1", tags=["Alpha Research"])
    app.include_router(reconciliation.router, prefix="/api/v1", tags=["Reconciliation"])
    app.include_router(capital.router, prefix="/api/v1", tags=["Capital"])
    app.include_router(audit.router, prefix="/api/v1", tags=["Audit"])
    app.include_router(performance.router, prefix="/api/v1/performance", tags=["Performance"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])
    app.include_router(broker.router, prefix="/api/v1", tags=["Broker"])

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Accept WebSocket for live updates. Supports two auth modes:
        1. Sec-WebSocket-Protocol header with JWT (preferred, avoids token in URL)
        2. Query param ?token=<jwt> (legacy fallback)
        """
        from .ws_manager import get_ws_manager
        from .auth import _decode_token
        import os
        from urllib.parse import parse_qs

        user_id = "anonymous"

        # Try auth via subprotocol header first (preferred - avoids token in URL)
        subprotocols = websocket.headers.get("sec-websocket-protocol", "")
        token = None
        if subprotocols:
            for proto in subprotocols.split(","):
                proto = proto.strip()
                if proto.startswith("access_token."):
                    token = proto[len("access_token."):]
                    break

        # Fallback: query param (legacy)
        if not token:
            query = parse_qs(websocket.scope.get("query_string", b"").decode())
            token = query.get("token", [None])[0]

        if os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET"):
            if not token:
                await websocket.close(code=4001)
                return
            payload = _decode_token(token)
            if not payload:
                await websocket.close(code=4001)
                return
            user_id = payload.get("sub") or payload.get("user_id") or "unknown"

        mgr = get_ws_manager()
        if mgr:
            await mgr.connect(websocket, user_id=user_id)
        else:
            await websocket.accept()
        try:
            await websocket.send_json({"type": "connected", "message": "Live", "user_id": user_id})
            while True:
                try:
                    data = await websocket.receive_text()
                    # Handle ping/pong heartbeat from client
                    if data == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    break
        except Exception:
            pass
        finally:
            if mgr:
                await mgr.disconnect(websocket)
            try:
                await websocket.close()
            except Exception:
                pass

    app.get("/metrics", include_in_schema=False)(_prometheus_metrics)

    @app.get("/api/v1/debug/bar-cache", include_in_schema=False)
    def debug_bar_cache():
        bc = getattr(app.state, "bar_cache", None)
        if not bc:
            return {"error": "no bar_cache"}
        keys = list(bc._bars.keys())
        counts = {k: len(bc._bars[k]) for k in keys[:30]}
        from src.core.events import Exchange
        symbols = bc.symbols_with_bars(Exchange.NSE, "1m", min_bars=20)
        return {"total_keys": len(keys), "sample_counts": counts, "symbols_with_20_bars": symbols}

    return app


app = create_app()

"""
Lifespan: OrderEntryService, OrderRouter, gateways, fill handler, circuit breaker,
kill switch, periodic risk snapshot, gap risk pre-market check, reconciliation,
broker heartbeat, capital gate.
"""

import asyncio
import logging
import os
import time

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Module-level references for background tasks (needed for shutdown cancellation)
_snapshot_task = None
_heartbeat_task = None


async def init_execution(app: FastAPI) -> None:
    """Initialize execution layer: RiskManager, OrderEntryService, FillHandler,
    circuit breaker, kill switch, periodic tasks, reconciliation, capital gate."""
    global _snapshot_task, _heartbeat_task

    _snapshot_task = None
    _heartbeat_task = None

    safe_mode = False
    persistence_service = getattr(app.state, "persistence_service", None)
    order_repo = getattr(app.state, "order_repo", None)
    position_repo = getattr(app.state, "position_repo", None)

    try:
        from src.core.config import get_settings
        from src.core.events import OrderStatus
        from src.execution import AngelOneExecutionGateway, OrderLifecycle, OrderRouter
        from src.execution.order_entry import ExposureReservation, IdempotencyStore, KillSwitch, OrderEntryService
        from src.execution.order_entry.redis_cluster_reservation import RedisClusterReservation
        from src.execution.order_entry.redis_distributed_lock import RedisDistributedLock
        from src.risk_engine import RiskManager
        from src.risk_engine.circuit_breaker import CircuitBreaker
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
        # Single paper flag from ExecutionConfig (no more paper_execution_mode)
        paper_mode = bool(_exec.paper)
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
            get_risk_snapshot = persistence_service.get_risk_snapshot_sync if persistence_service else None
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
            from src.api.ws_manager import get_ws_manager

            mgr = get_ws_manager()
            if mgr:
                await mgr.broadcast(
                    {
                        "type": "order_created",
                        "order_id": getattr(o, "order_id", None),
                        "broker_order_id": getattr(o, "broker_order_id", None),
                        "symbol": getattr(o, "symbol", ""),
                        "exchange": getattr(getattr(o, "exchange", None), "value", str(getattr(o, "exchange", ""))),
                        "side": getattr(o, "side", ""),
                        "quantity": getattr(o, "quantity", 0),
                        "status": getattr(getattr(o, "status", None), "value", str(getattr(o, "status", ""))),
                    }
                )

        # GAP-18: Create rate limiter for order flood protection
        from src.execution.order_entry.rate_limiter import OrderRateLimiter, RateLimitConfig

        _rate_limiter = OrderRateLimiter(RateLimitConfig(max_orders_per_minute=60))

        # GAP-17: Wire Kafka publisher when KAFKA_BROKER_URL is configured
        _kafka_publish_fn = None
        _kafka_broker_url = os.environ.get("KAFKA_BROKER_URL") or os.environ.get("MD_KAFKA_BROKERS")
        if _kafka_broker_url:
            try:
                import json as _json

                from aiokafka import AIOKafkaProducer

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

        # P2-2: Wire OTR monitor into OrderEntryService for NSE 10:1 enforcement
        try:
            from src.compliance.otr_monitor import OTRMonitor

            _otr_monitor = OTRMonitor()
            app.state.order_entry_service.set_otr_monitor(_otr_monitor)
            app.state.otr_monitor = _otr_monitor
            logger.info("OTR monitor wired to OrderEntryService (NSE 10:1 limit enforcement)")
        except Exception as _otr_err:
            logger.debug("OTR monitor not wired: %s", _otr_err)

        # Wire SurveillanceEngine for post-trade manipulation detection (audit API)
        try:
            from src.compliance.surveillance import SurveillanceEngine

            _surveillance = SurveillanceEngine()
            app.state.surveillance_engine = _surveillance
            logger.info("SurveillanceEngine initialized (layering/spoofing/wash detection)")
        except Exception as _surv_err:
            logger.debug("SurveillanceEngine not initialized: %s", _surv_err)

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
                status = (
                    OrderStatus.REJECTED
                    if event.fill_type == FillType.REJECT
                    else (
                        OrderStatus.CANCELLED
                        if event.fill_type == FillType.CANCEL
                        else (OrderStatus.FILLED if event.fill_type == FillType.FILL else OrderStatus.PARTIALLY_FILLED)
                    )
                )
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
            from src.api.ws_manager import get_ws_manager

            mgr = get_ws_manager()
            if mgr:
                await mgr.broadcast(
                    {
                        "type": "order_filled",
                        "order_id": event.order_id,
                        "symbol": event.symbol,
                        "side": event.side,
                        "filled_qty": event.filled_qty,
                        "avg_price": event.avg_price,
                        "fill_type": getattr(event.fill_type, "value", str(event.fill_type)),
                    }
                )
                rm = getattr(app.state, "risk_manager", None)
                if rm:
                    positions = [
                        {
                            "symbol": p.symbol,
                            "exchange": getattr(p.exchange, "value", str(p.exchange)),
                            "side": getattr(p.side, "value", str(p.side)),
                            "quantity": p.quantity,
                            "avg_price": getattr(p, "avg_price", 0.0),
                        }
                        for p in rm.positions
                    ]
                    await mgr.broadcast({"type": "position_updated", "positions": positions})
                    await mgr.broadcast({"type": "equity_updated", "equity": rm.equity, "daily_pnl": rm.daily_pnl})

        from src.ai.performance_tracker import PerformanceTracker
        from src.api.ws_manager import get_ws_manager

        def _broadcast_strategy_disabled(sid: str, reason: str):
            mgr = get_ws_manager()
            if mgr:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(mgr.broadcast({"type": "strategy_disabled", "strategy_id": sid, "reason": reason}))
                except RuntimeError:
                    logger.warning("No running event loop for strategy_disabled broadcast")

        def _broadcast_exposure_multiplier(sid: str, mult: float):
            mgr = get_ws_manager()
            if mgr:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        mgr.broadcast({"type": "exposure_multiplier_changed", "strategy_id": sid, "multiplier": mult})
                    )
                except RuntimeError:
                    logger.warning("No running event loop for exposure_multiplier broadcast")

        performance_tracker = PerformanceTracker(
            on_strategy_disabled=_broadcast_strategy_disabled,
            on_exposure_multiplier_changed=_broadcast_exposure_multiplier,
        )
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
                        await _loop.run_in_executor(
                            None, lambda: _ps.save_risk_snapshot_sync(_rm.equity, _rm.daily_pnl)
                        )
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
                            _vix_lvl = getattr(getattr(_trp, "state", None), "vix_level", None)
                            if _vix_lvl and hasattr(_vix_lvl, "value"):
                                if _vix_lvl.value == "EXTREME":
                                    ks = getattr(app.state, "kill_switch", None)
                                    if ks:
                                        from src.execution.order_entry.kill_switch import KillReason

                                        await ks.arm(KillReason.INDIA_VIX_SPIKE, "India VIX EXTREME (>35)")
                                    _an = getattr(app.state, "alert_notifier", None)
                                    if _an:
                                        from src.alerts.notifier import AlertSeverity

                                        await _an.send(
                                            AlertSeverity.CRITICAL,
                                            "India VIX Extreme",
                                            "VIX > 35 — kill switch armed",
                                            source="tail_risk",
                                        )
                                elif _vix_lvl.value == "HIGH":
                                    _an = getattr(app.state, "alert_notifier", None)
                                    if _an:
                                        from src.alerts.notifier import AlertSeverity

                                        await _an.send(
                                            AlertSeverity.WARNING,
                                            "India VIX High",
                                            "VIX 25-35 — exposure halved",
                                            source="tail_risk",
                                        )
                        except Exception as e:
                            logger.debug("Tail risk check: %s", e)

                    # ── Portfolio VaR update ──
                    _pvar = getattr(app.state, "portfolio_var", None)
                    if _pvar and rm:
                        try:
                            positions = [
                                {"symbol": p.symbol, "notional": abs(getattr(p, "avg_price", 0) * p.quantity)}
                                for p in rm.positions
                            ]
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
                        # India VIX (run blocking yfinance call in executor)
                        _vix_data = await _loop.run_in_executor(
                            None, lambda: _yf.Ticker("^INDIAVIX").history(period="1d")
                        )
                        if not _vix_data.empty:
                            _vix_val = float(_vix_data["Close"].iloc[-1])
                            app.state._india_vix = _vix_val
                            if _trp_ca:
                                _trp_ca.update_vix(_vix_val)
                        # USDINR (run blocking yfinance call in executor)
                        _fx_data = await _loop.run_in_executor(
                            None, lambda: _yf.Ticker("USDINR=X").history(period="2d")
                        )
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
                                logger.info(
                                    "Safe mode AUTO-CLEARED after %.0fs (broker OK, feed OK, circuit closed)",
                                    _cool_down,
                                )

                    # ── P1-2: Stale order sweep (phantom order protection) ──
                    _lifecycle = getattr(getattr(app.state, "order_entry_service", None), "lifecycle", None)
                    if _lifecycle is not None:
                        try:
                            stale = await _lifecycle.sweep_stale_orders(max_age_seconds=300.0)
                            if stale:
                                logger.warning("Stale order sweep cancelled %d orders: %s", len(stale), stale[:5])
                        except Exception as _e:
                            logger.debug("Stale order sweep: %s", _e)

                    # ── P2-3: Overnight gap risk check (15:00 IST — flag large positions) ──
                    try:
                        from datetime import datetime as _dt_ovn
                        from datetime import timedelta as _td_ovn
                        from datetime import timezone as _tz_ovn

                        _IST_ovn = _tz_ovn(_td_ovn(hours=5, minutes=30))
                        _now_ist_ovn = _dt_ovn.now(_IST_ovn)
                        if (
                            _now_ist_ovn.hour == 15
                            and _now_ist_ovn.minute < 5
                            and rm
                            and hasattr(rm, "check_overnight_risk")
                        ):
                            flagged = rm.check_overnight_risk(max_overnight_pct=3.0)
                            if flagged:
                                logger.warning(
                                    "OVERNIGHT RISK: %d positions exceed 3%% equity — reduce before close: %s",
                                    len(flagged),
                                    flagged[:5],
                                )
                                _an_ovn = getattr(app.state, "alert_notifier", None)
                                if _an_ovn:
                                    from src.alerts.notifier import AlertSeverity

                                    await _an_ovn.send(
                                        AlertSeverity.WARNING,
                                        "Overnight Risk",
                                        f"{len(flagged)} positions exceed 3% equity: {flagged[:3]}",
                                        source="overnight_risk",
                                    )
                    except Exception as _ovn_e:
                        logger.debug("Overnight risk check: %s", _ovn_e)

                    # ── Record intraday PnL snapshot for rolling loss tracking (Sprint 7.11) ──
                    if rm:
                        rm.record_intraday_snapshot()

                    # ── Alert on circuit breaker events ──
                    _an = getattr(app.state, "alert_notifier", None)
                    if _an and rm and rm.is_circuit_open():
                        from src.alerts.notifier import AlertSeverity

                        await _an.send(
                            AlertSeverity.CRITICAL,
                            "Circuit Breaker Open",
                            f"Daily PnL: {rm.daily_pnl:.2f}, Equity: {rm.equity:.2f}",
                            source="risk_heartbeat",
                        )

                    from src.api.ws_manager import get_ws_manager

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
                        if _trp2 and hasattr(_trp2, "state"):
                            _risk_update["vix_level"] = getattr(
                                getattr(_trp2.state, "vix_level", None), "value", "UNKNOWN"
                            )
                            _risk_update["tail_risk_exposure_scale"] = getattr(_trp2.state, "exposure_scale", 1.0)
                        await mgr.broadcast(_risk_update)

                        if rm.is_circuit_open():
                            await mgr.broadcast({"type": "circuit_open"})
                        ks = getattr(app.state, "kill_switch", None)
                        if ks:
                            state = await ks.get_state()
                            if state.armed:
                                await mgr.broadcast(
                                    {
                                        "type": "kill_switch_armed",
                                        "reason": str(state.reason) if state.reason else "",
                                        "detail": state.detail or "",
                                    }
                                )
                        # Market data feed status
                        _mds = getattr(app.state, "market_data_service", None)
                        if _mds:
                            st = _mds.get_status()
                            await mgr.broadcast(
                                {
                                    "type": "market_status_updated",
                                    "connected": st.get("connected", False),
                                    "healthy": st.get("healthy", False),
                                    "last_tick_ts": st.get("last_tick_ts"),
                                }
                            )
                        elif getattr(app.state, "yf_feeder", None):
                            _yff = app.state.yf_feeder
                            _yf_healthy = getattr(_yff, "_running", False)
                            await mgr.broadcast(
                                {
                                    "type": "market_status_updated",
                                    "connected": _yf_healthy,
                                    "healthy": _yf_healthy,
                                    "last_tick_ts": None,
                                }
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception("Risk snapshot save failed: %s", e)

        _snapshot_task = asyncio.create_task(_periodic_risk_snapshot())

    # ── Nightly archival task (Sprint 10.13) ──
    _archival_mgr = getattr(app.state, "archival_manager", None)
    if _archival_mgr is not None and getattr(app.state, "persistence_service", None):

        async def _nightly_archival():
            from datetime import datetime as _dt
            from datetime import timedelta as _td
            from datetime import timezone as _tz

            _IST = _tz(_td(hours=5, minutes=30))
            _last_archival_date = None
            while True:
                await asyncio.sleep(3600)  # check every hour
                try:
                    now_ist = _dt.now(_IST)
                    today = now_ist.strftime("%Y-%m-%d")
                    # Run at 02:00 IST (low activity window)
                    if now_ist.hour != 2:
                        continue
                    if _last_archival_date == today:
                        continue
                    _last_archival_date = today
                    logger.info("Starting nightly data archival...")
                    ps = getattr(app.state, "persistence_service", None)
                    if ps and hasattr(ps, "_engine"):
                        from sqlalchemy.orm import Session

                        with Session(ps._engine) as session:
                            results = _archival_mgr.run_full_archival(session)
                            logger.info("Nightly archival complete: %d operations", len(results))
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Nightly archival error: %s", e)

        _archival_task = asyncio.create_task(_nightly_archival())
        app.state._archival_task = _archival_task
        logger.info("Nightly archival task scheduled (02:00 IST)")

    # ── Gap Risk pre-market check (Sprint 7.3) ──
    _gap_risk_task = None
    _grm = getattr(app.state, "gap_risk_manager", None)
    if _grm is not None:

        async def _pre_market_gap_check():
            from datetime import datetime as _dt
            from datetime import timedelta as _td
            from datetime import timezone as _tz

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
                    await asyncio.sleep(120)  # P1-3: 2 minutes (was 5min)
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
            logger.info("Reconciliation periodic task started (every 2 minutes)")
        except Exception as e:
            logger.debug("Reconciliation not configured: %s", e)

    # ── Enhanced Broker Reconciliator with discrepancy alerting (Sprint 11.2) ──
    try:
        from src.execution.reconciliation import BrokerReconciliator

        def _get_local_pos():
            rm = getattr(app.state, "risk_manager", None)
            return list(rm.positions) if rm else []

        async def _get_broker_pos():
            gw = getattr(app.state, "gateway", None)
            if gw and hasattr(gw, "get_positions"):
                return await gw.get_positions()
            return []

        def _on_discrepancy(report):
            logger.critical(
                "BROKER RECONCILIATION DISCREPANCY: status=%s, impact=₹%.0f",
                report.status,
                report.total_notional_impact,
            )
            if report.status == "critical":
                rm = getattr(app.state, "risk_manager", None)
                if rm:
                    rm.open_circuit("reconciliation_discrepancy")

        _broker_reconciliator = BrokerReconciliator(
            get_local_positions=_get_local_pos,
            get_broker_positions=_get_broker_pos,
            alert_threshold_inr=10_000,
            critical_threshold_inr=100_000,
            on_discrepancy=_on_discrepancy,
        )
        app.state.broker_reconciliator = _broker_reconciliator
        logger.info("Enhanced broker reconciliator configured")
    except Exception as e:
        logger.debug("Broker reconciliator not configured: %s", e)

    # Capital deployment gate
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

        # Need _set_safe_mode_cb for heartbeat
        def _set_safe_mode_cb_heartbeat():
            app.state.safe_mode = True
            app.state.safe_mode_since = time.time()
            if getattr(app.state, "audit_repo", None):
                try:
                    app.state.audit_repo.append_sync("broker_failure", "system", {"reason": "heartbeat_failure"})
                except Exception as e:
                    logger.warning("Audit append failed (broker_failure): %s", e)

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
                        _set_safe_mode_cb_heartbeat()
                        _heartbeat_failures[0] = 0

        _heartbeat_task = asyncio.create_task(_broker_heartbeat())

    # Alpha Research pipeline (optional)
    app.state.alpha_research_pipeline = None
    try:
        from src.ai.alpha_research import (
            AlphaHypothesisGenerator,
            AlphaQualityScorer,
            CapacityModel,
            DecayMonitor,
            EdgePreservationRules,
            PipelineConfig,
            ResearchPipeline,
            SignalClustering,
            StatisticalValidator,
        )
        from src.ai.alpha_research.clustering import ClusterConfig
        from src.ai.alpha_research.decay import DecayConfig
        from src.ai.alpha_research.scoring import AlphaQualityScoreConfig

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


async def shutdown_execution(app: FastAPI) -> None:
    """Shutdown: cancel periodic tasks, stop fill listener/simulator, disconnect broker,
    stop Kafka producer."""
    global _snapshot_task, _heartbeat_task

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
    for _task_name in ("_gap_risk_task", "_reconciliation_task", "_drift_task", "_report_task", "_archival_task"):
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

    # ── Disconnect broker gateway ──
    try:
        _gw = getattr(app.state, "gateway", None)
        if _gw and hasattr(_gw, "disconnect"):
            await _gw.disconnect()
            logger.info("Broker gateway disconnected")
    except Exception as e:
        logger.warning("Broker disconnect failed: %s", e)

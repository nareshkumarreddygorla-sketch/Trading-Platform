"""
Single order entry pipe. ALL order flows (API, AI, manual, test) MUST go through
OrderEntryService.submit_order. No alternate path to broker.
Pipeline: validate -> lot size -> circuit limit -> idempotency -> kill switch -> circuit -> risk -> reserve -> market impact -> router -> lifecycle -> persist -> kafka -> return.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.core.events import Order, OrderStatus
from src.risk_engine import RiskManager

from ..circuit_limits import CircuitLimitChecker
from ..lifecycle import OrderLifecycle
from ..nse_lot_sizes import NSELotSizeValidator
from ..order_router import OrderRouter
from .error_classification import classify_error, is_retryable
from .idempotency import IdempotencyStore
from .kill_switch import KillSwitch
from .rate_limiter import OrderRateLimiter
from .request import OrderEntryRequest, OrderEntryResult, RejectReason
from .reservation import ExposureReservation

logger = logging.getLogger(__name__)


class OrderEntryService:
    """
    Mandatory single order entry. Enforced by architecture: no code path
    should call OrderRouter or gateway directly; all go through submit_order.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        order_router: OrderRouter,
        lifecycle: OrderLifecycle,
        idempotency_store: IdempotencyStore,
        kill_switch: KillSwitch,
        reservation: ExposureReservation,
        *,
        persist_order: Callable | None = None,
        persist_order_submitting: Callable | None = None,
        update_order_after_broker_ack: Callable | None = None,
        reject_order_submitting: Callable | None = None,
        publish_order_event: Callable | None = None,
        distributed_lock: object | None = None,
        cluster_reservation: object | None = None,
        rate_limiter: OrderRateLimiter | None = None,
        on_risk_rejected: Callable[[], None] | None = None,
        on_order_created: Callable[[Order], Any] | None = None,
        market_impact_model=None,
        adv_cache=None,
        lot_size_validator: NSELotSizeValidator | None = None,
        circuit_limit_checker: CircuitLimitChecker | None = None,
        max_single_order_adv_pct: float = 5.0,
    ):
        self.on_risk_rejected = on_risk_rejected
        self.on_order_created = on_order_created
        self._market_impact_model = market_impact_model
        self._adv_cache = adv_cache
        self._lot_size_validator = lot_size_validator
        self._circuit_limit_checker = circuit_limit_checker
        self._max_single_order_adv_pct = max_single_order_adv_pct
        self.risk_manager = risk_manager
        self.order_router = order_router
        self.lifecycle = lifecycle
        self.idempotency = idempotency_store
        self.kill_switch = kill_switch
        self.reservation = reservation
        self.persist_order = persist_order
        self.persist_order_submitting = persist_order_submitting
        self.update_order_after_broker_ack = update_order_after_broker_ack
        self.reject_order_submitting = reject_order_submitting
        self.publish_order_event = publish_order_event
        self._distributed_lock = distributed_lock
        self._cluster_reservation = cluster_reservation
        self._rate_limiter = rate_limiter
        self._global_lock = asyncio.Lock()
        self._get_safe_mode: Callable[[], bool] | None = None
        self._otr_monitor: object | None = None  # P2-2: OTR enforcement
        self._persist_retries = (0.5, 1.0, 2.0)
        self._persist_max_attempts = 3
        self._write_ahead = bool(persist_order_submitting and update_order_after_broker_ack and reject_order_submitting)
        # P2-5: Dead letter queue for timed-out/uncertain orders
        self._dead_letter_queue: list = []  # List of {idem_key, order_id, timestamp, reason}

    async def _release_reservation(self, order_id: str) -> None:
        """Release local and cluster reservation."""
        await self.reservation.release(order_id)
        if self._cluster_reservation:
            await self._cluster_reservation.release(order_id)

    async def _release_distributed_lock_if_held(self, lock_held_ref: list) -> None:
        """Release distributed lock and set ref to False. Idempotent."""
        if not lock_held_ref or not lock_held_ref[0] or not self._distributed_lock:
            return
        try:
            await self._distributed_lock.release()
        except Exception as ex:
            logger.warning("Distributed lock release failed: %s", ex)
        lock_held_ref[0] = False

    def set_safe_mode_getter(self, fn: Callable[[], bool]) -> None:
        """Set callable that returns True when safe_mode is active."""
        self._get_safe_mode = fn

    def set_otr_monitor(self, monitor) -> None:
        """P2-2: Set OTR monitor for NSE compliance enforcement."""
        self._otr_monitor = monitor

    async def _persist_order_with_retry(self, order: Order) -> None:
        """Persist order with retry and backoff. Increments orders_persist_failed_total on final failure."""
        last_error = None
        for attempt in range(self._persist_max_attempts):
            try:
                await self.persist_order(order)
                return
            except Exception as e:
                last_error = e
                if attempt < self._persist_max_attempts - 1:
                    delay = self._persist_retries[attempt]
                    logger.warning("Persist order attempt %s failed, retry in %s s: %s", attempt + 1, delay, e)
                    await asyncio.sleep(delay)
        try:
            from src.monitoring.metrics import track_orders_persist_failed_total

            track_orders_persist_failed_total()
        except Exception as e:
            logger.debug("Metrics track_orders_persist_failed_total unavailable: %s", e)
        raise last_error

    async def submit_order(self, request: OrderEntryRequest) -> OrderEntryResult:
        """
        Single entry point. Pipeline:
        1. Validate input
        2. Idempotency check
        3. Global kill-switch check
        4. Circuit breaker check
        5. RiskManager.can_place_order()
        6. Atomic position reservation
        7. OrderRouter.place_order()
        8. OrderLifecycle.register()
        9. Persist order state
        10. Publish to Kafka
        11. Return order_id
        """
        start = time.perf_counter()
        # 1. Validate
        ok, err = request.validate()
        if not ok:
            try:
                from src.monitoring.metrics import track_orders_rejected_total

                track_orders_rejected_total()
            except Exception as e:
                logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.VALIDATION,
                reject_detail=err,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        signal = request.signal
        quantity = request.quantity
        price = request.limit_price or signal.price
        if price is None or price <= 0:
            try:
                from src.monitoring.metrics import track_orders_rejected_total

                track_orders_rejected_total()
            except Exception as e:
                logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.VALIDATION,
                reject_detail="price required and positive",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 1b. Rate limiter: reject if order flood detected (skip for force_reduce)
        if (
            self._rate_limiter is not None
            and not getattr(request, "force_reduce", False)
            and not self._rate_limiter.allow()
        ):
            try:
                from src.monitoring.metrics import track_orders_rejected_total

                track_orders_rejected_total()
            except Exception as e:
                logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.VALIDATION,
                reject_detail="rate_limit_exceeded",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 1b2. Safe mode: block all non-reduce orders when safe_mode is active
        if self._get_safe_mode is not None and self._get_safe_mode() and not getattr(request, "force_reduce", False):
            logger.warning("Order BLOCKED: safe_mode active (symbol=%s side=%s)", signal.symbol, signal.side.value)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.CIRCUIT_BREAKER,
                reject_detail="safe_mode_active",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 1b3. P2-2: OTR enforcement — reject if OTR exceeds NSE 10:1 limit
        if self._otr_monitor is not None and not getattr(request, "force_reduce", False):
            try:
                algo_id = signal.strategy_id or "default_algo"
                otr_status = self._otr_monitor.record_order(algo_id, str(uuid.uuid4())[:8], signal.symbol)
                if hasattr(otr_status, "value") and otr_status.value == "HALTED":
                    logger.critical("OTR HALT: algo=%s — order rejected per NSE limit", algo_id)
                    return OrderEntryResult(
                        False,
                        reject_reason=RejectReason.VALIDATION,
                        reject_detail=f"otr_halt: algo {algo_id} exceeds NSE 10:1 limit",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
            except Exception as e:
                logger.debug("OTR check failed: %s", e)

        # 1c. Minimum order value check (₹1,000)
        if price is not None and price > 0:
            order_value = quantity * price
            if order_value < 1000:
                try:
                    from src.monitoring.metrics import track_orders_rejected_total

                    track_orders_rejected_total()
                except Exception as e:
                    logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                return OrderEntryResult(
                    False,
                    reject_reason=RejectReason.VALIDATION,
                    reject_detail=f"order_value_below_minimum: {order_value:.0f} < 1000",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

        # 1d. NSE lot size validation: quantity must be a multiple of the lot size
        if self._lot_size_validator is not None:
            segment = request.metadata.get("segment", "EQ")
            is_valid, adjusted_qty, lot_msg = self._lot_size_validator.validate_and_adjust(
                signal.symbol,
                quantity,
                segment=segment,
            )
            if not is_valid:
                if adjusted_qty > 0:
                    logger.info(
                        "Lot size adjustment: %s %s qty %d -> %d (%s)",
                        signal.side.value,
                        signal.symbol,
                        quantity,
                        adjusted_qty,
                        lot_msg,
                    )
                    quantity = adjusted_qty
                else:
                    try:
                        from src.monitoring.metrics import track_orders_rejected_total

                        track_orders_rejected_total()
                    except Exception as e:
                        logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                    return OrderEntryResult(
                        False,
                        reject_reason=RejectReason.VALIDATION,
                        reject_detail=f"lot_size_invalid: {lot_msg}",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

        # 1e. Circuit limit check: reject if price would hit exchange circuit limit
        if self._circuit_limit_checker is not None and price is not None and price > 0:
            segment = request.metadata.get("segment", "EQ")
            circuit_result = self._circuit_limit_checker.check_order(
                signal.symbol,
                price,
                segment=segment,
            )
            if not circuit_result.allowed:
                try:
                    from src.monitoring.metrics import track_orders_rejected_total

                    track_orders_rejected_total()
                except Exception as e:
                    logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                logger.warning(
                    "Circuit limit rejection: %s %s @ %.2f — %s",
                    signal.side.value,
                    signal.symbol,
                    price,
                    circuit_result.reason,
                )
                return OrderEntryResult(
                    False,
                    reject_reason=RejectReason.CIRCUIT_BREAKER,
                    reject_detail=f"circuit_limit: {circuit_result.reason}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

        # 1f. Maximum single-order size limit: reject if order > max_pct of ADV
        if self._adv_cache is not None and self._max_single_order_adv_pct > 0:
            try:
                adv = self._adv_cache.get_adv(signal.symbol)
                if adv and adv > 0:
                    max_qty = int(adv * self._max_single_order_adv_pct / 100)
                    if quantity > max_qty > 0:
                        logger.warning(
                            "Order size %d exceeds %.1f%% of ADV (%d) for %s; capping to %d",
                            quantity,
                            self._max_single_order_adv_pct,
                            adv,
                            signal.symbol,
                            max_qty,
                        )
                        quantity = max_qty
            except Exception as e:
                logger.debug("ADV max-order-size check failed (proceeding): %s", e)

        idem_key = request.idempotency_key or IdempotencyStore.derive_key(
            signal.strategy_id, signal.symbol, signal.side.value, quantity, price, datetime.now(UTC).isoformat()
        )

        # 2. Idempotency: require Redis so duplicates are never sent to broker
        if not await self.idempotency.is_available():
            try:
                from src.monitoring.metrics import track_orders_rejected_total

                track_orders_rejected_total()
            except Exception as e:
                logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.IDEMPOTENCY_UNAVAILABLE,
                reject_detail="idempotency_unavailable",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        existing = await self.idempotency.get(idem_key)
        if existing:
            return OrderEntryResult(
                True,
                order_id=existing.get("order_id"),
                broker_order_id=existing.get("broker_order_id") or None,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        order_id_placeholder = str(uuid.uuid4()) if self._write_ahead else f"res_{uuid.uuid4().hex[:16]}"
        reserved = await self.idempotency.set(idem_key, order_id_placeholder, None, "PENDING")
        if not reserved:
            existing = await self.idempotency.get(idem_key)
            if existing:
                return OrderEntryResult(
                    True,
                    order_id=existing.get("order_id"),
                    broker_order_id=existing.get("broker_order_id") or None,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            try:
                from src.monitoring.metrics import track_orders_rejected_total

                track_orders_rejected_total()
            except Exception as e:
                logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.IDEMPOTENCY_UNAVAILABLE,
                reject_detail="idempotency_reserve_failed",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 3. Kill switch (all checks under single lock to prevent race with FillHandler — FIX BUG 1.8)
        armed = await self.kill_switch.is_armed()
        if armed:
            async with self._global_lock:
                net_pos = self._net_position(signal.symbol, signal.exchange.value)
                state = await self.kill_switch.get_state()
                kill_allowed = KillSwitch.allow_reduce_only_order(
                    state, signal.symbol, signal.side.value, quantity, net_pos
                )
            if not kill_allowed:
                try:
                    from src.monitoring.metrics import track_orders_rejected_total

                    track_orders_rejected_total()
                except Exception as e:
                    logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                return OrderEntryResult(
                    False,
                    reject_reason=RejectReason.KILL_SWITCH,
                    reject_detail=state.detail or (state.reason.value if state.reason else ""),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

        # 4. Circuit breaker
        if self.risk_manager.is_circuit_open():
            try:
                from src.monitoring.metrics import track_orders_rejected_total

                track_orders_rejected_total()
            except Exception as e:
                logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
            return OrderEntryResult(
                False,
                reject_reason=RejectReason.CIRCUIT_BREAKER,
                reject_detail="circuit_breaker_open",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 4b. Distributed lock (cluster-wide single-writer for critical section)
        lock_held = [False]
        if self._distributed_lock:
            try:
                lock_held[0] = await self._distributed_lock.acquire()
            except Exception as e:
                logger.warning("Distributed lock acquire failed: %s", e)
            if not lock_held[0]:
                try:
                    from src.monitoring.metrics import track_orders_rejected_total

                    track_orders_rejected_total()
                except Exception as e:
                    logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                return OrderEntryResult(
                    False,
                    reject_reason=RejectReason.RESERVATION_FAILED,
                    reject_detail="distributed_lock_unavailable",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
        try:
            # 5 + 6. Risk check and reservation under one lock (atomic)
            # _global_lock (asyncio.Lock) serializes both submit_order and
            # FillHandler.on_fill_event, preventing concurrent position mutations.
            # No need for the threading-bridge dual-lock (acquire_async_lock) which
            # caused deadlocks: run_in_executor acquired _positions_lock on a worker
            # thread, then can_place_order tried to re-acquire it on the event loop.
            async with self._global_lock:
                # Determine if this order reduces existing exposure (exit order)
                net_pos = self._net_position(
                    signal.symbol, signal.exchange.value if hasattr(signal.exchange, "value") else str(signal.exchange)
                )
                is_reducing = False
                force_reduce = getattr(request, "force_reduce", False)
                if net_pos > 0 and signal.side.value == "SELL":
                    is_reducing = True
                elif net_pos < 0 and signal.side.value == "BUY":
                    is_reducing = True
                r = self.risk_manager.can_place_order(
                    signal, quantity, price, is_reducing=is_reducing, force_reduce=force_reduce
                )
                if not r.allowed:
                    try:
                        from src.monitoring.metrics import track_orders_rejected_total

                        track_orders_rejected_total()
                    except Exception as e:
                        logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                    if self.on_risk_rejected:
                        try:
                            self.on_risk_rejected()
                        except Exception as ex:
                            logger.warning("on_risk_rejected callback failed: %s", ex)
                    return OrderEntryResult(
                        False,
                        reject_reason=RejectReason.RISK_REJECTED,
                        reject_detail=r.reason,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

                active_order_count = await self.lifecycle.count_active()
                if self._cluster_reservation:
                    # Use max() to avoid double-counting: active orders overlap with positions
                    max_allowed = self.risk_manager.limits.max_open_positions - max(
                        len(self.risk_manager.positions), active_order_count
                    )
                    if not await self._cluster_reservation.reserve(order_id_placeholder, max(max_allowed, 0)):
                        await self._release_distributed_lock_if_held(lock_held)
                        try:
                            from src.monitoring.metrics import track_orders_rejected_total

                            track_orders_rejected_total()
                        except Exception as e:
                            logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                        return OrderEntryResult(
                            False,
                            reject_reason=RejectReason.RESERVATION_FAILED,
                            reject_detail="cluster_max_open_positions_exceeded",
                            latency_ms=(time.perf_counter() - start) * 1000,
                        )
                ok, res_reason = await self.reservation.reserve(
                    order_id_placeholder,
                    signal.symbol,
                    signal.exchange.value,
                    signal.side.value,
                    quantity,
                    price,
                    self.risk_manager.positions,
                    self.risk_manager.limits.max_open_positions,
                    self.risk_manager.limits.max_position_pct,
                    self.risk_manager.equity,
                    active_order_count=active_order_count,
                )
                if not ok:
                    if self._cluster_reservation:
                        await self._cluster_reservation.release(order_id_placeholder)
                    await self._release_distributed_lock_if_held(lock_held)
                    try:
                        from src.monitoring.metrics import track_orders_rejected_total

                        track_orders_rejected_total()
                    except Exception as e:
                        logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                    if reserved:
                        await self.idempotency.update(idem_key, order_id_placeholder, None, "REJECTED")
                    return OrderEntryResult(
                        False,
                        reject_reason=RejectReason.RESERVATION_FAILED,
                        reject_detail=res_reason,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

            # 6b. Write-ahead: persist order with status SUBMITTING before broker (eliminates crash window)
            if self._write_ahead and self.persist_order_submitting:
                order_wa = Order(
                    order_id=order_id_placeholder,
                    strategy_id=signal.strategy_id or "",
                    symbol=signal.symbol,
                    exchange=signal.exchange,
                    side=signal.side,
                    quantity=quantity,
                    order_type=request.order_type,
                    limit_price=request.limit_price,
                    status=OrderStatus.PENDING,
                    filled_qty=0.0,
                    avg_price=None,
                    broker_order_id=None,
                    metadata={},
                )
                try:
                    await self.persist_order_submitting(order_wa, idem_key)
                except Exception as e:
                    logger.exception("Write-ahead persist failed: %s", e)
                    await self._release_reservation(order_id_placeholder)
                    await self.idempotency.update(idem_key, order_id_placeholder, None, "REJECTED")
                    await self._release_distributed_lock_if_held(lock_held)
                    try:
                        from src.monitoring.metrics import track_orders_rejected_total

                        track_orders_rejected_total()
                    except Exception as e:
                        logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                    return OrderEntryResult(
                        False,
                        reject_reason=RejectReason.BROKER_ERROR,
                        reject_detail="write_ahead_persist_failed",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

            # 6c. Market impact check: reduce quantity if alpha < cost (fail open if no data)
            if self._market_impact_model is not None and not getattr(request, "force_reduce", False):
                try:
                    adv = 500_000  # default
                    if self._adv_cache:
                        adv = self._adv_cache.get_adv(signal.symbol) or adv
                    alpha_bps = (signal.score or 0) * 100  # signal score as alpha proxy (bps)
                    if alpha_bps > 0:
                        impact_result = self._market_impact_model.check_alpha_sufficient(
                            quantity, price, adv, alpha_bps, sigma=0.02, transaction_cost_bps=12.0
                        )
                        if not impact_result.alpha_sufficient and impact_result.recommended_qty > 0:
                            logger.info(
                                "Market impact: reducing %s qty %d -> %d (alpha=%0.1fbps < cost=%0.1fbps)",
                                signal.symbol,
                                quantity,
                                impact_result.recommended_qty,
                                alpha_bps,
                                impact_result.total_cost_bps,
                            )
                            quantity = impact_result.recommended_qty
                        elif not impact_result.alpha_sufficient and impact_result.recommended_qty <= 0:
                            logger.info(
                                "Market impact: rejecting %s (alpha=%0.1fbps < min cost)", signal.symbol, alpha_bps
                            )
                except Exception as e:
                    logger.debug("Market impact check failed (proceeding): %s", e)

            # 7. Router (outside lock to avoid holding lock during network call); timeout to avoid leaking reservation
            BROKER_TIMEOUT_SECONDS = 30
            try:
                order = await asyncio.wait_for(
                    self.order_router.place_order(
                        signal,
                        quantity,
                        order_type=request.order_type,
                        limit_price=request.limit_price,
                    ),
                    timeout=BROKER_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                try:
                    from src.monitoring.metrics import track_orders_rejected_total

                    track_orders_rejected_total()
                except Exception as e:
                    logger.debug("Metrics track_orders_rejected_total unavailable: %s", e)
                if self.reject_order_submitting:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: self.reject_order_submitting(order_id_placeholder))
                await self._release_reservation(order_id_placeholder)
                await self._release_distributed_lock_if_held(lock_held)
                logger.error(
                    "Broker place_order timeout after %s s — ORDER MAY HAVE BEEN SUBMITTED. "
                    "Reconciliation required: check broker order book for idem_key=%s",
                    BROKER_TIMEOUT_SECONDS,
                    idem_key,
                )
                # Mark as TIMEOUT_UNCERTAIN in idempotency store so reconciliation can find it
                await self.idempotency.update(idem_key, order_id_placeholder, None, "TIMEOUT_UNCERTAIN")
                # P2-5: Add to dead letter queue for reconciliation
                self._dead_letter_queue.append(
                    {
                        "idem_key": idem_key,
                        "order_id": order_id_placeholder,
                        "symbol": signal.symbol,
                        "side": signal.side.value,
                        "quantity": quantity,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "reason": "broker_timeout",
                    }
                )
                logger.critical(
                    "DLQ: added timeout order %s (symbol=%s) — requires reconciliation",
                    order_id_placeholder,
                    signal.symbol,
                )
                return OrderEntryResult(
                    False,
                    reject_reason=RejectReason.TIMEOUT,
                    reject_detail=f"broker_timeout_uncertain:idem={idem_key}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                error_cat = classify_error(e)
                try:
                    from src.monitoring.metrics import track_orders_rejected_total

                    track_orders_rejected_total()
                except Exception as _me:
                    logger.debug("Metrics track_orders_rejected_total unavailable: %s", _me)
                if self.reject_order_submitting:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: self.reject_order_submitting(order_id_placeholder))
                await self._release_reservation(order_id_placeholder)
                await self._release_distributed_lock_if_held(lock_held)
                if reserved:
                    await self.idempotency.update(idem_key, order_id_placeholder, None, "REJECTED")
                logger.exception(
                    "Broker place_order failed (category=%s retryable=%s): %s",
                    error_cat.value,
                    is_retryable(e),
                    e,
                )
                return OrderEntryResult(
                    False,
                    reject_reason=RejectReason.BROKER_ERROR,
                    reject_detail=f"[{error_cat.value}] {e}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            real_order_id = order_id_placeholder if self._write_ahead else (order.order_id or order_id_placeholder)
            broker_order_id = order.broker_order_id or order.order_id
            await self.reservation.commit(order_id_placeholder)
            if self._cluster_reservation:
                await self._cluster_reservation.release(order_id_placeholder)

            # 8. Write-ahead: update SUBMITTING -> NEW; else idempotency overwrite
            if self._write_ahead and self.update_order_after_broker_ack:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, lambda: self.update_order_after_broker_ack(order_id_placeholder, broker_order_id)
                )
            await self.idempotency.update(idem_key, real_order_id, broker_order_id, order.status.value)

            # 9. Lifecycle (use our order_id and broker_order_id from response)
            order_to_register = (
                Order(
                    order_id=real_order_id,
                    strategy_id=order.strategy_id or signal.strategy_id or "",
                    symbol=order.symbol or signal.symbol,
                    exchange=order.exchange or signal.exchange,
                    side=order.side or signal.side,
                    quantity=order.quantity or quantity,
                    order_type=order.order_type or request.order_type,
                    limit_price=order.limit_price or request.limit_price,
                    status=order.status,
                    filled_qty=getattr(order, "filled_qty", 0.0) or 0.0,
                    avg_price=order.avg_price,
                    broker_order_id=broker_order_id,
                    metadata=getattr(order, "metadata", {}) or {},
                )
                if self._write_ahead
                else order
            )
            if not await self.lifecycle.register(order_to_register):
                logger.warning("Lifecycle refused to register order (empty order_id); order_id=%s", real_order_id)

            # 10. Persist (skip when write-ahead: already persisted and updated)
            if not self._write_ahead and self.persist_order and real_order_id and str(real_order_id).strip():
                try:
                    await self._persist_order_with_retry(order)
                except Exception as e:
                    logger.exception("Persist order failed after retries: %s", e)

            # 11. Publish
            if self.publish_order_event:
                try:
                    await self.publish_order_event(order)
                except Exception as e:
                    logger.exception("Publish order event failed: %s", e)

            # 12. Order-created callback (e.g. WebSocket broadcast)
            if self.on_order_created:
                try:
                    o = (
                        order_to_register
                        if self._write_ahead
                        else Order(
                            order_id=real_order_id,
                            strategy_id=order.strategy_id or signal.strategy_id or "",
                            symbol=order.symbol or signal.symbol,
                            exchange=order.exchange or signal.exchange,
                            side=order.side or signal.side,
                            quantity=order.quantity or quantity,
                            order_type=order.order_type or request.order_type,
                            limit_price=order.limit_price or request.limit_price,
                            status=order.status,
                            filled_qty=getattr(order, "filled_qty", 0.0) or 0.0,
                            avg_price=order.avg_price,
                            broker_order_id=broker_order_id,
                            metadata=getattr(order, "metadata", {}) or {},
                        )
                    )
                    res = self.on_order_created(o)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    logger.exception("on_order_created failed: %s", e)

            await self._release_distributed_lock_if_held(lock_held)
            try:
                from src.monitoring.metrics import track_orders_total

                track_orders_total()
            except Exception as e:
                logger.debug("Metrics track_orders_total unavailable: %s", e)
            return OrderEntryResult(
                True,
                order_id=real_order_id,
                broker_order_id=order.broker_order_id,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        finally:
            if lock_held[0] and self._distributed_lock:
                try:
                    await self._distributed_lock.release()
                except Exception as ex:
                    logger.warning("Distributed lock release failed: %s", ex)
                lock_held[0] = False

    def _net_position(self, symbol: str, exchange: str) -> float:
        """Net position for symbol (long positive, short negative).

        BUG 33 NOTE: This method must ALWAYS be called under ``_global_lock`` to
        prevent races with FillHandler updating positions concurrently. All current
        call sites already hold the lock (verified).
        """
        net = 0.0
        for p in self.risk_manager.positions:
            if p.symbol == symbol and p.exchange.value == exchange:
                if p.side.value == "BUY":
                    net += p.quantity
                else:
                    net -= p.quantity
        return net

    def get_dead_letter_queue(self) -> list:
        """P2-5: Return current dead letter queue entries for reconciliation."""
        return list(self._dead_letter_queue)

    def clear_dead_letter(self, order_id: str) -> bool:
        """P2-5: Remove a resolved entry from the DLQ."""
        before = len(self._dead_letter_queue)
        self._dead_letter_queue = [e for e in self._dead_letter_queue if e.get("order_id") != order_id]
        return len(self._dead_letter_queue) < before

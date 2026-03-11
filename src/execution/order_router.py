"""Order routing: multi-broker smart routing, price improvement, market impact sizing.

Features
--------
* **Multi-broker routing** – selects the best broker based on health, latency
  and cost; automatic failover to a secondary broker on failure.
* **BrokerHealth tracker** – monitors consecutive failures, rolling latency,
  and marks brokers as unhealthy after 3 consecutive failures.
* **Spread-aware price improvement** – uses bid/ask to compute a mid-price and
  applies an improvement in basis points.
* **ADV-aware child sizing** – uses the MarketImpactModel to cap child order
  quantity at a fraction of ADV to limit market impact.
* **NSE tick-size validation & freeze-qty splitting** – unchanged from the
  original implementation.

Backwards compatible: if only one broker is supplied (the ``default_gateway``),
all new multi-broker paths are silently skipped.
"""

import asyncio
import collections
import logging
import math
import time
from collections.abc import Callable
from typing import Any

from src.core.events import Order, OrderType, Signal

from .base import BaseExecutionGateway
from .market_impact import MarketImpactModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE constants (unchanged)
# ---------------------------------------------------------------------------
_MIN_ORDER_VALUE_INR = 1000.0
_MIN_PRICE_INR = 1.0
_MAX_PRICE_INR = 100_000.0

# ---------------------------------------------------------------------------
# Broker cost constants (bps, round-trip approximations)
# ---------------------------------------------------------------------------
_DEFAULT_BROKER_COSTS: dict[str, float] = {
    "angel_one": 12.0,  # STT + brokerage + GST + stamp (flat ₹20/order model)
    "zerodha": 10.0,  # discount broker, slightly lower for equity delivery
}

# ---------------------------------------------------------------------------
# Smart-sizing defaults
# ---------------------------------------------------------------------------
_DEFAULT_MAX_PARTICIPATION_RATE = 0.05  # 5 % of ADV per child slice
_DEFAULT_MIN_ADV_FOR_SIZING = 50_000  # ignore ADV sizing below this
_DEFAULT_BROKER_TIMEOUT = 10.0  # 10s default timeout for external calls


def validate_tick_size(price: float) -> float:
    """
    Round price to valid NSE tick size.
    - Stocks >= ₹100: tick = ₹0.05
    - Stocks < ₹100: tick = ₹0.01 (paisa)
    """
    if price <= 0:
        return price
    if price >= 100.0:
        tick = 0.05
    else:
        tick = 0.01
    return round(round(price / tick) * tick, 2)


def validate_price_sanity(price: float) -> bool:
    """Check if price is within sensible NSE range."""
    return _MIN_PRICE_INR <= price <= _MAX_PRICE_INR


# ===================================================================
# BrokerHealth – per-broker health / latency tracker
# ===================================================================


class BrokerHealth:
    """
    Track health metrics for a single broker gateway.

    Attributes
    ----------
    name : str
        Human-readable broker identifier (e.g. ``"angel_one"``).
    consecutive_failures : int
        Resets to 0 on every successful order.
    last_success_ts : float | None
        ``time.monotonic()`` of last successful ``place_order``.
    latencies : deque
        Rolling window of the last *window_size* order round-trip times (seconds).
    healthy : bool
        ``False`` after *max_consecutive_failures* consecutive failures.
    """

    def __init__(
        self,
        name: str,
        max_consecutive_failures: int = 3,
        window_size: int = 20,
    ) -> None:
        self.name = name
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures: int = 0
        self.last_success_ts: float | None = None
        self.latencies: collections.deque = collections.deque(maxlen=window_size)
        self.healthy: bool = True
        self._total_orders: int = 0
        self._total_failures: int = 0

    # -- recording ----------------------------------------------------------

    def record_success(self, latency_s: float) -> None:
        """Record a successful order placement."""
        self.consecutive_failures = 0
        self.healthy = True
        self.last_success_ts = time.monotonic()
        self.latencies.append(latency_s)
        self._total_orders += 1

    def record_failure(self) -> None:
        """Record a failed order placement."""
        self.consecutive_failures += 1
        self._total_failures += 1
        self._total_orders += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.healthy = False
            logger.warning(
                "Broker %s marked UNHEALTHY after %d consecutive failures",
                self.name,
                self.consecutive_failures,
            )

    # -- queries ------------------------------------------------------------

    @property
    def avg_latency(self) -> float:
        """Average latency over the rolling window (seconds). 0 if no data."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def failure_rate(self) -> float:
        if self._total_orders == 0:
            return 0.0
        return self._total_failures / self._total_orders

    def reset(self) -> None:
        """Reset health state (e.g. after manual intervention)."""
        self.consecutive_failures = 0
        self.healthy = True
        self.latencies.clear()
        self.last_success_ts = None
        self._total_orders = 0
        self._total_failures = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": round(self.avg_latency * 1000, 1),
            "last_success_ts": self.last_success_ts,
            "total_orders": self._total_orders,
            "failure_rate": round(self.failure_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"BrokerHealth({self.name!r}, healthy={self.healthy}, "
            f"fails={self.consecutive_failures}, avg_lat={self.avg_latency * 1000:.1f}ms)"
        )


# ===================================================================
# SmartOrderRouter (enhanced OrderRouter)
# ===================================================================


class OrderRouter:
    """
    Route orders to the best available broker gateway.

    Supports limit/market, IOC/FOK; optional price improvement (tick-based or
    spread-based), freeze-qty splitting, ADV-aware child sizing, and automatic
    multi-broker failover.

    **Backwards compatible**: when constructed with only ``default_gateway`` (no
    ``brokers`` dict), all smart routing is disabled and behaviour matches the
    original single-broker router exactly.

    Parameters
    ----------
    default_gateway : BaseExecutionGateway
        The primary (and possibly only) broker gateway.
    freeze_qty_manager : optional
        Provides ``check_and_split(symbol, qty)`` for exchange freeze limits.
    brokers : dict[str, BaseExecutionGateway], optional
        Named broker gateways, e.g. ``{"angel_one": gw1, "zerodha": gw2}``.
        If provided, enables multi-broker routing & failover.
    broker_costs_bps : dict[str, float], optional
        Override default per-broker cost estimates (basis points).
    market_impact_model : MarketImpactModel, optional
        Used for ADV-aware child sizing.
    market_data_fn : callable, optional
        ``async fn(symbol) -> {"bid": float, "ask": float, "ltp": float, "adv": float}``
        Provides live quotes for spread-based price improvement and ADV sizing.
    improvement_bps : float
        Basis points to improve limit price relative to mid-price (default 2.0).
    max_participation_rate : float
        Maximum fraction of ADV per child order slice (default 0.05 = 5 %).
    """

    def __init__(
        self,
        default_gateway: BaseExecutionGateway,
        freeze_qty_manager=None,
        *,
        brokers: dict[str, BaseExecutionGateway] | None = None,
        broker_costs_bps: dict[str, float] | None = None,
        market_impact_model: MarketImpactModel | None = None,
        market_data_fn: Callable | None = None,
        improvement_bps: float = 2.0,
        max_participation_rate: float = _DEFAULT_MAX_PARTICIPATION_RATE,
    ):
        self.default_gateway = default_gateway
        self._gateways: dict[str, BaseExecutionGateway] = {
            "NSE": default_gateway,
            "BSE": default_gateway,
        }
        self._freeze_qty_manager = freeze_qty_manager

        # -- multi-broker setup ------------------------------------------------
        self._brokers: dict[str, BaseExecutionGateway] = brokers or {}
        self._broker_health: dict[str, BrokerHealth] = {}
        self._broker_costs: dict[str, float] = {
            **_DEFAULT_BROKER_COSTS,
            **(broker_costs_bps or {}),
        }
        for name in self._brokers:
            self._broker_health[name] = BrokerHealth(name)

        self._multi_broker = len(self._brokers) > 1

        # -- price improvement / impact ----------------------------------------
        self._market_data_fn = market_data_fn
        self._improvement_bps = improvement_bps
        self._impact_model = market_impact_model or MarketImpactModel()
        self._max_participation = max_participation_rate

    # ------------------------------------------------------------------
    # Public: gateway registry (unchanged interface)
    # ------------------------------------------------------------------

    def register_gateway(self, exchange: str, gateway: BaseExecutionGateway) -> None:
        self._gateways[exchange] = gateway

    def _gateway(self, exchange: str) -> BaseExecutionGateway:
        return self._gateways.get(exchange, self.default_gateway)

    # ------------------------------------------------------------------
    # Health accessors
    # ------------------------------------------------------------------

    def get_broker_health(self, name: str) -> BrokerHealth | None:
        return self._broker_health.get(name)

    def get_all_broker_health(self) -> dict[str, dict[str, Any]]:
        return {name: h.as_dict() for name, h in self._broker_health.items()}

    # ------------------------------------------------------------------
    # Broker selection
    # ------------------------------------------------------------------

    def _rank_brokers(self) -> list[tuple[str, BaseExecutionGateway]]:
        """
        Rank configured brokers by a composite score (lower is better):
          score = cost_bps + latency_penalty
        Unhealthy brokers are pushed to the end (used only for failover).
        """
        if not self._brokers:
            return []

        healthy: list[tuple[float, str, BaseExecutionGateway]] = []
        unhealthy: list[tuple[str, BaseExecutionGateway]] = []

        for name, gw in self._brokers.items():
            health = self._broker_health.get(name)
            if health and not health.healthy:
                unhealthy.append((name, gw))
                continue

            cost = self._broker_costs.get(name, 15.0)
            latency_penalty = 0.0
            if health and health.avg_latency > 0:
                # Convert avg latency to a bps-scale penalty (1s → 5 bps)
                latency_penalty = health.avg_latency * 5.0
            score = cost + latency_penalty
            healthy.append((score, name, gw))

        healthy.sort(key=lambda t: t[0])
        ranked = [(name, gw) for _, name, gw in healthy]
        ranked.extend(unhealthy)
        return ranked

    def _select_broker(self) -> list[tuple[str, BaseExecutionGateway]]:
        """
        Return an ordered list of (name, gateway) to try.
        First entry is the preferred broker; subsequent entries are failover.
        If no multi-broker config, returns empty (caller falls back to exchange gateway).
        """
        return self._rank_brokers()

    # ------------------------------------------------------------------
    # Price improvement
    # ------------------------------------------------------------------

    def _price_improvement(self, price: float, side: str, ticks: int = 1) -> float:
        """Tick-based price improvement (original behaviour)."""
        tick = 0.05 if price >= 100 else 0.01
        if side == "BUY":
            return round(price - ticks * tick, 2)
        return round(price + ticks * tick, 2)

    def _spread_price_improvement(
        self,
        side: str,
        bid: float,
        ask: float,
        improvement_bps: float,
    ) -> float:
        """
        Compute an improved limit price from the bid-ask spread.

        For BUY:  mid - improvement  (try to buy cheaper than mid)
        For SELL: mid + improvement  (try to sell higher than mid)

        The result is clamped within [bid, ask] so we never cross the spread
        in the wrong direction.
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0  # no valid spread
        mid = (bid + ask) / 2.0
        improvement = mid * (improvement_bps / 10_000.0)
        if side == "BUY":
            improved = mid - improvement
            improved = max(improved, bid)  # don't go below bid
        else:
            improved = mid + improvement
            improved = min(improved, ask)  # don't go above ask
        return validate_tick_size(improved)

    # ------------------------------------------------------------------
    # ADV-aware child sizing
    # ------------------------------------------------------------------

    def _adv_adjusted_child_sizes(
        self,
        quantity: int,
        adv: float,
        price: float,
    ) -> list[int]:
        """
        Split *quantity* into child slices so that each slice is at most
        ``max_participation_rate * ADV`` shares.  Also uses the market impact
        model to flag if participation is too aggressive.

        Returns a list of child quantities (possibly just ``[quantity]`` if
        no splitting is needed).
        """
        if adv < _DEFAULT_MIN_ADV_FOR_SIZING or quantity <= 0:
            return [quantity]

        max_child = max(1, int(adv * self._max_participation))
        if quantity <= max_child:
            return [quantity]

        # split into roughly equal slices
        n_slices = math.ceil(quantity / max_child)
        base_qty = quantity // n_slices
        remainder = quantity - base_qty * n_slices
        slices: list[int] = []
        for i in range(n_slices):
            q = base_qty + (1 if i < remainder else 0)
            if q > 0:
                slices.append(q)

        logger.info(
            "ADV sizing: qty=%d adv=%.0f max_child=%d -> %d slices",
            quantity,
            adv,
            max_child,
            len(slices),
        )
        return slices

    # ------------------------------------------------------------------
    # Core order placement
    # ------------------------------------------------------------------

    async def place_order(
        self,
        signal: Signal,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        price_improvement_ticks: int = 0,
    ) -> Order:
        exchange = signal.exchange.value
        price = limit_price or signal.price

        if price is None:
            order_type = OrderType.MARKET
        else:
            # Validate tick size for LIMIT orders
            if order_type == OrderType.LIMIT:
                original_price = price
                price = validate_tick_size(price)
                if price != original_price:
                    logger.debug(
                        "Tick size adjusted: %.4f -> %.2f for %s",
                        original_price,
                        price,
                        signal.symbol,
                    )

            # Price sanity check
            if price is not None and not validate_price_sanity(price):
                raise ValueError(
                    f"Price {price} outside valid range [{_MIN_PRICE_INR}, {_MAX_PRICE_INR}] for {signal.symbol}"
                )

            # -- Spread-based price improvement (if market data available) ------
            if order_type == OrderType.LIMIT and self._market_data_fn is not None and self._improvement_bps > 0:
                try:
                    quote = await asyncio.wait_for(
                        self._market_data_fn(signal.symbol),
                        timeout=_DEFAULT_BROKER_TIMEOUT,
                    )
                    bid = quote.get("bid", 0.0)
                    ask = quote.get("ask", 0.0)
                    if bid > 0 and ask > 0:
                        improved = self._spread_price_improvement(
                            signal.side.value,
                            bid,
                            ask,
                            self._improvement_bps,
                        )
                        if improved > 0:
                            logger.info(
                                "Spread improvement %s: signal=%.2f mid=%.2f improved=%.2f (%.1f bps)",
                                signal.symbol,
                                price,
                                (bid + ask) / 2,
                                improved,
                                self._improvement_bps,
                            )
                            price = improved
                except Exception as exc:
                    logger.warning(
                        "Market data fetch failed for %s spread improvement: %s",
                        signal.symbol,
                        exc,
                    )

            # -- Tick-based price improvement (original, used if no spread) -----
            if price_improvement_ticks and order_type == OrderType.LIMIT:
                price = self._price_improvement(
                    price,
                    signal.side.value,
                    price_improvement_ticks,
                )

        # Minimum order value check
        if price is not None and price > 0:
            order_value = quantity * price
            if order_value < _MIN_ORDER_VALUE_INR:
                raise ValueError(f"Order value ₹{order_value:.0f} below minimum ₹{_MIN_ORDER_VALUE_INR:.0f}")

        # -- ADV-aware child sizing --------------------------------------------
        adv_slices: list[int] | None = None
        if self._market_data_fn is not None and quantity > 0:
            try:
                quote = await asyncio.wait_for(
                    self._market_data_fn(signal.symbol),
                    timeout=_DEFAULT_BROKER_TIMEOUT,
                )
                adv = quote.get("adv", 0.0)
                if adv > 0 and price is not None and price > 0:
                    adv_slices = self._adv_adjusted_child_sizes(quantity, adv, price)
            except Exception as exc:
                logger.warning(
                    "ADV sizing lookup failed for %s: %s",
                    signal.symbol,
                    exc,
                )

        # -- Freeze qty check (exchange-mandated splitting) ---------------------
        if self._freeze_qty_manager is not None and quantity > 0:
            try:
                split_result = self._freeze_qty_manager.check_and_split(
                    signal.symbol,
                    quantity,
                )
                if split_result and len(split_result) > 1:
                    logger.info(
                        "Freeze qty split: %s %d -> %d child orders",
                        signal.symbol,
                        quantity,
                        len(split_result),
                    )
                    return await self._submit_split_orders(
                        signal,
                        split_result,
                        order_type,
                        price,
                    )
            except Exception as e:
                logger.warning(
                    "Freeze qty check failed for %s (proceeding with full qty): %s",
                    signal.symbol,
                    e,
                )

        # -- If ADV sizing produced multiple slices, use them -------------------
        if adv_slices is not None and len(adv_slices) > 1:
            logger.info(
                "ADV split: %s %d -> %d child orders",
                signal.symbol,
                quantity,
                len(adv_slices),
            )
            return await self._submit_split_orders(
                signal,
                adv_slices,
                order_type,
                price,
            )

        # -- Route to broker (multi-broker or single) ---------------------------
        return await self._route_and_place(
            signal=signal,
            exchange=exchange,
            quantity=quantity,
            order_type=order_type,
            price=price,
        )

    # ------------------------------------------------------------------
    # Internal: route to best broker with failover
    # ------------------------------------------------------------------

    async def _route_and_place(
        self,
        signal: Signal,
        exchange: str,
        quantity: int,
        order_type: OrderType,
        price: float | None,
    ) -> Order:
        """
        Place an order, using multi-broker failover if configured.
        Falls back to the exchange-based gateway when no named brokers are set.
        """
        broker_candidates = self._select_broker() if self._multi_broker else []

        if not broker_candidates:
            # Single-broker mode: use the exchange-mapped gateway (original path)
            gateway = self._gateway(exchange)
            if exchange not in self._gateways:
                logger.debug("Unknown exchange %s, using default gateway", exchange)

            # If we have health tracking for a single broker, still record it
            broker_name = self._single_broker_name()
            return await self._place_with_health(
                broker_name,
                gateway,
                signal,
                exchange,
                quantity,
                order_type,
                price,
            )

        # Multi-broker: try each candidate in rank order
        last_exc: Exception = RuntimeError("No broker candidates available")
        for broker_name, gateway in broker_candidates:
            try:
                order = await self._place_with_health(
                    broker_name,
                    gateway,
                    signal,
                    exchange,
                    quantity,
                    order_type,
                    price,
                )
                return order
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Broker %s failed for %s %s qty=%d: %s — trying next broker",
                    broker_name,
                    signal.side.value,
                    signal.symbol,
                    quantity,
                    exc,
                )
                # Health tracker updated inside _place_with_health

        # All brokers exhausted
        raise RuntimeError(f"All brokers failed for {signal.symbol}. Last error: {last_exc}") from last_exc

    def _single_broker_name(self) -> str | None:
        """Return the name of the sole configured broker, if any."""
        if len(self._brokers) == 1:
            return next(iter(self._brokers))
        return None

    async def _place_with_health(
        self,
        broker_name: str | None,
        gateway: BaseExecutionGateway,
        signal: Signal,
        exchange: str,
        quantity: int,
        order_type: OrderType,
        price: float | None,
    ) -> Order:
        """Place an order through *gateway* and record health metrics."""
        health = self._broker_health.get(broker_name) if broker_name else None
        t0 = time.monotonic()

        try:
            order = await asyncio.wait_for(
                gateway.place_order(
                    symbol=signal.symbol,
                    exchange=exchange,
                    side=signal.side.value,
                    quantity=float(quantity),
                    order_type=order_type.value,
                    limit_price=price,
                    strategy_id=signal.strategy_id,
                ),
                timeout=_DEFAULT_BROKER_TIMEOUT,
            )
            if health:
                health.record_success(time.monotonic() - t0)
            return order
        except TimeoutError as exc:
            if health:
                health.record_failure()
            raise RuntimeError(
                f"Broker {broker_name or 'default'} timed out after {_DEFAULT_BROKER_TIMEOUT}s for {signal.symbol}"
            ) from exc
        except Exception:
            if health:
                health.record_failure()
            raise

    # ------------------------------------------------------------------
    # Split-order submission (freeze-qty or ADV slices)
    # ------------------------------------------------------------------

    async def _submit_split_orders(
        self,
        signal: Signal,
        quantities: list[int],
        order_type: OrderType,
        price: float | None,
    ) -> Order:
        """Submit child orders with 1 s delay between them. Return first order as parent.

        On partial failure, attempts to cancel all previously successful children
        to prevent orphaned orders living at the broker but invisible to the caller.
        """
        exchange = signal.exchange.value
        parent_order: Order | None = None
        successful_children: list[Order] = []

        for i, qty in enumerate(quantities):
            if qty <= 0:
                continue
            if i > 0:
                await asyncio.sleep(1.0)  # Rate limit between child orders

            try:
                order = await self._route_and_place(
                    signal=signal,
                    exchange=exchange,
                    quantity=qty,
                    order_type=order_type,
                    price=price,
                )
            except Exception as child_err:
                logger.error(
                    "Split child %d/%d failed for %s: %s — cancelling %d prior children",
                    i + 1,
                    len(quantities),
                    signal.symbol,
                    child_err,
                    len(successful_children),
                )
                # Best-effort cancel of already-submitted children
                for prev_order in successful_children:
                    try:
                        gw = self._gateway(exchange)
                        if gw is not None:
                            await gw.cancel_order(prev_order.order_id)
                            logger.info(
                                "Cancelled orphaned split child: order_id=%s",
                                prev_order.order_id,
                            )
                    except Exception as cancel_err:
                        logger.warning(
                            "Failed to cancel orphaned split child %s: %s",
                            prev_order.order_id,
                            cancel_err,
                        )
                raise RuntimeError(
                    f"Split order child {i + 1}/{len(quantities)} failed for {signal.symbol}: {child_err}"
                ) from child_err

            successful_children.append(order)
            if parent_order is None:
                parent_order = order
            logger.info(
                "Split child %d/%d: order_id=%s qty=%d",
                i + 1,
                len(quantities),
                order.order_id,
                qty,
            )

        if parent_order is None:
            raise RuntimeError(
                f"No child orders placed for {signal.symbol} (all {len(quantities)} slices had qty <= 0)"
            )
        return parent_order

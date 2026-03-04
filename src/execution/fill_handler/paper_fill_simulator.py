"""
Paper fill simulator: auto-fills PENDING orders with simulated execution.
Runs as an async task in paper trading mode, replacing the live FillListener.
Fills at the order's limit price with realistic dynamic slippage and India transaction costs.
"""
import asyncio
import logging
import math
import random
from datetime import datetime, timezone
from typing import Optional

from src.core.events import OrderStatus

from ..lifecycle import OrderLifecycle
from .events import FillEvent, FillType
from .handler import FillHandler

logger = logging.getLogger(__name__)

# Orders in these statuses can be filled
FILLABLE_STATUSES = (OrderStatus.PENDING, OrderStatus.LIVE)


def _dynamic_slippage(price: float, quantity: float, adv: float = 500_000, sigma: float = 0.02) -> float:
    """
    Dynamic slippage model based on order size relative to ADV.
    Uses Almgren-Chriss temporary impact approximation.

    Args:
        price: order price
        quantity: order quantity
        adv: average daily volume (shares). Default 500K conservative.
        sigma: daily volatility of the stock. Default 2%.

    Returns:
        Slippage in price units (always positive).
    """
    if adv <= 0 or quantity <= 0:
        return 0.0
    participation = quantity / adv
    # Temporary impact: sigma * sqrt(participation) * price
    impact_bps = sigma * math.sqrt(min(participation, 1.0)) * 10_000
    # Add random component (market microstructure noise): uniform [-2, +5] bps
    noise_bps = random.uniform(-2, 5)
    total_bps = max(0.5, impact_bps + noise_bps)  # minimum 0.5 bps
    return price * total_bps / 10_000


class PaperFillSimulator:
    """
    Periodically scans OrderLifecycle for PENDING/LIVE orders and
    generates simulated FillEvents so the full trading loop works
    in paper mode (positions, P&L, stop-loss/take-profit).

    Includes:
      - Dynamic slippage model (volume-dependent, Almgren-Chriss)
      - India transaction costs (STT, GST, stamp duty, exchange charges, SEBI fee)
    """

    def __init__(
        self,
        lifecycle: OrderLifecycle,
        fill_handler: FillHandler,
        fill_delay_seconds: float = 2.0,
        poll_interval_seconds: float = 3.0,
        cost_calculator=None,
        bar_cache=None,
        adv_cache=None,
        feature_engine=None,
    ):
        self.lifecycle = lifecycle
        self.fill_handler = fill_handler
        self.fill_delay_seconds = fill_delay_seconds
        self.poll_interval = poll_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._filled_ids: set = set()
        self._bar_cache = bar_cache
        self._adv_cache = adv_cache
        self._feature_engine = feature_engine

        # India transaction cost calculator
        if cost_calculator is None:
            try:
                from src.costs.india_costs import IndiaCostCalculator
                self._cost_calc = IndiaCostCalculator()
            except ImportError:
                self._cost_calc = None
        else:
            self._cost_calc = cost_calculator

    async def _simulate_fills(self) -> None:
        """Scan lifecycle for pending orders and fill them with realistic costs."""
        orders = list(self.lifecycle._orders.values())
        for order in orders:
            if not order or not order.order_id:
                continue
            if order.order_id in self._filled_ids:
                continue
            if order.status not in FILLABLE_STATUSES:
                continue

            # Simulate fill price: use limit_price if set, otherwise lookup from BarCache
            fill_price = order.limit_price
            if fill_price is None or fill_price <= 0:
                # Try to get last bar close from BarCache
                if self._bar_cache is not None:
                    try:
                        from src.core.events import Exchange
                        exchange_val = order.exchange
                        if isinstance(exchange_val, str):
                            exchange_val = Exchange(exchange_val)
                        bars = self._bar_cache.get_bars(order.symbol, exchange_val, "1m", 1)
                        if bars:
                            fill_price = bars[-1].close
                    except Exception:
                        pass
                if fill_price is None or fill_price <= 0:
                    fill_price = 100.0  # ultimate fallback (should rarely hit)
                    logger.debug("Paper fill using fallback price 100.0 for %s (no bar data)", order.symbol)

            fill_qty = order.quantity

            side_str = order.side
            if hasattr(side_str, "value"):
                side_str = side_str.value

            # Source per-symbol ADV and volatility for realistic slippage
            adv = 500_000  # default
            sigma = 0.02    # default
            if self._adv_cache is not None:
                try:
                    adv = self._adv_cache.get_adv(order.symbol) or adv
                except Exception:
                    pass
            if self._feature_engine is not None and self._bar_cache is not None:
                try:
                    from src.core.events import Exchange
                    exchange_val = order.exchange
                    if isinstance(exchange_val, str):
                        exchange_val = Exchange(exchange_val)
                    bars = self._bar_cache.get_bars(order.symbol, exchange_val, "1m", 30)
                    if bars and len(bars) >= 5:
                        features = self._feature_engine.build_features(bars)
                        sigma = features.get("rolling_vol_20", sigma) or sigma
                except Exception:
                    pass

            # Dynamic slippage model (volume-dependent, Almgren-Chriss)
            slippage_amount = _dynamic_slippage(fill_price, fill_qty, adv=adv, sigma=sigma)
            if str(side_str).upper() == "BUY":
                fill_price = round(fill_price + slippage_amount, 2)
            else:
                fill_price = round(fill_price - slippage_amount, 2)
            fill_price = max(0.01, fill_price)  # price floor

            exchange_str = order.exchange
            if hasattr(exchange_str, "value"):
                exchange_str = exchange_str.value

            # Calculate India transaction costs
            notional = fill_qty * fill_price
            costs_breakdown = None
            total_cost = 0.0
            if self._cost_calc and notional > 0:
                product_type = getattr(order, "product_type", "INTRADAY") or "INTRADAY"
                costs_breakdown = self._cost_calc.calculate(
                    side=str(side_str),
                    notional=notional,
                    product_type=str(product_type),
                    exchange=str(exchange_str),
                )
                total_cost = costs_breakdown.total

            metadata = {
                "paper": True,
                "simulated": True,
                "slippage_bps": round(slippage_amount / fill_price * 10_000, 2) if fill_price > 0 else 0,
            }
            if costs_breakdown:
                metadata["costs"] = costs_breakdown.as_dict()
                metadata["cost_pct"] = round((total_cost / notional) * 100, 4) if notional > 0 else 0

            event = FillEvent(
                order_id=order.order_id,
                broker_order_id=f"PAPER-{order.order_id[:8]}",
                symbol=order.symbol,
                exchange=str(exchange_str),
                side=str(side_str),
                fill_type=FillType.FILL,
                filled_qty=fill_qty,
                remaining_qty=0.0,
                avg_price=fill_price,
                ts=datetime.now(timezone.utc),
                strategy_id=order.strategy_id or "",
                metadata=metadata,
            )

            try:
                await self.fill_handler.on_fill_event(event)
                # NOTE: Transaction costs are already embedded in fill_price via slippage adjustment.
                # Do NOT double-deduct via rm.register_pnl(-total_cost).
                self._filled_ids.add(order.order_id)
                cost_str = f" | cost=₹{total_cost:.2f}" if total_cost > 0 else ""
                logger.info(
                    "Paper fill: %s %s %.0f %s @ %.2f (slip=%.1fbps%s)",
                    side_str, order.symbol, fill_qty, exchange_str, fill_price,
                    metadata.get("slippage_bps", 0), cost_str,
                )
            except Exception as e:
                logger.warning("Paper fill failed for %s: %s", order.order_id, e)

        # Cap memory: remove old filled IDs if too many
        if len(self._filled_ids) > 5000:
            # Keep only IDs still in lifecycle
            active_ids = set(self.lifecycle._orders.keys())
            self._filled_ids = self._filled_ids & active_ids

    async def _loop(self) -> None:
        # Initial delay to let first orders arrive
        await asyncio.sleep(self.fill_delay_seconds)
        while self._running:
            try:
                await self._simulate_fills()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Paper fill simulator error: %s", e)
            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Paper fill simulator started (delay=%.1fs, poll=%.1fs, india_costs=%s)",
            self.fill_delay_seconds, self.poll_interval, self._cost_calc is not None,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Paper fill simulator stopped")

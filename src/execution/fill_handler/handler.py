"""
Fill handling: broker WebSocket → FillHandler.
On fill: update OrderLifecycle, risk_manager.add_or_merge_position (merge by symbol+exchange+side), persist, emit metric.
On reject/cancel: update lifecycle and persist status.
"""

import asyncio
import logging
from collections.abc import Callable

from src.core.events import Exchange, OrderStatus, Position, SignalSide
from src.risk_engine import RiskManager

from ..lifecycle import OrderLifecycle
from .events import FillEvent, FillType

logger = logging.getLogger(__name__)


class FillHandler:
    """
    Process fill events from broker. Updates lifecycle, risk positions, and PnL.
    Caller (broker adapter) must push FillEvent when WebSocket or poll receives fill.
    When order_lock is provided, hold it when updating risk_manager to avoid race with order entry.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        lifecycle: OrderLifecycle,
        *,
        on_fill_persist: Callable | None = None,
        on_fill_metric: Callable | None = None,
        order_lock: asyncio.Lock | None = None,
        on_equity_after_fill: Callable[[], float] | None = None,
        on_fill_callback: Callable[[FillEvent], None] | None = None,
    ):
        self.risk_manager = risk_manager
        self.lifecycle = lifecycle
        self.on_fill_persist = on_fill_persist
        self.on_fill_metric = on_fill_metric
        self._order_lock = order_lock
        self.on_equity_after_fill = on_equity_after_fill
        self.on_fill_callback = on_fill_callback

    async def _run_persist(self, event: FillEvent) -> None:
        if not self.on_fill_persist:
            return
        try:
            await self.on_fill_persist(event)
        except Exception as e:
            try:
                from src.monitoring.metrics import track_orders_fill_persist_failed_total

                track_orders_fill_persist_failed_total()
            except Exception:
                pass
            logger.exception("Persist fill failed: %s", e)

    async def on_fill_event(self, event: FillEvent) -> None:
        """Process one fill event."""
        if event.fill_type == FillType.REJECT:
            await self.lifecycle.update_status(event.order_id, OrderStatus.REJECTED, filled_qty=0, avg_price=None)
            await self._run_persist(event)
            if self.on_fill_metric:
                self.on_fill_metric("reject", event)
            return

        if event.fill_type == FillType.CANCEL:
            await self.lifecycle.update_status(
                event.order_id, OrderStatus.CANCELLED, filled_qty=event.filled_qty, avg_price=event.avg_price
            )
            await self._run_persist(event)
            if self.on_fill_metric:
                self.on_fill_metric("cancel", event)
            return

        await self.lifecycle.update_status(
            event.order_id,
            OrderStatus.FILLED if event.fill_type == FillType.FILL else OrderStatus.PARTIALLY_FILLED,
            filled_qty=event.filled_qty,
            avg_price=event.avg_price,
        )

        if event.filled_qty > 0 and event.avg_price is not None:
            side = SignalSide.BUY if event.side.upper() == "BUY" else SignalSide.SELL
            try:
                exch = Exchange(event.exchange) if isinstance(event.exchange, str) else event.exchange
            except ValueError:
                exch = Exchange.NSE
            pos = Position(
                symbol=event.symbol,
                exchange=exch,
                side=side,
                quantity=event.filled_qty,
                avg_price=event.avg_price,
                unrealized_pnl=0.0,
                strategy_id=event.strategy_id or None,
            )
            if self._order_lock:
                async with self._order_lock:
                    self.risk_manager.add_or_merge_position(pos)
            else:
                self.risk_manager.add_or_merge_position(pos)
            if self.on_equity_after_fill:
                try:
                    eq = self.on_equity_after_fill()
                    if eq is not None:
                        self.risk_manager.update_equity(float(eq))
                except Exception as e:
                    logger.debug("Equity update on fill failed: %s", e)

        await self._run_persist(event)
        if self.on_fill_metric:
            self.on_fill_metric("fill", event)
        if self.on_fill_callback and event.fill_type == FillType.FILL:
            try:
                self.on_fill_callback(event)
            except Exception as e:
                logger.debug("on_fill_callback failed: %s", e)

    async def on_close_position(
        self, symbol: str, exchange: str, side: str, closed_qty: float, realized_pnl: float
    ) -> None:
        """Called when a position is closed (e.g. SELL to close long). Update risk and PnL. side: BUY or SELL."""
        if self._order_lock:
            async with self._order_lock:
                self.risk_manager.remove_position(symbol, exchange, side)
                self.risk_manager.register_pnl(realized_pnl)
        else:
            self.risk_manager.remove_position(symbol, exchange, side)
            self.risk_manager.register_pnl(realized_pnl)

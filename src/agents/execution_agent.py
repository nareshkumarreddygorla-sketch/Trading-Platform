"""
Execution Agent: receives signals from Research Agent,
optimizes entry timing, submits via OrderEntryService.
"""
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """
    Smart order execution agent:
    - Receives opportunities from Research Agent
    - Optimizes entry timing (waits for favorable price)
    - Splits large orders across bars (TWAP-like)
    - Submits via OrderEntryService (single-pipe)
    """

    name = "execution_agent"
    description = "Smart order execution with timing optimization"

    def __init__(
        self,
        submit_order_fn: Optional[Callable[..., Awaitable]] = None,
        get_bars: Optional[Callable] = None,
        get_positions: Optional[Callable] = None,
        max_concurrent_orders: int = 5,
        execution_interval: float = 15.0,
    ):
        super().__init__()
        self._submit_order_fn = submit_order_fn
        self._get_bars = get_bars
        self._get_positions = get_positions
        self._max_concurrent = max_concurrent_orders
        self._execution_interval = execution_interval
        self._pending_opportunities: List[Dict[str, Any]] = []
        self._executed_count = 0
        self._rejected_count = 0

    @property
    def interval_seconds(self) -> float:
        return self._execution_interval

    def _is_favorable_entry(self, opportunity: Dict, current_price: float) -> bool:
        """Check if current price is favorable for entry."""
        signal_price = opportunity.get("price", 0)
        direction = opportunity.get("direction", "BUY")

        if signal_price <= 0:
            return True  # No reference price, proceed

        # For BUY: want to enter at or below signal price
        # For SELL: want to enter at or above signal price
        price_diff_pct = (current_price - signal_price) / signal_price * 100

        if direction == "BUY":
            return price_diff_pct <= 0.3  # Allow 0.3% above signal price
        else:
            return price_diff_pct >= -0.3

    async def run_cycle(self) -> None:
        # Process incoming messages
        msg = await self.receive_message()
        while msg is not None:
            if msg.msg_type == "opportunities":
                new_opps = msg.payload.get("opportunities", [])
                self._pending_opportunities = new_opps[:self._max_concurrent * 2]
                logger.debug("ExecutionAgent: received %d opportunities", len(new_opps))

            elif msg.msg_type == "risk_alert":
                severity = msg.payload.get("severity", "")
                if severity == "critical":
                    self._pending_opportunities.clear()
                    logger.warning("ExecutionAgent: cleared pending orders (critical risk alert)")

            msg = await self.receive_message()

        if not self._submit_order_fn or not self._pending_opportunities:
            return

        # Snapshot the list to avoid race condition with incoming messages
        work_queue = list(self._pending_opportunities)
        self._pending_opportunities.clear()

        # Check existing positions to avoid duplicates
        existing_symbols = set()
        if self._get_positions:
            positions = self._get_positions()
            existing_symbols = {p.symbol for p in positions}

        executed = 0
        remaining = []

        for opp in work_queue:
            if executed >= self._max_concurrent:
                remaining.append(opp)
                continue

            symbol = opp.get("symbol", "")
            exchange_str = opp.get("exchange", "NSE")
            direction = opp.get("direction", "BUY")
            confidence = opp.get("confidence", 0)
            price = opp.get("price", 0)

            # Skip if already have position
            if symbol in existing_symbols:
                continue

            # Check entry timing
            if self._get_bars:
                try:
                    from src.core.events import Exchange
                    exchange = Exchange(exchange_str) if isinstance(exchange_str, str) else exchange_str
                    bars = self._get_bars(symbol, exchange, "1m", 5)
                    if bars:
                        current_price = bars[-1].close
                        if current_price <= 0 or current_price != current_price:  # NaN check
                            remaining.append(opp)
                            continue
                        if not self._is_favorable_entry(opp, current_price):
                            remaining.append(opp)
                            continue
                        price = current_price
                except Exception:
                    pass

            # Validate price before submitting
            if price <= 0 or price != price:  # NaN check
                logger.debug("ExecutionAgent: invalid price %.4f for %s, skipping", price, symbol)
                continue

            # Build and submit order
            try:
                from src.core.events import Exchange, Signal, SignalSide, OrderType
                from src.execution.order_entry.request import OrderEntryRequest
                from src.execution.order_entry.idempotency import IdempotencyStore
                from datetime import datetime, timezone

                exchange = Exchange(exchange_str) if isinstance(exchange_str, str) else exchange_str
                side = SignalSide.BUY if direction == "BUY" else SignalSide.SELL

                # Calculate quantity from portfolio weight and price
                # HARD CAP at 4.5% to stay well within risk manager's 5% limit
                max_weight = 0.045
                portfolio_weight = min(max_weight, confidence * 0.08)
                if price > 0:
                    # Use equity from risk manager if available, else default
                    equity = 100_000.0
                    if self._get_positions:
                        try:
                            # Try to get equity from risk manager via positions callback
                            positions_list = self._get_positions()
                            if hasattr(positions_list, '__self__') and hasattr(positions_list.__self__, 'equity'):
                                equity = positions_list.__self__.equity
                        except Exception:
                            pass
                    quantity = max(1, int(equity * portfolio_weight / price))
                    # Double-check: enforce hard max_position_pct cap
                    max_qty = int(equity * max_weight / price)
                    quantity = min(quantity, max(1, max_qty))
                else:
                    quantity = 1  # Minimum order

                signal = Signal(
                    strategy_id="execution_agent",
                    symbol=symbol,
                    exchange=exchange,
                    side=side,
                    score=confidence,
                    price=price,
                    portfolio_weight=portfolio_weight,
                    risk_level="NORMAL",
                )

                bar_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                idem_key = IdempotencyStore.derive_key_bar_stable(
                    bar_ts, "execution_agent", symbol, direction,
                )

                req = OrderEntryRequest(
                    signal=signal,
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    limit_price=price,
                    idempotency_key=idem_key,
                    source="execution_agent",
                )

                result = await self._submit_order_fn(req)
                if result.success:
                    self._executed_count += 1
                    executed += 1
                    existing_symbols.add(symbol)
                    logger.info("ExecutionAgent: order submitted %s %s @ %.2f (conf=%.2f)",
                                direction, symbol, price, confidence)
                else:
                    self._rejected_count += 1
                    logger.debug("ExecutionAgent: order rejected %s %s: %s",
                                 direction, symbol, result.reject_reason)

            except Exception as e:
                logger.debug("ExecutionAgent: submit failed for %s: %s", symbol, e)

        # Re-queue unprocessed opportunities
        self._pending_opportunities = remaining

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["pending_opportunities"] = len(self._pending_opportunities)
        status["executed_count"] = self._executed_count
        status["rejected_count"] = self._rejected_count
        return status

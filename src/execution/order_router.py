"""Order routing: select gateway (Angel One / FIX), apply price improvement logic.
Includes NSE tick size validation, minimum order value check, and freeze qty splitting."""
import asyncio
import logging
import math
from typing import List, Optional

from src.core.events import Order, OrderType, Signal

from .base import BaseExecutionGateway
from .angel_one_gateway import AngelOneExecutionGateway

logger = logging.getLogger(__name__)

# NSE minimum order value (INR)
_MIN_ORDER_VALUE_INR = 1000.0

# NSE price limits for sanity check
_MIN_PRICE_INR = 1.0
_MAX_PRICE_INR = 100_000.0


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


class OrderRouter:
    """
    Route orders to appropriate gateway by exchange/venue.
    Supports limit/market, IOC/FOK; optional price improvement (e.g. limit slightly better than signal).
    Includes NSE tick size validation and freeze qty splitting.
    """

    def __init__(self, default_gateway: BaseExecutionGateway, freeze_qty_manager=None):
        self.default_gateway = default_gateway
        self._gateways: dict[str, BaseExecutionGateway] = {"NSE": default_gateway, "BSE": default_gateway}
        self._freeze_qty_manager = freeze_qty_manager

    def register_gateway(self, exchange: str, gateway: BaseExecutionGateway) -> None:
        self._gateways[exchange] = gateway

    def _gateway(self, exchange: str) -> BaseExecutionGateway:
        return self._gateways.get(exchange, self.default_gateway)

    def _price_improvement(self, price: float, side: str, ticks: int = 1) -> float:
        """Optional: improve limit by ticks (buy lower, sell higher)."""
        tick = 0.05 if price >= 100 else 0.01
        if side == "BUY":
            return round(price - ticks * tick, 2)
        return round(price + ticks * tick, 2)

    async def place_order(
        self,
        signal: Signal,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
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
                    logger.debug("Tick size adjusted: %.4f -> %.2f for %s", original_price, price, signal.symbol)

            # Price sanity check
            if price is not None and not validate_price_sanity(price):
                raise ValueError(f"Price {price} outside valid range [{_MIN_PRICE_INR}, {_MAX_PRICE_INR}] for {signal.symbol}")

            if price_improvement_ticks and order_type == OrderType.LIMIT:
                price = self._price_improvement(price, signal.side.value, price_improvement_ticks)

        # Minimum order value check
        if price is not None and price > 0:
            order_value = quantity * price
            if order_value < _MIN_ORDER_VALUE_INR:
                raise ValueError(f"Order value ₹{order_value:.0f} below minimum ₹{_MIN_ORDER_VALUE_INR:.0f}")

        # Freeze qty check: split order if above exchange limit
        if self._freeze_qty_manager is not None and quantity > 0:
            try:
                split_result = self._freeze_qty_manager.check_and_split(signal.symbol, quantity)
                if split_result and len(split_result) > 1:
                    logger.info("Freeze qty split: %s %d -> %d child orders", signal.symbol, quantity, len(split_result))
                    return await self._submit_split_orders(signal, split_result, order_type, price)
            except Exception as e:
                logger.warning("Freeze qty check failed for %s (proceeding with full qty): %s", signal.symbol, e)

        gateway = self._gateway(exchange)
        if exchange not in self._gateways:
            logger.debug("Unknown exchange %s, using default gateway", exchange)
        return await gateway.place_order(
            symbol=signal.symbol,
            exchange=exchange,
            side=signal.side.value,
            quantity=float(quantity),
            order_type=order_type.value,
            limit_price=price,
            strategy_id=signal.strategy_id,
        )

    async def _submit_split_orders(
        self,
        signal: Signal,
        quantities: List[int],
        order_type: OrderType,
        price: Optional[float],
    ) -> Order:
        """Submit child orders with 1s delay between them. Return first order as parent."""
        exchange = signal.exchange.value
        gateway = self._gateway(exchange)
        parent_order = None

        for i, qty in enumerate(quantities):
            if qty <= 0:
                continue
            if i > 0:
                await asyncio.sleep(1.0)  # Rate limit between child orders

            order = await gateway.place_order(
                symbol=signal.symbol,
                exchange=exchange,
                side=signal.side.value,
                quantity=float(qty),
                order_type=order_type.value,
                limit_price=price,
                strategy_id=signal.strategy_id,
            )
            if parent_order is None:
                parent_order = order
            logger.info("Freeze qty child %d/%d: order_id=%s qty=%d", i + 1, len(quantities), order.order_id, qty)

        return parent_order

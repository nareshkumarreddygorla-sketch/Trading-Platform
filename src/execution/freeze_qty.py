"""
NSE freeze quantity handling.

NSE enforces maximum order quantity limits (freeze quantities) per stock.
Orders exceeding the freeze quantity are rejected by the exchange.
This module:
  - Fetches freeze quantity data from NSE (with caching)
  - Checks orders against limits
  - Splits large orders into compliant child orders
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Conservative default freeze qty for unknown stocks
DEFAULT_FREEZE_QTY = 100

# Well-known NSE freeze quantities (updated periodically from NSE circular)
# These are approximate — actual values should be fetched from NSE
KNOWN_FREEZE_QTY: dict[str, int] = {
    "RELIANCE": 1800,
    "TCS": 1500,
    "HDFCBANK": 1800,
    "INFY": 2400,
    "ICICIBANK": 2800,
    "HINDUNILVR": 1500,
    "SBIN": 3000,
    "BHARTIARTL": 1800,
    "ITC": 3200,
    "KOTAKBANK": 1800,
    "LT": 1500,
    "AXISBANK": 2400,
    "BAJFINANCE": 500,
    "ASIANPAINT": 600,
    "MARUTI": 400,
    "TITAN": 600,
    "SUNPHARMA": 2800,
    "NESTLEIND": 200,
    "ULTRACEMCO": 600,
    "WIPRO": 3200,
    "HCLTECH": 1400,
    "TATAMOTORS": 5700,
    "TATASTEEL": 6400,
    "POWERGRID": 10800,
    "NTPC": 5700,
    "ONGC": 5700,
    "JSWSTEEL": 4000,
    "ADANIENT": 500,
    "ADANIPORTS": 3200,
    "COALINDIA": 6400,
    "DIVISLAB": 400,
    "DRREDDY": 400,
    "CIPLA": 1600,
    "GRASIM": 800,
    "EICHERMOT": 400,
    "BRITANNIA": 400,
    "BAJAJFINSV": 500,
    "BAJAJ-AUTO": 400,
    "HEROMOTOCO": 400,
    "M&M": 1400,
    "TECHM": 1400,
    "INDUSINDBK": 1800,
    "NIFTY": 1800,  # Nifty options
    "BANKNIFTY": 900,
}


@dataclass
class SplitOrder:
    """A child order from splitting a large order."""

    symbol: str
    side: str
    quantity: int
    parent_order_id: str
    child_index: int
    total_children: int
    delay_seconds: float = 1.0  # delay between child orders


class FreezeQuantityManager:
    """
    Manages NSE freeze quantity compliance.

    Usage:
        fqm = FreezeQuantityManager()
        if fqm.exceeds_limit("RELIANCE", 3000):
            children = fqm.split_order("RELIANCE", "BUY", 3000, "order-123")
            # Submit children with delays
    """

    def __init__(
        self,
        cache_ttl_hours: float = 24.0,
        inter_order_delay_seconds: float = 1.0,
    ):
        self._cache: dict[str, int] = dict(KNOWN_FREEZE_QTY)
        self._cache_ts: float = _time.time()
        self._cache_ttl = cache_ttl_hours * 3600
        self._inter_order_delay = inter_order_delay_seconds

    def get_freeze_qty(self, symbol: str) -> int:
        """Get freeze quantity for a symbol."""
        symbol = symbol.upper().strip()
        if symbol in self._cache:
            return self._cache[symbol]
        return DEFAULT_FREEZE_QTY

    def set_freeze_qty(self, symbol: str, qty: int) -> None:
        """Manually set freeze quantity for a symbol (e.g., from NSE data)."""
        self._cache[symbol.upper().strip()] = qty

    def update_from_dict(self, data: dict[str, int]) -> None:
        """Bulk update freeze quantities from a dict."""
        for sym, qty in data.items():
            self._cache[sym.upper().strip()] = qty
        self._cache_ts = _time.time()
        logger.info("Freeze quantities updated: %d symbols", len(data))

    def exceeds_limit(self, symbol: str, quantity: int) -> bool:
        """Check if an order quantity exceeds the freeze limit."""
        return quantity > self.get_freeze_qty(symbol)

    def split_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        parent_order_id: str,
    ) -> list[SplitOrder]:
        """
        Split a large order into compliant child orders.

        Args:
            symbol: stock symbol
            side: BUY or SELL
            quantity: total quantity to fill
            parent_order_id: parent order ID for tracking

        Returns:
            List of SplitOrder objects, each within freeze limit
        """
        freeze_qty = self.get_freeze_qty(symbol)
        if quantity <= freeze_qty:
            return [
                SplitOrder(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    parent_order_id=parent_order_id,
                    child_index=0,
                    total_children=1,
                    delay_seconds=0,
                )
            ]

        children = []
        remaining = quantity
        idx = 0
        while remaining > 0:
            child_qty = min(remaining, freeze_qty)
            children.append(
                SplitOrder(
                    symbol=symbol,
                    side=side,
                    quantity=child_qty,
                    parent_order_id=parent_order_id,
                    child_index=idx,
                    total_children=0,  # updated below
                    delay_seconds=self._inter_order_delay * idx,
                )
            )
            remaining -= child_qty
            idx += 1

        # Update total_children
        for c in children:
            c.total_children = len(children)

        logger.info(
            "Order split: %s %s %d → %d child orders (freeze_qty=%d)",
            side,
            symbol,
            quantity,
            len(children),
            freeze_qty,
        )
        return children

    def validate_and_split(
        self,
        symbol: str,
        side: str,
        quantity: int,
        parent_order_id: str,
    ) -> tuple[bool, list[SplitOrder]]:
        """
        Validate order against freeze qty; split if needed.

        Returns:
            (needs_split, children)
        """
        if not self.exceeds_limit(symbol, quantity):
            return False, [
                SplitOrder(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    parent_order_id=parent_order_id,
                    child_index=0,
                    total_children=1,
                    delay_seconds=0,
                )
            ]
        return True, self.split_order(symbol, side, quantity, parent_order_id)

    def is_cache_stale(self) -> bool:
        """Check if freeze qty cache is stale."""
        return (_time.time() - self._cache_ts) > self._cache_ttl

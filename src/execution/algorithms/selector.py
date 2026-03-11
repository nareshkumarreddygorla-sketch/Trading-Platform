"""
Algorithm Selector: automatically choose the best execution algorithm
based on order size relative to Average Daily Volume (ADV).

Routing rules:
  < 1% ADV  -> Direct LIMIT order
  1-5% ADV  -> TWAP (time-weighted)
  5-10% ADV -> VWAP (volume-weighted)
  > 10% ADV -> Iceberg + VWAP hybrid
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from src.execution.algorithms.iceberg import IcebergAlgorithm, IcebergConfig
from src.execution.algorithms.twap import TWAPAlgorithm, TWAPConfig
from src.execution.algorithms.vwap import VWAPAlgorithm, VWAPConfig

logger = logging.getLogger(__name__)


class AlgorithmSelector:
    """
    Selects and executes the optimal order algorithm based on order characteristics.
    """

    def __init__(
        self,
        submit_order_fn: Callable[..., Coroutine],
        get_adv_fn: Callable | None = None,
        get_price_fn: Callable | None = None,
    ):
        """
        Args:
            submit_order_fn: Async function to submit a single order.
            get_adv_fn: Optional fn(symbol) -> average daily volume.
            get_price_fn: Optional async fn(symbol) -> current market price.
        """
        self._submit = submit_order_fn
        self._get_adv = get_adv_fn
        self._get_price = get_price_fn
        self._twap = TWAPAlgorithm(submit_order_fn)
        self._vwap = VWAPAlgorithm(submit_order_fn)
        self._iceberg = IcebergAlgorithm(submit_order_fn)

    def select_algorithm(
        self,
        symbol: str,
        quantity: int,
        adv: int | None = None,
    ) -> str:
        """
        Select the best algorithm for an order.

        Returns: "direct", "twap", "vwap", or "iceberg"
        """
        if adv is None and self._get_adv:
            try:
                adv = self._get_adv(symbol)
            except Exception:
                adv = None

        if adv is None or adv <= 0:
            # Unknown ADV, play it safe
            if quantity > 1000:
                return "twap"
            return "direct"

        participation = quantity / adv * 100  # percentage

        if participation < 1:
            return "direct"
        elif participation < 5:
            return "twap"
        elif participation < 10:
            return "vwap"
        else:
            return "iceberg"

    async def execute_smart(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange: str = "NSE",
        adv: int | None = None,
        force_algorithm: str | None = None,
    ) -> dict[str, Any]:
        """
        Smart execution: auto-select algorithm and execute.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Total order quantity
            exchange: Exchange code
            adv: Average daily volume (optional, will be fetched if None)
            force_algorithm: Override auto-selection

        Returns:
            Dict with execution details: algo, exec_id, filled, vwap, status
        """
        algo = force_algorithm or self.select_algorithm(symbol, quantity, adv)

        logger.info(
            "Smart execution: %s %s x%d -> algorithm=%s (ADV=%s)",
            side,
            symbol,
            quantity,
            algo,
            adv or "unknown",
        )

        if algo == "direct":
            # Direct limit order
            try:
                order_id = await self._submit(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="LIMIT",
                    limit_price=None,
                    exchange=exchange,
                )
                return {
                    "algorithm": "direct",
                    "order_id": order_id,
                    "filled": quantity,
                    "status": "SUBMITTED",
                }
            except Exception as e:
                return {"algorithm": "direct", "error": str(e), "status": "FAILED"}

        elif algo == "twap":
            config = TWAPConfig(
                total_quantity=quantity,
                symbol=symbol,
                side=side,
                exchange=exchange,
                duration_minutes=max(10, min(60, quantity // 100)),
                num_slices=min(20, max(5, quantity // 50)),
            )
            execution = self._twap.create_schedule(config)
            result = await self._twap.execute(execution, self._get_price)
            return {
                "algorithm": "twap",
                "exec_id": result.exec_id,
                "filled": result.total_filled,
                "vwap": result.vwap,
                "slices": len(result.slices),
                "status": result.status,
            }

        elif algo == "vwap":
            config = VWAPConfig(
                total_quantity=quantity,
                symbol=symbol,
                side=side,
                exchange=exchange,
            )
            execution = self._vwap.create_schedule(config)
            result = await self._vwap.execute(execution, self._get_price)
            return {
                "algorithm": "vwap",
                "exec_id": result.exec_id,
                "filled": result.total_filled,
                "vwap": result.achieved_vwap,
                "slices": len(result.slices),
                "status": result.status,
            }

        elif algo == "iceberg":
            config = IcebergConfig(
                total_quantity=quantity,
                symbol=symbol,
                side=side,
                exchange=exchange,
                display_pct=8.0,  # Show 8% at a time
            )
            execution = self._iceberg.create_execution(config)
            result = await self._iceberg.execute(execution, self._get_price)
            return {
                "algorithm": "iceberg",
                "exec_id": result.exec_id,
                "filled": result.total_filled,
                "avg_price": result.avg_fill_price,
                "replenishments": result.replenish_count,
                "status": result.status,
            }

        return {"algorithm": algo, "error": "unknown algorithm", "status": "FAILED"}

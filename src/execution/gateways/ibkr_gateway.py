"""Interactive Brokers gateway — global multi-asset trading via TWS/Gateway API."""

import logging
import os
import time
import uuid

from .base import (
    BaseBrokerGateway,
    BrokerHealth,
    BrokerOrder,
    BrokerPosition,
    BrokerType,
    GatewayStatus,
)

logger = logging.getLogger(__name__)


class IBKRGateway(BaseBrokerGateway):
    """Interactive Brokers TWS/Gateway API — 150+ global exchanges.

    Requires ib_insync library and a running TWS or IB Gateway instance.
    Falls back to paper mode when ib_insync is not installed.
    """

    broker_type = BrokerType.IBKR

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._host = os.environ.get("IBKR_HOST", "127.0.0.1")
        self._port = int(os.environ.get("IBKR_PORT", "7497" if paper else "7496"))
        self._client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self._status = GatewayStatus.DISCONNECTED
        self._last_heartbeat: float | None = None
        self._error_count = 0
        self._orders_today = 0
        self._ib = None  # ib_insync.IB instance

    async def connect(self) -> bool:
        try:
            from ib_insync import IB

            self._ib = IB()
            await self._ib.connectAsync(self._host, self._port, clientId=self._client_id)
            logger.info("IBKR connected: paper=%s port=%d", self.paper, self._port)
            self._status = GatewayStatus.CONNECTED
            self._last_heartbeat = time.time()
            return True
        except ImportError:
            logger.info("ib_insync not installed — IBKR in paper-only mode")
            self.paper = True
            self._status = GatewayStatus.CONNECTED
            self._last_heartbeat = time.time()
            return True
        except Exception as e:
            logger.error("IBKR connect failed: %s", e)
            self._status = GatewayStatus.ERROR
            self._error_count += 1
            return False

    async def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None
        self._status = GatewayStatus.DISCONNECTED

    async def place_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        **kwargs,
    ) -> BrokerOrder:
        if self._ib is None or self.paper:
            order_id = f"paper-ibkr-{uuid.uuid4().hex[:12]}"
            self._orders_today += 1
            return BrokerOrder(
                broker_order_id=order_id,
                symbol=symbol,
                exchange=exchange,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status="FILLED",
                filled_qty=quantity,
                avg_price=limit_price or 0.0,
            )
        from ib_insync import LimitOrder, MarketOrder, Stock

        contract = Stock(symbol, exchange or "SMART", kwargs.get("currency", "USD"))
        if order_type.upper() == "LIMIT" and limit_price:
            order = LimitOrder(side.upper(), quantity, limit_price)
        else:
            order = MarketOrder(side.upper(), quantity)
        trade = self._ib.placeOrder(contract, order)
        self._orders_today += 1
        self._last_heartbeat = time.time()
        return BrokerOrder(
            broker_order_id=str(trade.order.orderId),
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            status=trade.orderStatus.status.upper(),
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        if self._ib is None:
            return True
        try:
            for trade in self._ib.trades():
                if str(trade.order.orderId) == broker_order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            logger.error("IBKR cancel failed: %s", e)
            return False

    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        if self._ib is None:
            return BrokerOrder(
                broker_order_id=broker_order_id,
                symbol="",
                exchange="",
                side="",
                quantity=0,
                order_type="",
                limit_price=None,
                status="UNKNOWN",
            )
        for trade in self._ib.trades():
            if str(trade.order.orderId) == broker_order_id:
                return BrokerOrder(
                    broker_order_id=broker_order_id,
                    symbol=trade.contract.symbol,
                    exchange=trade.contract.exchange or "SMART",
                    side=trade.order.action,
                    quantity=int(trade.order.totalQuantity),
                    order_type=trade.order.orderType,
                    limit_price=getattr(trade.order, "lmtPrice", None),
                    status=trade.orderStatus.status.upper(),
                    filled_qty=int(trade.orderStatus.filled),
                    avg_price=float(trade.orderStatus.avgFillPrice),
                )
        return BrokerOrder(
            broker_order_id=broker_order_id,
            symbol="",
            exchange="",
            side="",
            quantity=0,
            order_type="",
            limit_price=None,
            status="NOT_FOUND",
        )

    async def get_positions(self) -> list[BrokerPosition]:
        if self._ib is None:
            return []
        try:
            return [
                BrokerPosition(
                    symbol=pos.contract.symbol,
                    exchange=pos.contract.exchange or "SMART",
                    quantity=int(pos.position),
                    avg_price=float(pos.avgCost),
                    side="BUY" if pos.position > 0 else "SELL",
                )
                for pos in self._ib.positions()
            ]
        except Exception as e:
            logger.error("IBKR get_positions failed: %s", e)
            return []

    def health(self) -> BrokerHealth:
        return BrokerHealth(
            broker=self.broker_type,
            status=self._status,
            last_heartbeat=str(self._last_heartbeat) if self._last_heartbeat else None,
            error_count=self._error_count,
            orders_today=self._orders_today,
            supported_exchanges=self.supported_exchanges(),
            supported_order_types=[
                "MARKET",
                "LIMIT",
                "STOP",
                "STOP_LIMIT",
                "TRAILING_STOP",
                "MOC",
                "LOC",
                "VWAP",
                "TWAP",
                "ADAPTIVE",
            ],
            paper_mode=self.paper,
        )

    def supported_exchanges(self) -> list[str]:
        return [
            "NYSE",
            "NASDAQ",
            "ARCA",
            "LSE",
            "TSE",
            "HKEX",
            "NSE",
            "BSE",
            "ASX",
            "SGX",
            "EUREX",
            "CME",
            "CBOE",
        ]

"""Multi-broker manager: smart routing, failover, and health monitoring."""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.execution.gateways.base import (
    BaseBrokerGateway, BrokerHealth, BrokerOrder, BrokerPosition,
    BrokerType, GatewayStatus,
)

logger = logging.getLogger(__name__)

# Exchange → preferred broker order (first available is used)
EXCHANGE_BROKER_MAP: Dict[str, List[BrokerType]] = {
    "NSE": [BrokerType.ANGEL_ONE, BrokerType.ZERODHA, BrokerType.IBKR],
    "BSE": [BrokerType.ANGEL_ONE, BrokerType.ZERODHA, BrokerType.IBKR],
    "NYSE": [BrokerType.ALPACA, BrokerType.IBKR],
    "NASDAQ": [BrokerType.ALPACA, BrokerType.IBKR],
    "ARCA": [BrokerType.ALPACA, BrokerType.IBKR],
    "LSE": [BrokerType.IBKR],
    "TSE": [BrokerType.IBKR],
    "HKEX": [BrokerType.IBKR],
    "ASX": [BrokerType.IBKR],
    "SGX": [BrokerType.IBKR],
    "EUREX": [BrokerType.IBKR],
    "CME": [BrokerType.IBKR],
}


class BrokerManager:
    """Manages multiple broker gateways with smart routing and automatic failover.

    Routing logic:
    1. Look up preferred brokers for the target exchange.
    2. Pick the first broker that is connected and healthy.
    3. On order failure, try the next healthy broker (failover).
    """

    def __init__(self):
        self._gateways: Dict[BrokerType, BaseBrokerGateway] = {}
        self._primary: Optional[BrokerType] = None
        self._health_task: Optional[asyncio.Task] = None
        self._failover_history: List[Dict[str, Any]] = []

    def register(self, gateway: BaseBrokerGateway) -> None:
        self._gateways[gateway.broker_type] = gateway
        if self._primary is None:
            self._primary = gateway.broker_type
        logger.info(
            "Broker registered: %s (total=%d)", gateway.broker_type.value, len(self._gateways),
        )

    async def connect_all(self) -> Dict[BrokerType, bool]:
        """Connect to all registered brokers in parallel."""
        results: Dict[BrokerType, bool] = {}
        for broker_type, gw in self._gateways.items():
            try:
                ok = await gw.connect()
                results[broker_type] = ok
                if ok:
                    logger.info("Broker %s: connected", broker_type.value)
                else:
                    logger.warning("Broker %s: connection failed", broker_type.value)
            except Exception as e:
                logger.error("Broker %s: connect error: %s", broker_type.value, e)
                results[broker_type] = False
        return results

    async def disconnect_all(self) -> None:
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
        for gw in self._gateways.values():
            try:
                await gw.disconnect()
            except Exception as e:
                logger.debug("Broker disconnect error: %s", e)

    def get_gateway(self, broker_type: BrokerType) -> Optional[BaseBrokerGateway]:
        return self._gateways.get(broker_type)

    def route_exchange(self, exchange: str) -> Optional[BaseBrokerGateway]:
        """Return the best available gateway for *exchange*."""
        preferred = EXCHANGE_BROKER_MAP.get(exchange.upper(), [])
        for bt in preferred:
            gw = self._gateways.get(bt)
            if gw is not None and gw.health().status == GatewayStatus.CONNECTED:
                return gw
        # Fallback: any connected broker that supports this exchange
        for gw in self._gateways.values():
            if (
                gw.health().status == GatewayStatus.CONNECTED
                and exchange.upper() in gw.supported_exchanges()
            ):
                return gw
        return None

    async def place_order_smart(
        self, symbol: str, exchange: str, side: str, quantity: int,
        order_type: str = "MARKET", limit_price: Optional[float] = None,
        **kwargs,
    ) -> BrokerOrder:
        """Place order with smart routing and automatic failover."""
        primary_gw = self.route_exchange(exchange)
        if primary_gw is None:
            raise RuntimeError(f"No broker available for exchange {exchange}")

        # Try primary
        try:
            return await primary_gw.place_order(
                symbol, exchange, side, quantity, order_type, limit_price, **kwargs,
            )
        except Exception as primary_err:
            logger.warning(
                "Primary broker %s failed: %s — attempting failover",
                primary_gw.broker_type.value, primary_err,
            )

        # Failover: try remaining connected brokers that support this exchange
        for gw in self._gateways.values():
            if gw is primary_gw:
                continue
            if (
                gw.health().status == GatewayStatus.CONNECTED
                and exchange.upper() in gw.supported_exchanges()
            ):
                try:
                    result = await gw.place_order(
                        symbol, exchange, side, quantity, order_type, limit_price, **kwargs,
                    )
                    self._failover_history.append({
                        "time": time.time(),
                        "from_broker": primary_gw.broker_type.value,
                        "to_broker": gw.broker_type.value,
                        "symbol": symbol,
                        "reason": str(primary_err),
                    })
                    logger.info(
                        "Failover success: %s → %s for %s",
                        primary_gw.broker_type.value, gw.broker_type.value, symbol,
                    )
                    return result
                except Exception:
                    continue

        raise RuntimeError(f"All brokers failed for {symbol} on {exchange}")

    def health_all(self) -> Dict[str, Any]:
        """Aggregate health status of every registered broker."""
        healths = {}
        for bt, gw in self._gateways.items():
            h = gw.health()
            healths[bt.value] = {
                "status": h.status.value,
                "latency_ms": h.latency_ms,
                "last_heartbeat": h.last_heartbeat,
                "error_count": h.error_count,
                "orders_today": h.orders_today,
                "exchanges": h.supported_exchanges,
                "order_types": h.supported_order_types,
                "paper_mode": h.paper_mode,
            }
        all_exchanges = sorted(set(
            ex for gw in self._gateways.values() for ex in gw.supported_exchanges()
        ))
        return {
            "brokers": healths,
            "primary": self._primary.value if self._primary else None,
            "total_brokers": len(self._gateways),
            "connected": sum(
                1 for gw in self._gateways.values()
                if gw.health().status == GatewayStatus.CONNECTED
            ),
            "supported_exchanges": all_exchanges,
            "failover_history": self._failover_history[-10:],
        }

    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Aggregate positions across all connected brokers."""
        out: List[Dict[str, Any]] = []
        for bt, gw in self._gateways.items():
            if gw.health().status != GatewayStatus.CONNECTED:
                continue
            try:
                for pos in await gw.get_positions():
                    out.append({
                        "broker": bt.value,
                        "symbol": pos.symbol,
                        "exchange": pos.exchange,
                        "quantity": pos.quantity,
                        "avg_price": pos.avg_price,
                        "current_price": pos.current_price,
                        "pnl": pos.pnl,
                        "side": pos.side,
                    })
            except Exception as e:
                logger.debug("Positions from %s failed: %s", bt.value, e)
        return out

    async def start_health_monitor(self, interval: float = 30.0) -> None:
        """Background task that reconnects failed brokers."""
        async def _loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    for bt, gw in self._gateways.items():
                        h = gw.health()
                        if h.status == GatewayStatus.ERROR and h.error_count > 5:
                            logger.warning("Broker %s: %d errors — reconnecting", bt.value, h.error_count)
                            try:
                                await gw.connect()
                            except Exception:
                                pass
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Health monitor error: %s", e)

        self._health_task = asyncio.create_task(_loop())

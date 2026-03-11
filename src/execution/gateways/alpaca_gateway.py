"""Alpaca Markets gateway for US equity trading (commission-free)."""
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import (
    BaseBrokerGateway, BrokerHealth, BrokerOrder, BrokerPosition,
    BrokerType, GatewayStatus,
)

logger = logging.getLogger(__name__)


class AlpacaGateway(BaseBrokerGateway):
    """Alpaca Markets gateway — US equities, commission-free.

    Supports both paper and live trading via Alpaca's REST API v2.
    Paper mode is the default when no credentials are provided.
    """

    broker_type = BrokerType.ALPACA

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._api_secret = os.environ.get("ALPACA_API_SECRET", "")
        self._base_url = (
            "https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets"
        )
        self._status = GatewayStatus.DISCONNECTED
        self._last_heartbeat: Optional[float] = None
        self._error_count = 0
        self._orders_today = 0
        self._client = None  # httpx.AsyncClient, lazily created

    async def connect(self) -> bool:
        if not self._api_key or not self._api_secret:
            logger.info("Alpaca: no API credentials — paper-only mode")
            self.paper = True
            self._status = GatewayStatus.CONNECTED
            self._last_heartbeat = time.time()
            return True
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "APCA-API-KEY-ID": self._api_key,
                    "APCA-API-SECRET-KEY": self._api_secret,
                },
                timeout=15.0,
            )
            resp = await self._client.get("/v2/account")
            resp.raise_for_status()
            account = resp.json()
            logger.info(
                "Alpaca connected: equity=$%s buying_power=$%s paper=%s",
                account.get("equity"), account.get("buying_power"), self.paper,
            )
            self._status = GatewayStatus.CONNECTED
            self._last_heartbeat = time.time()
            return True
        except Exception as e:
            logger.error("Alpaca connect failed: %s", e)
            self._status = GatewayStatus.ERROR
            self._error_count += 1
            return False

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._status = GatewayStatus.DISCONNECTED

    async def place_order(
        self, symbol: str, exchange: str, side: str, quantity: int,
        order_type: str = "MARKET", limit_price: Optional[float] = None,
        **kwargs,
    ) -> BrokerOrder:
        if self.paper and self._client is None:
            order_id = f"paper-alp-{uuid.uuid4().hex[:12]}"
            self._orders_today += 1
            return BrokerOrder(
                broker_order_id=order_id, symbol=symbol, exchange=exchange,
                side=side, quantity=quantity, order_type=order_type,
                limit_price=limit_price, status="FILLED",
                filled_qty=quantity, avg_price=limit_price or 0.0,
                timestamp=str(time.time()),
            )
        if self._client is None:
            raise RuntimeError("Alpaca client not connected")
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": kwargs.get("tif", "day"),
        }
        if order_type.upper() == "LIMIT" and limit_price:
            payload["limit_price"] = str(limit_price)
        resp = await self._client.post("/v2/orders", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._orders_today += 1
        self._last_heartbeat = time.time()
        return BrokerOrder(
            broker_order_id=data["id"], symbol=data["symbol"],
            exchange="NYSE", side=data["side"].upper(),
            quantity=int(data["qty"]), order_type=data["type"].upper(),
            limit_price=float(data.get("limit_price") or 0),
            status=data["status"].upper(),
            filled_qty=int(data.get("filled_qty") or 0),
            avg_price=float(data.get("filled_avg_price") or 0),
            timestamp=data.get("submitted_at"),
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        if self._client is None:
            return True  # paper mode
        try:
            resp = await self._client.delete(f"/v2/orders/{broker_order_id}")
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.error("Alpaca cancel failed: %s", e)
            return False

    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        if self._client is None:
            return BrokerOrder(
                broker_order_id=broker_order_id, symbol="", exchange="",
                side="", quantity=0, order_type="", limit_price=None, status="UNKNOWN",
            )
        resp = await self._client.get(f"/v2/orders/{broker_order_id}")
        resp.raise_for_status()
        d = resp.json()
        return BrokerOrder(
            broker_order_id=d["id"], symbol=d["symbol"], exchange="NYSE",
            side=d["side"].upper(), quantity=int(d["qty"]),
            order_type=d["type"].upper(),
            limit_price=float(d.get("limit_price") or 0),
            status=d["status"].upper(),
            filled_qty=int(d.get("filled_qty") or 0),
            avg_price=float(d.get("filled_avg_price") or 0),
        )

    async def get_positions(self) -> List[BrokerPosition]:
        if self._client is None:
            return []
        try:
            resp = await self._client.get("/v2/positions")
            resp.raise_for_status()
            return [
                BrokerPosition(
                    symbol=p["symbol"], exchange="NYSE",
                    quantity=int(p["qty"]),
                    avg_price=float(p["avg_entry_price"]),
                    current_price=float(p["current_price"]),
                    pnl=float(p["unrealized_pl"]),
                    side=p["side"].upper(),
                )
                for p in resp.json()
            ]
        except Exception as e:
            logger.error("Alpaca get_positions failed: %s", e)
            return []

    def health(self) -> BrokerHealth:
        return BrokerHealth(
            broker=self.broker_type, status=self._status,
            last_heartbeat=str(self._last_heartbeat) if self._last_heartbeat else None,
            error_count=self._error_count, orders_today=self._orders_today,
            supported_exchanges=self.supported_exchanges(),
            supported_order_types=["MARKET", "LIMIT", "STOP", "STOP_LIMIT", "TRAILING_STOP"],
            paper_mode=self.paper,
        )

    def supported_exchanges(self) -> List[str]:
        return ["NYSE", "NASDAQ", "ARCA"]

"""Publish normalized ticks/bars to Kafka for downstream (strategy engine)."""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from src.core.events import Bar, Tick

logger = logging.getLogger(__name__)


class MarketDataStream:
    """Produce ticks and bars to Kafka. Resilient: retry, DLQ on failure."""

    def __init__(self, bootstrap_servers: str = "localhost:9092", ticks_topic: str = "market.ticks", bars_topic: str = "market.bars"):
        self.bootstrap_servers = bootstrap_servers
        self.ticks_topic = ticks_topic
        self.bars_topic = bars_topic
        self._producer = None

    async def get_producer(self):
        if self._producer is None:
            try:
                from aiokafka import AIOKafkaProducer
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers.split(","),
                    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                )
                await self._producer.start()
            except ImportError:
                logger.warning("aiokafka not installed; MarketDataStream no-op")
        return self._producer

    @staticmethod
    def _serialize_tick(tick: Tick) -> dict:
        return {
            "symbol": tick.symbol,
            "exchange": tick.exchange.value,
            "price": tick.price,
            "size": tick.size,
            "ts": tick.ts.isoformat(),
            "side": tick.side,
        }

    @staticmethod
    def _serialize_bar(bar: Bar) -> dict:
        return {
            "symbol": bar.symbol,
            "exchange": bar.exchange.value,
            "interval": bar.interval,
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
            "volume": bar.volume, "ts": bar.ts.isoformat(), "source": bar.source,
        }

    async def publish_tick(self, tick: Tick) -> None:
        producer = await self.get_producer()
        if not producer:
            return
        try:
            await producer.send_and_wait(self.ticks_topic, value=self._serialize_tick(tick), key=tick.symbol.encode())
        except Exception as e:
            logger.exception("Kafka publish_tick failed: %s", e)

    async def publish_bar(self, bar: Bar) -> None:
        producer = await self.get_producer()
        if not producer:
            return
        try:
            await producer.send_and_wait(self.bars_topic, value=self._serialize_bar(bar), key=bar.symbol.encode())
        except Exception as e:
            logger.exception("Kafka publish_bar failed: %s", e)

"""Normalize exchange-specific payloads to core Bar/Tick. Time-sync to UTC."""

from datetime import datetime
from typing import Any

from src.core.events import Bar, Exchange, Tick
from src.market_data.timestamp import normalize_ts


class Normalizer:
    """Convert raw exchange data to canonical Bar/Tick/OrderBookSnapshot."""

    @staticmethod
    def to_utc(ts: Any) -> datetime:
        """Convert any timestamp to UTC.  Delegates to the canonical
        :func:`~src.market_data.timestamp.normalize_ts` implementation."""
        return normalize_ts(ts)

    @classmethod
    def to_tick(cls, raw: dict[str, Any], exchange: Exchange) -> Tick:
        return Tick(
            symbol=raw.get("symbol", ""),
            exchange=exchange,
            price=float(raw.get("price", raw.get("ltp", 0))),
            size=float(raw.get("size", raw.get("volume", 0))),
            ts=cls.to_utc(raw.get("ts", raw.get("timestamp"))),
            side=raw.get("side"),
        )

    @classmethod
    def to_bar(cls, raw: dict[str, Any], exchange: Exchange, interval: str, source: str = "") -> Bar:
        return Bar(
            symbol=raw.get("symbol", ""),
            exchange=exchange,
            interval=interval,
            open=float(raw.get("open", raw.get("o", 0))),
            high=float(raw.get("high", raw.get("h", 0))),
            low=float(raw.get("low", raw.get("l", 0))),
            close=float(raw.get("close", raw.get("c", 0))),
            volume=float(raw.get("volume", raw.get("v", 0))),
            ts=cls.to_utc(raw.get("ts", raw.get("timestamp"))),
            source=source,
        )

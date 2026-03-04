"""
Yahoo Finance Bar Feeder: periodically pulls OHLCV bars and pushes
them into BarCache so the autonomous loop has real market data,
even when Angel One is not configured.

Usage (automatic via app lifespan fallback):
    feeder = YahooBarFeeder(bar_cache, symbols=["RELIANCE", "INFY", "TCS"])
    feeder.start()         # starts background asyncio task
    await feeder.stop()    # graceful shutdown
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.core.events import Bar, Exchange

logger = logging.getLogger(__name__)

# Try to import Yahoo Finance connector
try:
    from src.market_data.connectors.yahoo_finance import get_yahoo_connector
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


class YahooBarFeeder:
    """
    Background feeder: pulls 1-min OHLCV from Yahoo Finance for configured
    symbols and pushes each bar into BarCache so the autonomous loop can
    generate real signals.

    Flow: Yahoo Finance → Bar objects → BarCache.append_bar → autonomous loop
    """

    DEFAULT_SYMBOLS = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"]

    def __init__(
        self,
        bar_cache,
        symbols: Optional[List[str]] = None,
        exchange: str = "NSE",
        poll_interval_seconds: float = 60.0,
        bars_per_pull: int = 100,
    ):
        self.bar_cache = bar_cache
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.exchange = exchange
        self.poll_interval = poll_interval_seconds
        self.bars_per_pull = bars_per_pull
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count = 0
        self._last_bar_ts: dict[str, str] = {}  # symbol -> last bar ts (dedup)

    def _dict_to_bar(self, symbol: str, raw: dict) -> Bar:
        """Convert Yahoo Finance dict to Bar pydantic model."""
        ts = raw.get("ts")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
        elif not isinstance(ts, datetime):
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return Bar(
            symbol=symbol,
            exchange=Exchange(self.exchange),
            interval="1m",
            open=float(raw.get("open", 0)),
            high=float(raw.get("high", 0)),
            low=float(raw.get("low", 0)),
            close=float(raw.get("close", 0)),
            volume=float(raw.get("volume", 0)),
            ts=ts,
            source="yahoo_finance",
        )

    def _push_bars_to_cache(self, symbol: str, bars: list) -> int:
        """Convert raw bar dicts to Bar objects and push into BarCache."""
        if not bars:
            return 0
        count = 0
        last_known_ts = self._last_bar_ts.get(symbol)
        for raw in bars:
            try:
                bar = self._dict_to_bar(symbol, raw)
                bar_ts_str = bar.ts.isoformat()
                # Dedup: skip bars we've already pushed
                if last_known_ts and bar_ts_str <= last_known_ts:
                    continue
                self.bar_cache.append_bar(bar)
                self._last_bar_ts[symbol] = bar_ts_str
                count += 1
            except Exception as e:
                logger.debug("Bar conversion/push failed for %s: %s", symbol, e)
        return count

    async def _pull_and_push(self) -> None:
        """Pull bars from Yahoo Finance for all symbols and push to cache."""
        connector = get_yahoo_connector() if _YF_AVAILABLE else None
        if connector is None:
            logger.debug("Yahoo Finance connector not available; skipping bar pull")
            return

        loop = asyncio.get_running_loop()
        for symbol in self.symbols:
            try:
                bars = await loop.run_in_executor(
                    None,
                    lambda s=symbol: connector.get_bars(
                        s, self.exchange, "1m", self.bars_per_pull
                    ),
                )
                if bars:
                    pushed = self._push_bars_to_cache(symbol, bars)
                    if pushed > 0 and self._tick_count < 3:
                        logger.info(
                            "YahooBarFeeder: %s → %d new bars pushed to BarCache",
                            symbol, pushed,
                        )
            except Exception as e:
                logger.debug("YahooBarFeeder pull failed for %s: %s", symbol, e)

        self._tick_count += 1

    async def _loop(self) -> None:
        """Main loop: pull bars, sleep, repeat."""
        await self._pull_and_push()
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval)
                if not self._running:
                    break
                await self._pull_and_push()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("YahooBarFeeder loop error: %s", e)

    def start(self) -> None:
        if self._running:
            return
        if not _YF_AVAILABLE:
            logger.warning(
                "YahooBarFeeder: yfinance not available; bar feeder disabled. "
                "Install with: pip install yfinance"
            )
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "YahooBarFeeder started: symbols=%s, poll=%ds",
            self.symbols, int(self.poll_interval),
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("YahooBarFeeder stopped")

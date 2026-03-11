"""
YFinance fallback feeder: populates BarCache with real NSE market data
when no Angel One WebSocket is configured (paper trading without broker keys).
Fetches 1-minute OHLCV bars from yfinance and pushes them into the bar cache
so the autonomous loop, strategies, and agents can generate real signals.
"""

import asyncio
import logging

from src.market_data.timestamp import normalize_ts

logger = logging.getLogger(__name__)


def _discover_symbols(count: int = 10) -> list[str]:
    """Dynamically discover top liquid NSE symbols for paper trading feed."""
    try:
        from src.scanner.dynamic_universe import get_dynamic_universe

        symbols = get_dynamic_universe().get_tradeable_stocks(count=count)
        if symbols:
            return [f"{s}.NS" for s in symbols]
    except Exception:
        pass
    try:
        from src.market_data.symbol_token_map import get_symbol_token_map

        stm = get_symbol_token_map()
        if stm.is_loaded:
            nse = stm.get_all_nse_equity_symbols()
            return [f"{s}.NS" for s in nse[:count]]
    except Exception:
        pass
    return []


class YFinanceFallbackFeeder:
    """
    Periodically fetch 1-minute bars from yfinance and push into BarCache.
    Acts as a replacement for Angel One WebSocket when no broker keys are configured.
    """

    def __init__(
        self,
        bar_cache,
        symbols: list[str] | None = None,
        poll_interval_seconds: float = 60.0,
    ):
        self._bar_cache = bar_cache
        self._symbols = symbols or _discover_symbols(count=10)
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_bar_ts = {}  # symbol -> last bar timestamp to avoid duplicates
        self._initial_load_done = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "YFinance fallback feeder started: %d symbols, poll every %.0fs",
            len(self._symbols),
            self._poll_interval,
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

    async def _poll_loop(self) -> None:
        # Initial load: fetch historical bars to fill the cache
        try:
            await self._fetch_and_push(initial=True)
        except asyncio.CancelledError:
            logger.info("YFinance feeder initial fetch cancelled")
            return
        self._initial_load_done = True

        while self._running:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._fetch_and_push(initial=False)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("YFinance feeder poll error: %s", e)

    async def _fetch_and_push(self, initial: bool = False) -> None:
        loop = asyncio.get_event_loop()
        bars_added = await loop.run_in_executor(None, self._fetch_bars_sync, initial)
        if bars_added > 0:
            logger.info("YFinance feeder: pushed %d bars into cache", bars_added)

    def _fetch_bars_sync(self, initial: bool) -> int:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed, feeder disabled")
            return 0

        from src.core.events import Bar, Exchange

        total_added = 0
        # On initial load, fetch 2 days of 1m data to fill cache with 60+ bars
        # On poll, fetch last 1 day to get latest bars
        period = "2d" if initial else "1d"

        for symbol in self._symbols:
            try:
                data = yf.download(
                    symbol,
                    period=period,
                    interval="1m",
                    progress=False,
                    threads=False,
                )
                if data is None or data.empty:
                    continue

                # Handle multi-level columns from yfinance
                if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
                    data.columns = data.columns.droplevel(1)

                # Strip .NS suffix for internal symbol name
                clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
                last_ts = self._last_bar_ts.get(symbol)

                for idx, row in data.iterrows():
                    ts = idx.to_pydatetime()
                    # yfinance .NS symbols return IST timestamps (often
                    # naive or incorrectly tagged).  Use canonical
                    # normalize_ts with IST assumption for naive values.
                    ts = normalize_ts(ts, source_tz="Asia/Kolkata")

                    # Skip bars we've already pushed
                    if last_ts is not None and ts <= last_ts:
                        continue

                    bar = Bar(
                        symbol=clean_symbol,
                        exchange=Exchange.NSE,
                        interval="1m",
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        ts=ts,
                        source="yfinance",
                    )
                    self._bar_cache.append_bar(bar)
                    total_added += 1
                    self._last_bar_ts[symbol] = ts

            except Exception as e:
                logger.debug("YFinance fetch failed for %s: %s", symbol, e)

        return total_added

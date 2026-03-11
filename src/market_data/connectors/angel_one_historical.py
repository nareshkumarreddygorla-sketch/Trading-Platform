"""
Angel One Historical Data Connector: fetch OHLCV candles via SmartAPI REST.
Supports: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import aiohttp

from src.core.events import Bar, Exchange
from src.market_data.symbol_token_map import get_symbol_token_map

logger = logging.getLogger(__name__)

# Angel One SmartAPI historical data endpoint
HISTORICAL_URL = "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData"

# Interval mapping
INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

# Max records per request (Angel One limit)
MAX_RECORDS = {
    "ONE_MINUTE": 30,  # 30 days
    "FIVE_MINUTE": 90,  # 90 days
    "FIFTEEN_MINUTE": 180,  # 180 days
    "THIRTY_MINUTE": 180,
    "ONE_HOUR": 365,
    "ONE_DAY": 2000,  # ~5.5 years
}


class AngelOneHistorical:
    """
    Fetch historical OHLCV data from Angel One SmartAPI.

    Requires:
      - Valid JWT token from Angel One authentication
      - SymbolTokenMap loaded with instrument master
    """

    def __init__(self, api_key: str, jwt_token: str):
        self._api_key = api_key
        self._jwt_token = jwt_token
        self._stm = get_symbol_token_map()

    def update_token(self, jwt_token: str):
        """Update JWT after token refresh."""
        self._jwt_token = jwt_token

    async def fetch_candles(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1d",
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Bar]:
        """
        Fetch historical candles for a symbol.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            exchange: Exchange ("NSE", "BSE")
            interval: Bar interval ("1m", "5m", "15m", "30m", "1h", "1d")
            from_date: Start date (default: 1 year ago for daily)
            to_date: End date (default: today)

        Returns:
            List of Bar objects sorted by timestamp ascending.
        """
        angel_interval = INTERVAL_MAP.get(interval)
        if not angel_interval:
            logger.error("Invalid interval: %s", interval)
            return []

        token = self._stm.get_token(symbol, exchange)
        if not token:
            logger.error("No token found for %s on %s", symbol, exchange)
            return []

        if to_date is None:
            to_date = datetime.now(UTC)
        if from_date is None:
            max_days = MAX_RECORDS.get(angel_interval, 365)
            from_date = to_date - timedelta(days=max_days)

        # Angel One date format: "YYYY-MM-DD HH:MM"
        from_str = from_date.strftime("%Y-%m-%d 09:15")
        to_str = to_date.strftime("%Y-%m-%d 15:30")

        payload = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": angel_interval,
            "fromdate": from_str,
            "todate": to_str,
        }

        headers = {
            "Authorization": f"Bearer {self._jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1",
            "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "00:00:00:00:00:00",
            "X-PrivateKey": self._api_key,
        }

        bars: list[Bar] = []
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(HISTORICAL_URL, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("Historical API error HTTP %d: %s", resp.status, body[:500])
                        return []

                    data = await resp.json()

            if not data.get("status"):
                logger.error("Historical API error: %s", data.get("message", "unknown"))
                return []

            candles = data.get("data", [])
            if not candles:
                logger.warning("No candles returned for %s %s %s", symbol, exchange, interval)
                return []

            for candle in candles:
                # Angel One candle format: [timestamp, open, high, low, close, volume]
                if len(candle) < 6:
                    continue
                try:
                    ts = datetime.fromisoformat(candle[0].replace("T", " ").split("+")[0])
                    bars.append(
                        Bar(
                            symbol=symbol,
                            exchange=Exchange.NSE if exchange == "NSE" else Exchange.BSE,
                            interval=interval,
                            ts=ts,
                            open=float(candle[1]),
                            high=float(candle[2]),
                            low=float(candle[3]),
                            close=float(candle[4]),
                            volume=int(candle[5]),
                        )
                    )
                except (ValueError, IndexError) as e:
                    logger.debug("Skipping malformed candle: %s", e)
                    continue

            logger.info("Fetched %d candles for %s %s %s", len(bars), symbol, exchange, interval)
            return bars

        except TimeoutError:
            logger.error("Historical API timeout for %s", symbol)
            return []
        except Exception as e:
            logger.exception("Historical API error for %s: %s", symbol, e)
            return []

    async def fetch_candles_bulk(
        self,
        symbols: list[str],
        exchange: str = "NSE",
        interval: str = "1d",
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        max_concurrent: int = 5,
    ) -> dict[str, list[Bar]]:
        """
        Fetch candles for multiple symbols with rate limiting.

        Args:
            symbols: List of trading symbols
            max_concurrent: Max concurrent API calls (avoid rate limits)

        Returns:
            Dict of symbol -> List[Bar]
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: dict[str, list[Bar]] = {}

        async def _fetch_one(sym: str):
            async with semaphore:
                bars = await self.fetch_candles(sym, exchange, interval, from_date, to_date)
                results[sym] = bars
                # Rate limit: Angel One allows ~3 req/sec
                await asyncio.sleep(0.35)

        tasks = [_fetch_one(sym) for sym in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

        fetched = sum(1 for v in results.values() if v)
        logger.info("Bulk fetch: %d/%d symbols with data", fetched, len(symbols))
        return results

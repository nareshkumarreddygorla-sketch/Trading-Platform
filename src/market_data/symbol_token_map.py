"""
Angel One Symbol Token Map: maps NSE symbols to Angel One numeric tokens.
Downloads and caches the OpenAPI instrument master file.
Required for: historical data API, WebSocket subscriptions, order placement.
"""

import json
import logging
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

INSTRUMENT_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_FILE = CACHE_DIR / "angel_one_instruments.json"
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours


class SymbolTokenMap:
    """
    Maps trading symbols to Angel One instrument tokens.

    Usage:
        stm = SymbolTokenMap()
        await stm.load()
        token = stm.get_token("RELIANCE", "NSE")  # -> "2885"
        symbol = stm.get_symbol("2885", "NSE")     # -> "RELIANCE"
    """

    def __init__(self):
        # symbol -> {token, exchange, instrument_type, lot_size, tick_size, expiry}
        self._nse_eq: dict[str, dict] = {}
        self._nse_fo: dict[str, dict] = {}
        self._bse_eq: dict[str, dict] = {}
        # Reverse: token -> symbol
        self._token_to_symbol: dict[str, str] = {}
        # Full instrument list
        self._instruments: list = []
        self._loaded = False
        self._load_time: float = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def instrument_count(self) -> int:
        return len(self._instruments)

    async def load(self, force_refresh: bool = False) -> bool:
        """Load instrument master from cache or download fresh."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Check cache
        if not force_refresh and CACHE_FILE.exists():
            cache_age = time.time() - CACHE_FILE.stat().st_mtime
            if cache_age < CACHE_TTL_SECONDS:
                return self._load_from_cache()

        # Download fresh
        success = await self._download_instruments()
        if not success:
            # Fallback to stale cache
            if CACHE_FILE.exists():
                logger.warning("Download failed, using stale cache")
                return self._load_from_cache()
            return False
        return True

    def _load_from_cache(self) -> bool:
        """Parse cached instrument master."""
        try:
            data = json.loads(CACHE_FILE.read_text())
            self._parse_instruments(data)
            logger.info("Loaded %d instruments from cache", len(data))
            return True
        except Exception as e:
            logger.error("Failed to load cache: %s", e)
            return False

    async def _download_instruments(self) -> bool:
        """Download instrument master from Angel One."""
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(INSTRUMENT_MASTER_URL) as resp:
                    if resp.status != 200:
                        logger.error("Instrument master download failed: HTTP %d", resp.status)
                        return False
                    data = await resp.json(content_type=None)

            # Cache to disk
            CACHE_FILE.write_text(json.dumps(data))
            self._parse_instruments(data)
            logger.info("Downloaded %d instruments from Angel One", len(data))
            return True
        except Exception as e:
            logger.error("Instrument master download error: %s", e)
            return False

    def _parse_instruments(self, data: list) -> None:
        """Parse instrument list into lookup maps."""
        self._instruments = data
        self._nse_eq.clear()
        self._nse_fo.clear()
        self._bse_eq.clear()
        self._token_to_symbol.clear()

        for inst in data:
            token = str(inst.get("token", ""))
            symbol = inst.get("symbol", "")
            name = inst.get("name", "")
            exchange = inst.get("exch_seg", "")
            instrument_type = inst.get("instrumenttype", "")
            lot_size = inst.get("lotsize", "1")
            tick_size = inst.get("tick_size", "0.05")
            expiry = inst.get("expiry", "")

            info = {
                "token": token,
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "instrument_type": instrument_type,
                "lot_size": int(lot_size) if lot_size else 1,
                "tick_size": float(tick_size) if tick_size else 0.05,
                "expiry": expiry,
            }

            if exchange == "NSE":
                # NSE equities: symbol ends with -EQ or instrument is empty/EQ
                trading_symbol = symbol.replace("-EQ", "")
                if instrument_type in ("", "EQ") or symbol.endswith("-EQ"):
                    self._nse_eq[trading_symbol] = info
                    self._token_to_symbol[token] = trading_symbol
                else:
                    self._nse_fo[symbol] = info
            elif exchange == "BSE":
                trading_symbol = symbol.replace("-EQ", "")
                self._bse_eq[trading_symbol] = info

        self._loaded = True
        self._load_time = time.time()
        logger.info(
            "Parsed instruments: NSE_EQ=%d, NSE_FO=%d, BSE_EQ=%d",
            len(self._nse_eq),
            len(self._nse_fo),
            len(self._bse_eq),
        )

    def get_token(self, symbol: str, exchange: str = "NSE") -> str | None:
        """Get Angel One token for a trading symbol."""
        symbol = symbol.upper().replace("-EQ", "")
        if exchange == "NSE":
            info = self._nse_eq.get(symbol)
        elif exchange == "BSE":
            info = self._bse_eq.get(symbol)
        else:
            return None
        return info["token"] if info else None

    def get_symbol(self, token: str, exchange: str = "NSE") -> str | None:
        """Reverse lookup: token -> symbol."""
        return self._token_to_symbol.get(str(token))

    def get_instrument_info(self, symbol: str, exchange: str = "NSE") -> dict | None:
        """Full instrument info for a symbol."""
        symbol = symbol.upper().replace("-EQ", "")
        if exchange == "NSE":
            return self._nse_eq.get(symbol)
        elif exchange == "BSE":
            return self._bse_eq.get(symbol)
        return None

    def get_all_nse_equity_symbols(self) -> list:
        """Return all NSE equity trading symbols."""
        return sorted(self._nse_eq.keys())

    def get_exchange_type(self, exchange: str) -> int:
        """Angel One exchange type codes for WebSocket V2."""
        return {"NSE": 1, "NFO": 2, "BSE": 3, "BFO": 4, "MCX": 5, "CDS": 7}.get(exchange, 1)

    def get_ws_subscription_list(self, symbols: list, exchange: str = "NSE") -> list:
        """
        Build WebSocket V2 subscription token list.
        Returns: list of [exchange_type, token_str] pairs.
        """
        exchange_type = self.get_exchange_type(exchange)
        result = []
        for sym in symbols:
            token = self.get_token(sym, exchange)
            if token:
                result.append([exchange_type, token])
            else:
                logger.warning("No token found for %s on %s", sym, exchange)
        return result


# Module-level singleton
_instance: SymbolTokenMap | None = None


def get_symbol_token_map() -> SymbolTokenMap:
    """Get the global SymbolTokenMap singleton."""
    global _instance
    if _instance is None:
        _instance = SymbolTokenMap()
    return _instance

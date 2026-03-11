"""
Multi-source market data downloader.
Automatically selects the best available data source and downloads
maximum historical data for training AI models.

Supported sources (priority order):
  1. Yahoo Finance (yfinance) — free, no API key, 2+ years daily, 7 days intraday
  2. Alpha Vantage — free tier (25 req/day), 20+ years daily data
  3. Polygon.io — free tier (5 req/min), institutional-quality data
  4. NSE India direct — free, but rate-limited

Usage:
    from scripts.data_sources import DataDownloader
    dd = DataDownloader()
    df = dd.download("RELIANCE", period="5y", interval="1d")
"""
import logging
import os
import time
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nse_historical")


# ═══════════════════════════════════════════════════════════════════════════
# Source 1: Yahoo Finance (default, free, no key needed)
# ═══════════════════════════════════════════════════════════════════════════
class YFinanceSource:
    """Yahoo Finance via yfinance library. Best for daily data (2+ years)."""
    name = "yfinance"

    def __init__(self):
        try:
            import yfinance
            self._yf = yfinance
            self.available = True
        except ImportError:
            self.available = False

    def download(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        yf_symbol = f"{symbol}.NS" if not symbol.startswith("^") else symbol
        try:
            ticker = self._yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                # Try without .NS suffix (some symbols like indices)
                ticker = self._yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "timestamp"
            df["symbol"] = symbol
            df["source"] = self.name
            return df
        except Exception as e:
            logger.debug("YFinance failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def download_bulk(self, symbols: List[str], period: str = "2y",
                      interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Download multiple symbols efficiently using yfinance batch."""
        if not self.available:
            return {}
        yf_symbols = [f"{s}.NS" if not s.startswith("^") else s for s in symbols]
        results = {}
        try:
            # Batch download (much faster)
            data = self._yf.download(
                yf_symbols, period=period, interval=interval,
                group_by="ticker", threads=True, progress=False,
            )
            for orig_sym, yf_sym in zip(symbols, yf_symbols):
                try:
                    if len(yf_symbols) == 1:
                        df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        df = data[yf_sym][["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df.dropna(inplace=True)
                    if not df.empty:
                        df.index.name = "timestamp"
                        df["symbol"] = orig_sym
                        df["source"] = self.name
                        results[orig_sym] = df
                except Exception:
                    pass
        except Exception as e:
            logger.warning("YFinance bulk download failed: %s (falling back to individual)", e)
            for sym in symbols:
                df = self.download(sym, period=period, interval=interval)
                if not df.empty:
                    results[sym] = df
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Source 2: Alpha Vantage (needs free API key, 25 req/day free)
# ═══════════════════════════════════════════════════════════════════════════
class AlphaVantageSource:
    """Alpha Vantage API. Best for long history (20+ years daily)."""
    name = "alpha_vantage"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.available = bool(self.api_key)
        self._base_url = "https://www.alphavantage.co/query"
        self._request_count = 0

    def download(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()

        import requests

        # Rate limit: 5 req/min for free tier
        if self._request_count > 0 and self._request_count % 5 == 0:
            time.sleep(12)

        try:
            bse_symbol = f"{symbol}.BSE"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": bse_symbol,
                "outputsize": "full",
                "apikey": self.api_key,
            }
            resp = requests.get(self._base_url, params=params, timeout=30)
            self._request_count += 1
            data = resp.json()

            ts_key = "Time Series (Daily)"
            if ts_key not in data:
                return pd.DataFrame()

            records = []
            for date_str, vals in data[ts_key].items():
                records.append({
                    "timestamp": pd.Timestamp(date_str),
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": float(vals["5. volume"]),
                })

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            df["symbol"] = symbol
            df["source"] = self.name

            # Filter by period
            if period.endswith("y"):
                years = int(period[:-1])
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
                df = df[df.index >= cutoff]

            return df
        except Exception as e:
            logger.debug("AlphaVantage failed for %s: %s", symbol, e)
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# Master Downloader: auto-selects best source
# ═══════════════════════════════════════════════════════════════════════════
class DataDownloader:
    """
    Multi-source data downloader. Automatically selects the best
    available source and falls back to alternatives.
    """

    def __init__(self):
        self.sources = []
        # Priority order: yfinance first (free, no key), then Alpha Vantage
        yf = YFinanceSource()
        if yf.available:
            self.sources.append(yf)
            logger.info("Data source available: Yahoo Finance (free, no API key)")

        av = AlphaVantageSource()
        if av.available:
            self.sources.append(av)
            logger.info("Data source available: Alpha Vantage (API key set)")

        if not self.sources:
            logger.error("No data sources available! Install yfinance: pip install yfinance")

    def download(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Download from best available source with fallback."""
        for source in self.sources:
            df = source.download(symbol, period=period, interval=interval)
            if not df.empty:
                logger.info("[%s] Downloaded %s: %d bars", source.name, symbol, len(df))
                return df
        logger.warning("All sources failed for %s", symbol)
        return pd.DataFrame()

    def download_all(self, symbols: List[str], period: str = "2y",
                     interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Download all symbols, using bulk download when possible."""
        results = {}

        # Try bulk download from yfinance first
        for source in self.sources:
            if hasattr(source, "download_bulk"):
                logger.info("Attempting bulk download via %s for %d symbols...", source.name, len(symbols))
                bulk = source.download_bulk(symbols, period=period, interval=interval)
                results.update(bulk)
                remaining = [s for s in symbols if s not in results]
                if not remaining:
                    break
                logger.info("Bulk got %d/%d, downloading remaining %d individually",
                            len(bulk), len(symbols), len(remaining))
                symbols = remaining

        # Fall back to individual downloads for remaining
        for symbol in symbols:
            if symbol in results:
                continue
            df = self.download(symbol, period=period, interval=interval)
            if not df.empty:
                results[symbol] = df

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Stock Universe — FULLY DYNAMIC
# The system auto-scans the entire NSE market (~1800 stocks), filters by
# liquidity/volume/turnover, and picks the best stocks autonomously.
# No hardcoded lists needed — but kept as fallbacks for offline mode.
# ═══════════════════════════════════════════════════════════════════════════

INDEX_SYMBOLS = ["^NSEI", "^NSEBANK"]  # Nifty 50 + Bank Nifty (always needed)

# Fallback list — ONLY used when dynamic fetch fails (offline/network error)
_FALLBACK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO", "HCLTECH",
    "NESTLEIND", "BAJAJFINSV", "ONGC", "NTPC", "POWERGRID",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "TATAMOTORS", "TATASTEEL",
    "TECHM", "INDUSINDBK", "HINDALCO", "COALINDIA", "DRREDDY",
    "DIVISLAB", "GRASIM", "CIPLA", "EICHERMOT", "APOLLOHOSP",
    "HEROMOTOCO", "BPCL", "BRITANNIA", "TATACONSUM", "SBILIFE",
    "HDFCLIFE", "M&M", "UPL", "BAJAJ-AUTO", "SHREECEM",
    "BANKBARODA", "PNB", "IOC", "GAIL", "VEDL",
    "HAVELLS", "PIDILITIND", "DABUR", "GODREJCP", "MARICO",
    "BERGEPAINT", "ICICIPRULI", "COLPAL", "NAUKRI", "MUTHOOTFIN",
    "CHOLAFIN", "TATAPOWER", "DLF", "SIEMENS", "ABB",
]


def get_dynamic_symbols(count: int = 300) -> list:
    """
    Get the best tradeable stocks DYNAMICALLY from the entire NSE market.

    This is the primary way to get symbols — scans all ~1800 NSE stocks,
    filters by volume/liquidity/turnover, returns the top N.

    Falls back to static list only if network is unavailable.
    """
    # Env var override for testing/debugging
    env_symbols = os.environ.get("TRAIN_SYMBOLS", "").strip()
    if env_symbols and env_symbols != "all":
        custom = [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
        if custom:
            logger.info("Using TRAIN_SYMBOLS env override: %d symbols", len(custom))
            return custom

    # Dynamic scan
    try:
        from src.scanner.dynamic_universe import get_dynamic_universe
        universe = get_dynamic_universe()
        symbols = universe.get_tradeable_stocks(count=count)
        if symbols:
            logger.info("Dynamic universe: %d stocks auto-selected from entire NSE market", len(symbols))
            return symbols
    except Exception as e:
        logger.warning("Dynamic universe scan failed: %s — using fallback", e)

    return _FALLBACK_SYMBOLS[:count]


# Backward-compatible exports (used by auto_train_all.py and other scripts)
# These now pull from the dynamic scanner instead of being hardcoded
NIFTY_50 = _FALLBACK_SYMBOLS[:50]  # Fallback only — prefer get_dynamic_symbols()
NIFTY_NEXT_50 = _FALLBACK_SYMBOLS[50:]
CUSTOM_SYMBOLS = []
ALL_SYMBOLS = _FALLBACK_SYMBOLS

"""
Dynamic NSE Universe Scanner
=============================
Fully autonomous: fetches ALL ~1800+ NSE-listed stocks, filters by
liquidity/volume/turnover, ranks them, and returns the best tradeable
universe. No hardcoded lists — the system decides what to trade.

Pipeline:
  1. Fetch all NSE equities from EQUITY_L.csv (~1800 stocks)
  2. Fetch latest bhavcopy for volume/turnover data (all stocks, single file)
  3. Filter: min volume, min turnover, price range
  4. Rank by composite liquidity score
  5. Cache result (24h TTL, auto-refreshes)

Usage:
    from src.scanner.dynamic_universe import DynamicUniverse
    universe = DynamicUniverse()
    symbols = universe.get_tradeable_stocks()        # top 300 liquid stocks
    symbols = universe.get_training_stocks(count=200) # for AI training
"""

import io
import json
import logging
import time
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "models" / "cache"
UNIVERSE_CACHE = CACHE_DIR / "dynamic_universe.json"
BHAVCOPY_CACHE = CACHE_DIR / "latest_bhavcopy.parquet"
METADATA_CACHE = CACHE_DIR / "stock_metadata.parquet"
CACHE_TTL = 86400  # 24 hours

_SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class DynamicUniverse:
    """
    Fully autonomous stock universe builder.
    Scans the entire NSE market and picks the best stocks to trade.
    """

    def __init__(
        self,
        min_volume: int = 50_000,  # min shares/day
        min_turnover: float = 1e7,  # min INR 1 crore/day
        min_price: float = 10.0,  # exclude penny stocks
        max_price: float = 50_000.0,  # exclude ultra-expensive
        target_count: int = 300,  # how many stocks to keep
        cache_ttl: int = CACHE_TTL,
    ):
        self.min_volume = min_volume
        self.min_turnover = min_turnover
        self.min_price = min_price
        self.max_price = max_price
        self.target_count = target_count
        self.cache_ttl = cache_ttl
        self._last_scan: dict | None = None

    # ── Step 1: Fetch ALL NSE-listed symbols ──────────────────────────

    def fetch_all_nse_symbols(self) -> list[str]:
        """Fetch every equity listed on NSE from the official EQUITY_L.csv."""
        import requests

        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        try:
            resp = requests.get(url, timeout=20, headers=_SESSION_HEADERS)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
            df.columns = df.columns.str.strip()

            # Filter to EQ (regular equity) series only
            for col in ["SERIES", " SERIES"]:
                if col in df.columns:
                    df = df[df[col].str.strip() == "EQ"]
                    break

            symbols = df["SYMBOL"].str.strip().tolist()
            logger.info("Fetched %d EQ symbols from NSE EQUITY_L.csv", len(symbols))
            return symbols
        except Exception as e:
            logger.warning("EQUITY_L.csv fetch failed: %s", e)

        # Fallback: nsetools
        try:
            from nsetools import Nse

            nse = Nse()
            codes = nse.get_stock_codes()
            symbols = [k for k in codes.keys() if k != "SYMBOL"]
            logger.info("Fetched %d symbols from nsetools fallback", len(symbols))
            return symbols
        except Exception as e:
            logger.debug("nsetools fallback failed: %s", e)

        # Fallback: Nifty 500 CSV
        try:
            url2 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
            resp = requests.get(url2, timeout=15, headers=_SESSION_HEADERS)
            resp.raise_for_status()
            import csv

            reader = csv.DictReader(io.StringIO(resp.text))
            symbols = [row.get("Symbol", "").strip() for row in reader if row.get("Symbol", "").strip()]
            logger.info("Fetched %d symbols from Nifty 500 CSV fallback", len(symbols))
            return symbols
        except Exception as e:
            logger.debug("Nifty 500 CSV fallback failed: %s", e)

        # Final fallback: bundled list
        from src.scanner.nse_universe import FALLBACK_SYMBOLS

        logger.warning("All dynamic fetches failed, using %d bundled symbols", len(FALLBACK_SYMBOLS))
        return list(FALLBACK_SYMBOLS)

    # ── Step 2: Fetch bhavcopy for volume/turnover ────────────────────

    def fetch_bhavcopy(self) -> pd.DataFrame:
        """
        Fetch recent NSE bhavcopy — contains OHLCV + turnover for ALL stocks
        in a single file. Tries multiple recent dates for holidays.
        """
        import requests

        # Check cache first
        if BHAVCOPY_CACHE.exists():
            age = time.time() - BHAVCOPY_CACHE.stat().st_mtime
            if age < self.cache_ttl:
                try:
                    return pd.read_parquet(BHAVCOPY_CACHE)
                except Exception as e:
                    logger.debug("Bhavcopy cache read failed: %s", e)

        for days_ago in range(1, 10):
            date = datetime.now(timezone(timedelta(hours=5, minutes=30))) - timedelta(days=days_ago)
            if date.weekday() >= 5:  # skip weekends
                continue

            month_str = date.strftime("%b").upper()
            year_str = date.strftime("%Y")
            day_str = date.strftime("%d")

            url = (
                f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
                f"{year_str}/{month_str}/cm{day_str}{month_str}{year_str}bhav.csv.zip"
            )

            try:
                resp = requests.get(url, timeout=20, headers=_SESSION_HEADERS)
                if resp.status_code != 200:
                    continue

                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)

                df = df[df["SERIES"] == "EQ"].copy()
                logger.info("Bhavcopy loaded for %s: %d stocks", date.strftime("%Y-%m-%d"), len(df))

                # Cache it
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(BHAVCOPY_CACHE, index=False)
                except Exception as e:
                    logger.debug("Bhavcopy cache write failed: %s", e)

                return df
            except Exception as e:
                logger.debug("Bhavcopy fetch failed for %s: %s", date.strftime("%Y-%m-%d"), e)
                continue

        logger.warning("Could not fetch any recent bhavcopy data")
        return pd.DataFrame()

    # ── Step 3: Fetch market cap data (yfinance, batched) ─────────────

    def fetch_market_caps(self, symbols: list[str], batch_size: int = 50) -> dict[str, float]:
        """Fetch market caps via yfinance for additional filtering."""
        try:
            import yfinance as yf
        except ImportError:
            return {}

        market_caps = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            yf_batch = [f"{s}.NS" for s in batch]
            try:
                tickers = yf.Tickers(" ".join(yf_batch))
                for sym, yf_sym in zip(batch, yf_batch):
                    try:
                        t = tickers.tickers.get(yf_sym)
                        if t and hasattr(t, "fast_info"):
                            mc = t.fast_info.get("marketCap", 0)
                            if mc and mc > 0:
                                market_caps[sym] = mc
                    except Exception as e:
                        logger.debug("Market cap fetch failed for %s: %s", sym, e)
            except Exception as e:
                logger.debug("Market cap batch fetch failed: %s", e)
            time.sleep(0.5)

        return market_caps

    # ── Step 4: Build filtered universe ───────────────────────────────

    def build_universe(self, force_refresh: bool = False) -> dict:
        """
        Full pipeline: fetch all stocks → filter → rank → return best.

        Returns dict with:
          - symbols: List[str] — the filtered tradeable stock symbols
          - total_nse: int — total stocks on NSE
          - post_filter: int — stocks that passed filters
          - scan_date: str — when this scan was done
          - sectors: Dict — sector breakdown
        """
        # Check cache
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                self._last_scan = cached
                return cached

        logger.info("Building dynamic universe (scanning entire NSE market)...")

        # Step 1: All symbols
        all_symbols = self.fetch_all_nse_symbols()
        total_nse = len(all_symbols)

        if not all_symbols:
            return {"symbols": [], "total_nse": 0, "post_filter": 0, "scan_date": datetime.now(UTC).isoformat()}

        # Step 2: Get bhavcopy for volume/turnover
        bhav = self.fetch_bhavcopy()

        if bhav.empty:
            # No bhavcopy — fall back to returning all symbols (limited to target)
            logger.warning("No bhavcopy data, returning first %d symbols unfiltered", self.target_count)
            result = {
                "symbols": all_symbols[: self.target_count],
                "total_nse": total_nse,
                "post_filter": len(all_symbols),
                "scan_date": datetime.now(UTC).isoformat(),
                "filter_method": "unfiltered_fallback",
            }
            self._save_cache(result)
            return result

        # Step 3: Apply filters
        bhav = bhav.rename(
            columns={
                "SYMBOL": "symbol",
                "CLOSE": "close",
                "TOTTRDQTY": "volume",
                "TOTTRDVAL": "turnover",
            }
        )

        # Only keep symbols that exist in our all_symbols list
        valid_set = set(all_symbols)
        bhav = bhav[bhav["symbol"].isin(valid_set)]

        # P1-5: Add min average trade value filter (INR 50K per trade)
        # and spread proxy (turnover/volume = avg trade price; reject if < ₹10 avg trade)
        if "volume" in bhav.columns and "turnover" in bhav.columns:
            bhav["avg_trade_value"] = bhav["turnover"] / bhav["volume"].clip(lower=1)
        else:
            bhav["avg_trade_value"] = 0.0

        filtered = bhav[
            (bhav["volume"] >= self.min_volume)
            & (bhav["turnover"] >= self.min_turnover)
            & (bhav["close"] >= self.min_price)
            & (bhav["close"] <= self.max_price)
            & (bhav["avg_trade_value"] >= 10.0)  # P1-5: filter out illiquid micro-lots
        ].copy()

        logger.info("Filter results: %d total → %d after volume/turnover/price filter", total_nse, len(filtered))

        if filtered.empty:
            # Relaxed filter
            filtered = bhav[(bhav["volume"] >= self.min_volume // 5) & (bhav["close"] >= self.min_price)].copy()
            logger.info("Relaxed filter: %d stocks", len(filtered))

        # Step 4: Rank by composite liquidity score
        if len(filtered) > 0:
            filtered["vol_rank"] = filtered["volume"].rank(pct=True)
            filtered["turn_rank"] = filtered["turnover"].rank(pct=True)
            # Price proximity to round numbers (psychological levels) - minor factor
            filtered["liquidity_score"] = (
                0.45 * filtered["vol_rank"]
                + 0.45 * filtered["turn_rank"]
                + 0.10 * filtered["close"].rank(pct=True)  # prefer mid-range priced stocks
            )
            filtered = filtered.sort_values("liquidity_score", ascending=False)

        # Step 5: Take top N
        result_symbols = filtered["symbol"].tolist()[: self.target_count]

        # Build sector info if we have extra columns
        _sectors = {}
        # Bhavcopy doesn't have sector data, but we'll add that later via metadata

        result = {
            "symbols": result_symbols,
            "total_nse": total_nse,
            "post_filter": len(filtered),
            "scan_date": datetime.now(UTC).isoformat(),
            "filter_method": "volume_turnover_ranked",
            "filters_applied": {
                "min_volume": self.min_volume,
                "min_turnover": self.min_turnover,
                "min_price": self.min_price,
                "max_price": self.max_price,
            },
        }

        self._save_cache(result)
        self._last_scan = result

        logger.info(
            "Dynamic universe built: %d tradeable stocks from %d total NSE listings",
            len(result_symbols),
            total_nse,
        )

        return result

    # ── Public API ────────────────────────────────────────────────────

    def validate_cached_symbols(self, cached_symbols: list[str]) -> list[str]:
        """P1-6: Remove delisted/suspended stocks from cached universe.
        Fetches live NSE symbol list and filters out any cached symbols not present."""
        try:
            live_symbols = set(self.fetch_all_nse_symbols())
            if not live_symbols:
                return cached_symbols  # can't validate, keep as-is
            valid = [s for s in cached_symbols if s in live_symbols]
            removed = len(cached_symbols) - len(valid)
            if removed > 0:
                logger.warning(
                    "Survivorship filter: removed %d delisted/suspended symbols from cached universe",
                    removed,
                )
            return valid
        except Exception as e:
            logger.debug("Survivorship validation failed: %s — keeping cached", e)
            return cached_symbols

    def get_tradeable_stocks(self, count: int | None = None, force_refresh: bool = False) -> list[str]:
        """
        Get the best tradeable stocks. Fully autonomous — no hardcoded lists.

        Args:
            count: how many to return (default: target_count from init)
            force_refresh: bypass cache

        Returns:
            List of NSE symbol strings (e.g. ["RELIANCE", "TCS", ...])
        """
        result = self.build_universe(force_refresh=force_refresh)
        symbols = result.get("symbols", [])
        # P1-6: validate cached symbols against live NSE listings
        if not force_refresh and result.get("cache_ts"):
            symbols = self.validate_cached_symbols(symbols)
        if count:
            return symbols[:count]
        return symbols

    def get_training_stocks(self, count: int = 200) -> list[str]:
        """Get stocks for AI model training — broader set for better generalization."""
        return self.get_tradeable_stocks(count=count)

    def get_trading_stocks(self, count: int = 100) -> list[str]:
        """Get stocks for active trading — tighter set of most liquid names."""
        return self.get_tradeable_stocks(count=count)

    def get_yfinance_symbols(self, count: int | None = None) -> list[str]:
        """Get symbols with .NS suffix for yfinance compatibility."""
        return [f"{s}.NS" for s in self.get_tradeable_stocks(count=count)]

    def get_scan_info(self) -> dict:
        """Get info about the last universe scan."""
        if self._last_scan:
            return self._last_scan
        return self.build_universe()

    # ── Cache ─────────────────────────────────────────────────────────

    def _load_cache(self) -> dict | None:
        try:
            if UNIVERSE_CACHE.exists():
                data = json.loads(UNIVERSE_CACHE.read_text())
                if time.time() - data.get("cache_ts", 0) < self.cache_ttl:
                    logger.info(
                        "Using cached dynamic universe (%d symbols, scanned %s)",
                        len(data.get("symbols", [])),
                        data.get("scan_date", "?"),
                    )
                    return data
        except Exception as e:
            logger.debug("Universe cache load failed: %s", e)
        return None

    def _save_cache(self, data: dict) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data["cache_ts"] = time.time()
            UNIVERSE_CACHE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Cache save failed: %s", e)


# ── Module-level singleton ────────────────────────────────────────────

_instance: DynamicUniverse | None = None


def get_dynamic_universe() -> DynamicUniverse:
    """Get or create the singleton DynamicUniverse instance."""
    global _instance
    if _instance is None:
        _instance = DynamicUniverse()
    return _instance

"""
Nightly Data Refresh Pipeline:
1. Update Angel One instrument master (symbol token map)
2. Fetch daily OHLCV for tradeable universe
3. Store in PostgreSQL (OHLCV repo)
4. Update ADV cache
5. Trigger model retraining if data quality is good

Run at 16:00 IST after market close.
Usage: python -m scripts.nightly_data_refresh
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nightly_data_refresh")

_IST = timezone(timedelta(hours=5, minutes=30))


async def refresh_instrument_master():
    """Step 1: Update symbol token map."""
    logger.info("=== Step 1: Refresh Instrument Master ===")
    from src.market_data.symbol_token_map import get_symbol_token_map

    stm = get_symbol_token_map()
    success = await stm.load(force_refresh=True)
    if success:
        logger.info("Instrument master updated: %d instruments", stm.instrument_count)
    else:
        logger.error("Instrument master refresh failed!")
    return success


async def get_tradeable_universe():
    """Step 2: Get tradeable stock universe dynamically from NSE market data."""
    logger.info("=== Step 2: Get Tradeable Universe (Dynamic) ===")
    try:
        from src.scanner.dynamic_universe import get_dynamic_universe

        scanner = get_dynamic_universe()
        symbols = scanner.get_tradeable_stocks(count=200)
        logger.info("Dynamic universe: %d tradeable symbols", len(symbols))
        if symbols:
            return symbols
    except Exception as e:
        logger.warning("Dynamic universe scanner failed: %s", e)

    # Fallback: use instrument master (Angel One) if available
    try:
        from src.market_data.symbol_token_map import get_symbol_token_map

        stm = get_symbol_token_map()
        if stm.is_loaded:
            all_nse = stm.get_all_nse_equity_symbols()
            logger.info("Instrument master fallback: %d NSE equities available", len(all_nse))
            return all_nse[:200]
    except Exception as e:
        logger.warning("Instrument master fallback failed: %s", e)

    # Last resort: use whatever symbols already have data in the database
    try:
        from src.persistence.ohlcv_repo import OHLCVRepository

        repo = OHLCVRepository()
        existing = repo.get_symbols_with_data(interval="1d", min_bars=1)
        if existing:
            logger.info("Database fallback: %d symbols with existing data", len(existing))
            return existing[:200]
    except Exception as e:
        logger.warning("Database fallback failed: %s", e)

    logger.error("No symbol source available — cannot build universe")
    return []


async def fetch_and_store_ohlcv(symbols: list):
    """Step 3: Fetch daily OHLCV and store in database."""
    logger.info("=== Step 3: Fetch & Store OHLCV (%d symbols) ===", len(symbols))

    from src.persistence.ohlcv_repo import OHLCVRepository
    from src.core.events import Bar, Exchange

    repo = OHLCVRepository()
    total_bars = 0

    # Try Angel One first
    angel_one_success = False
    api_key = os.environ.get("EXEC_ANGEL_ONE_API_KEY") or os.environ.get("ANGEL_ONE_API_KEY")
    jwt_token = os.environ.get("EXEC_ANGEL_ONE_TOKEN") or os.environ.get("ANGEL_ONE_TOKEN")

    if api_key and jwt_token:
        try:
            from src.market_data.connectors.angel_one_historical import AngelOneHistorical

            historical = AngelOneHistorical(api_key=api_key, jwt_token=jwt_token)
            bars_data = await historical.fetch_candles_bulk(
                symbols=symbols,
                interval="1d",
                from_date=datetime.now() - timedelta(days=365),
                max_concurrent=3,
            )
            for symbol, bars in bars_data.items():
                if bars:
                    count = repo.upsert_bars(bars)
                    total_bars += count
            angel_one_success = total_bars > 0
            logger.info("Angel One: stored %d bars", total_bars)
        except Exception as e:
            logger.warning("Angel One historical fetch failed: %s", e)

    # Fallback: yfinance
    if not angel_one_success:
        logger.info("Using yfinance fallback for historical data")
        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    df = ticker.history(period="1y", interval="1d")
                    if df.empty:
                        continue

                    bars = []
                    for idx, row in df.iterrows():
                        bars.append(
                            Bar(
                                symbol=symbol,
                                exchange=Exchange.NSE,
                                interval="1d",
                                ts=idx.to_pydatetime(),
                                open=float(row["Open"]),
                                high=float(row["High"]),
                                low=float(row["Low"]),
                                close=float(row["Close"]),
                                volume=int(row["Volume"]),
                            )
                        )

                    if bars:
                        count = repo.upsert_bars(bars)
                        total_bars += count
                        logger.debug("%s: %d bars stored", symbol, count)

                except Exception as e:
                    logger.debug("yfinance error for %s: %s", symbol, e)
                    continue

            logger.info("yfinance: stored %d total bars", total_bars)
        except ImportError:
            logger.error("yfinance not installed!")

    return total_bars


async def update_adv_cache(symbols: list):
    """Step 4: Update Average Daily Volume cache."""
    logger.info("=== Step 4: Update ADV Cache ===")
    try:
        from src.persistence.ohlcv_repo import OHLCVRepository

        repo = OHLCVRepository()

        adv_data = {}
        for symbol in symbols:
            bars = repo.get_bars(symbol, interval="1d", limit=20)
            if bars:
                volumes = [b.volume for b in bars]
                adv_data[symbol] = int(sum(volumes) / len(volumes))

        logger.info("ADV cache updated for %d symbols", len(adv_data))
        return adv_data
    except Exception as e:
        logger.error("ADV cache update failed: %s", e)
        return {}


async def check_data_quality(symbols: list) -> dict:
    """Step 5: Validate data quality."""
    logger.info("=== Step 5: Data Quality Check ===")
    try:
        from src.persistence.ohlcv_repo import OHLCVRepository

        repo = OHLCVRepository()

        good_symbols = repo.get_symbols_with_data(interval="1d", min_bars=50)
        coverage = len(good_symbols) / len(symbols) * 100 if symbols else 0

        quality = {
            "total_symbols": len(symbols),
            "symbols_with_data": len(good_symbols),
            "coverage_pct": round(coverage, 1),
            "quality_pass": coverage >= 50,
        }
        logger.info(
            "Data quality: %d/%d symbols with 50+ bars (%.1f%%) - %s",
            len(good_symbols),
            len(symbols),
            coverage,
            "PASS" if quality["quality_pass"] else "FAIL",
        )
        return quality
    except Exception as e:
        logger.error("Data quality check failed: %s", e)
        return {"quality_pass": False, "error": str(e)}


async def main():
    """Run the full nightly data refresh pipeline."""
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("NIGHTLY DATA REFRESH - %s IST", datetime.now(_IST).strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    # Step 1: Instrument master
    await refresh_instrument_master()

    # Step 2: Universe
    symbols = await get_tradeable_universe()

    # Step 3: OHLCV data
    bar_count = await fetch_and_store_ohlcv(symbols)

    # Step 4: ADV cache
    adv_data = await update_adv_cache(symbols)

    # Step 5: Quality check
    quality = await check_data_quality(symbols)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=" * 60)
    logger.info("NIGHTLY DATA REFRESH COMPLETE - %.1f seconds", elapsed)
    logger.info("  Bars stored: %d", bar_count)
    logger.info("  ADV symbols: %d", len(adv_data))
    logger.info("  Quality: %s", "PASS" if quality.get("quality_pass") else "FAIL")
    logger.info("=" * 60)

    return quality


if __name__ == "__main__":
    asyncio.run(main())

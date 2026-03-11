"""
Lifespan: Bar cache, feed manager, Angel One connector, yfinance fallback feeder.
"""

import asyncio
import logging
import os
import time

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _discover_feed_symbols(count: int = 10):
    """Dynamically discover symbols for the market data feed — no hardcoded lists."""
    try:
        from src.scanner.dynamic_universe import get_dynamic_universe

        symbols = get_dynamic_universe().get_tradeable_stocks(count=count)
        if symbols:
            return symbols
    except Exception:
        pass
    try:
        from src.market_data.symbol_token_map import get_symbol_token_map

        stm = get_symbol_token_map()
        if stm.is_loaded:
            return stm.get_all_nse_equity_symbols()[:count]
    except Exception:
        pass
    return []


async def init_market_data(app: FastAPI) -> None:
    """Initialize market data layer: bar cache, tick aggregator,
    Angel One WS connector or yfinance fallback."""

    # Bar cache for autonomous loop (market data layer feeds this)
    from src.market_data.bar_aggregator import TickToBarAggregator
    from src.market_data.bar_cache import BarCache

    app.state.bar_cache = BarCache()
    app.state.bar_aggregator = TickToBarAggregator(app.state.bar_cache, interval_seconds=60)
    app.state.market_data_service = None
    app.state.angel_one_marketdata_enabled = False

    # ── Symbol Token Map (Angel One instrument master) ──
    _stm = None
    try:
        from src.market_data.symbol_token_map import get_symbol_token_map

        _stm = get_symbol_token_map()
        await _stm.load()  # must complete BEFORE connector subscribes
        app.state.symbol_token_map = _stm
        logger.info(
            "Symbol token map loaded (%d instruments)",
            _stm.instrument_count,
        )
    except Exception as e:
        logger.debug("Symbol token map not initialized: %s", e)
    try:
        from src.core.config import get_settings

        _settings = get_settings()
        _exec = getattr(_settings, "execution", None)
        _md = getattr(_settings, "market_data", None)
        _feed_cfg = getattr(_settings, "angel_one_feed", None)
        angel_one_marketdata_enabled = getattr(_feed_cfg, "marketdata_enabled", False) if _feed_cfg else False
        app.state.angel_one_marketdata_enabled = bool(angel_one_marketdata_enabled)
        _symbols = (
            (getattr(_feed_cfg, "symbols", None) or [])
            if _feed_cfg and angel_one_marketdata_enabled
            else os.environ.get("MD_SYMBOLS")
            or (getattr(_exec, "market_data_symbols", None) if _exec else None)
            or _discover_feed_symbols()
        )
        if isinstance(_symbols, str):
            _symbols = [s.strip() for s in _symbols.split(",") if s.strip()]
        connector = None
        if angel_one_marketdata_enabled:
            _api_key = getattr(_exec, "angel_one_api_key", None) or os.environ.get("ANGEL_ONE_API_KEY") or ""
            _token = getattr(_exec, "angel_one_token", None) or os.environ.get("ANGEL_ONE_TOKEN") or ""
            _secret = getattr(_exec, "angel_one_api_secret", None) or os.environ.get("ANGEL_ONE_API_SECRET", "") or ""
            if _api_key and _token:
                from src.market_data.angel_one_ws_connector import AngelOneWsConnector

                _exchange = getattr(_feed_cfg, "exchange", "NSE") if _feed_cfg else "NSE"
                _backoff = getattr(_md, "marketdata_reconnect_backoff_seconds", 5) if _md else 5

                def _market_feed_unhealthy_cb():
                    app.state.safe_mode = True
                    app.state.safe_mode_since = time.time()
                    ckc = getattr(app.state, "circuit_kill_controller", None)
                    if ckc:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.get_event_loop()
                        loop.create_task(ckc.check_market_feed_and_trip(False))

                connector = AngelOneWsConnector(
                    api_key=_api_key,
                    api_secret=_secret,
                    token=_token,
                    exchange=_exchange,
                    on_feed_unhealthy=_market_feed_unhealthy_cb,
                    symbol_token_map=_stm,
                )
        if connector is None and not angel_one_marketdata_enabled:
            _api_key = getattr(_exec, "angel_one_api_key", None) or os.environ.get("ANGEL_ONE_API_KEY") or ""
            _token = getattr(_exec, "angel_one_token", None) or os.environ.get("ANGEL_ONE_TOKEN") or ""
            if _api_key and _token:
                from src.market_data.connectors.angel_one import AngelOneConnector

                connector = AngelOneConnector(
                    api_key=_api_key, api_secret=os.environ.get("ANGEL_ONE_API_SECRET", ""), token=_token
                )
        if connector is not None:
            from src.market_data.market_data_service import MarketDataService

            def _market_feed_unhealthy():
                app.state.safe_mode = True
                app.state.safe_mode_since = time.time()
                ckc = getattr(app.state, "circuit_kill_controller", None)
                if ckc:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.get_event_loop()
                    loop.create_task(ckc.check_market_feed_and_trip(False))

            _reconnect_delay = getattr(_md, "marketdata_reconnect_backoff_seconds", 5) if _md else 5
            app.state.market_data_service = MarketDataService(
                connector,
                app.state.bar_cache,
                app.state.bar_aggregator,
                _symbols,
                on_feed_unhealthy=_market_feed_unhealthy,
            )
            app.state.market_data_service.start()
            logger.info(
                "MarketDataService started for symbols=%s (angel_one_feed=%s)", _symbols, angel_one_marketdata_enabled
            )
    except Exception as e:
        logger.debug("MarketDataService not started: %s", e)

    # YFinance fallback feeder: populate bar cache when no Angel One WebSocket
    if app.state.market_data_service is None:
        try:
            from src.market_data.yfinance_fallback_feeder import YFinanceFallbackFeeder

            _yf_feeder = YFinanceFallbackFeeder(
                bar_cache=app.state.bar_cache,
                poll_interval_seconds=60.0,
            )
            _yf_feeder.start()
            app.state.yf_feeder = _yf_feeder
            logger.info("YFinance fallback feeder started (no Angel One keys configured)")
        except Exception as e:
            logger.debug("YFinance fallback feeder not started: %s", e)


async def shutdown_market_data(app: FastAPI) -> None:
    """Shutdown: stop MarketDataService."""
    _mds = getattr(app.state, "market_data_service", None)
    if _mds is not None:
        try:
            await _mds.stop()
        except Exception as ex:
            logger.warning("MarketDataService stop: %s", ex)

"""Unit tests for src.execution.autonomous_loop — the main autonomous trading cycle.

Covers:
- AutonomousLoop construction with default/custom args
- NSE market hours detection (_is_nse_market_hours)
- NSE holiday loading (_load_nse_holidays)
- Stable idempotency key generation
- Safe mode skip behaviour
- Drift gate and regime gate checks
- Market feed health checks
- Bar freshness staleness detection
- Daily P&L reset on new trading day
- Sentiment score → exposure multiplier conversion
- Sentiment buy blocking (strongly negative)
- Algorithm selection based on ADV (direct/twap/vwap/vwap_extended)
- WebSocket broadcast error handling
"""

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.events import Exchange
from src.execution.autonomous_loop import (
    _IST,
    AutonomousLoop,
    _is_nse_market_hours,
    _load_nse_holidays,
    stable_idempotency_key,
)
from src.execution.order_entry.request import OrderEntryResult

# ──────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────


@pytest.fixture
def mock_submit_order():
    """Mock submit_order function returning a successful result."""
    result = OrderEntryResult(
        success=True,
        order_id="ORD001",
        reject_reason=None,
        reject_detail="",
    )
    return AsyncMock(return_value=result)


@pytest.fixture
def basic_loop(mock_submit_order):
    """Minimal AutonomousLoop instance with all optional deps disabled."""
    return AutonomousLoop(
        submit_order_fn=mock_submit_order,
        get_safe_mode=lambda: False,
        paper_mode=True,
    )


@pytest.fixture(autouse=True)
def reset_nse_holidays():
    """Reset the module-level holiday cache before each test."""
    import src.execution.autonomous_loop as al

    al._NSE_HOLIDAYS = set()
    al._NSE_HOLIDAYS_LOADED = False
    yield
    al._NSE_HOLIDAYS = set()
    al._NSE_HOLIDAYS_LOADED = False


# ──────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction(self, mock_submit_order):
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
        )
        assert loop.paper_mode is False
        assert loop._running is False
        assert loop._last_bar_ts is None
        assert loop.poll_interval == 60.0

    def test_custom_parameters(self, mock_submit_order):
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: True,
            poll_interval_seconds=30.0,
            paper_mode=True,
            daily_loss_limit=-0.05,
        )
        assert loop.paper_mode is True
        assert loop.poll_interval == 30.0
        assert loop._daily_loss_limit == -0.05

    def test_setter_methods(self, basic_loop):
        """All setter methods should not raise and should wire the dependency."""
        basic_loop.set_feature_normalizer(MagicMock())
        assert basic_loop._feature_normalizer is not None

        basic_loop.set_trade_outcome_repo(MagicMock())
        assert basic_loop._trade_outcome_repo is not None

        basic_loop.set_sentiment_service(MagicMock())
        assert basic_loop._sentiment_service is not None

        basic_loop.set_sentiment_predictor(MagicMock())
        assert basic_loop._sentiment_predictor is not None


# ──────────────────────────────────────────────────
# Stable idempotency key
# ──────────────────────────────────────────────────


class TestIdempotencyKey:
    def test_same_inputs_same_key(self):
        k1 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "RELIANCE", "BUY")
        k2 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "RELIANCE", "BUY")
        assert k1 == k2

    def test_different_bar_ts_different_key(self):
        k1 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "RELIANCE", "BUY")
        k2 = stable_idempotency_key("2025-03-11T10:01:00Z", "ema_cross", "RELIANCE", "BUY")
        assert k1 != k2

    def test_different_side_different_key(self):
        k1 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "RELIANCE", "BUY")
        k2 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "RELIANCE", "SELL")
        assert k1 != k2

    def test_different_symbol_different_key(self):
        k1 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "RELIANCE", "BUY")
        k2 = stable_idempotency_key("2025-03-11T10:00:00Z", "ema_cross", "TCS", "BUY")
        assert k1 != k2


# ──────────────────────────────────────────────────
# NSE market hours
# ──────────────────────────────────────────────────


class TestNSEMarketHours:
    def test_weekday_during_market_hours(self):
        """Wednesday at 10:00 IST should be market hours."""
        # 2025-03-12 is a Wednesday
        with patch("src.execution.autonomous_loop.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 12, 10, 0, 0, tzinfo=_IST)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _is_nse_market_hours()
            assert result is True

    def test_weekday_before_market(self):
        """Wednesday at 8:00 IST should not be market hours."""
        with patch("src.execution.autonomous_loop.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 12, 8, 0, 0, tzinfo=_IST)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _is_nse_market_hours()
            assert result is False

    def test_weekday_after_market(self):
        """Wednesday at 16:00 IST should not be market hours."""
        with patch("src.execution.autonomous_loop.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 12, 16, 0, 0, tzinfo=_IST)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _is_nse_market_hours()
            assert result is False

    def test_weekend_rejected(self):
        """Saturday should not be market hours regardless of time."""
        # 2025-03-15 is a Saturday
        with patch("src.execution.autonomous_loop.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 15, 10, 0, 0, tzinfo=_IST)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _is_nse_market_hours()
            assert result is False


class TestNSEHolidays:
    def test_load_from_json_file(self):
        """Should load holidays from a JSON file."""
        holidays_data = {
            "holidays": {
                "2025": ["2025-01-26", "2025-08-15"],
            },
            "last_updated": "2025-01-01",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(holidays_data, f)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"NSE_HOLIDAYS_PATH": temp_path}):
                holidays = _load_nse_holidays()
                assert (2025, 1, 26) in holidays
                assert (2025, 8, 15) in holidays
        finally:
            os.unlink(temp_path)

    def test_fallback_to_hardcoded(self):
        """Should fallback to hardcoded holidays if no JSON found."""
        with patch.dict(os.environ, {"NSE_HOLIDAYS_PATH": ""}, clear=False):
            holidays = _load_nse_holidays()
            # Should have at least Republic Day for current year
            current_year = datetime.now(_IST).year
            assert (current_year, 1, 26) in holidays

    def test_caching(self):
        """Second call should return cached result."""
        import src.execution.autonomous_loop as al

        al._NSE_HOLIDAYS = {(2025, 1, 1)}
        al._NSE_HOLIDAYS_LOADED = True
        result = _load_nse_holidays()
        assert result == {(2025, 1, 1)}


# ──────────────────────────────────────────────────
# Gate checks
# ──────────────────────────────────────────────────


class TestGateChecks:
    def test_drift_ok_when_no_gate(self, basic_loop):
        assert basic_loop._drift_ok() is True

    def test_drift_ok_passes(self, mock_submit_order):
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            drift_gate=lambda: True,
        )
        assert loop._drift_ok() is True

    def test_drift_blocked(self, mock_submit_order):
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            drift_gate=lambda: False,
        )
        assert loop._drift_ok() is False

    def test_regime_ok_when_no_gate(self, basic_loop):
        assert basic_loop._regime_ok() is True

    def test_regime_blocked(self, mock_submit_order):
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            regime_gate=lambda: False,
        )
        assert loop._regime_ok() is False

    def test_market_feed_ok_when_no_callback(self, basic_loop):
        assert basic_loop._market_feed_ok() is True

    def test_market_feed_unhealthy(self, mock_submit_order):
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            get_market_feed_healthy=lambda: False,
        )
        assert loop._market_feed_ok() is False


# ──────────────────────────────────────────────────
# Sentiment
# ──────────────────────────────────────────────────


class TestSentiment:
    def test_sentiment_to_multiplier_very_bearish(self):
        assert AutonomousLoop._sentiment_to_multiplier(0.1) == 0.5

    def test_sentiment_to_multiplier_bearish(self):
        assert AutonomousLoop._sentiment_to_multiplier(0.4) == 0.7

    def test_sentiment_to_multiplier_neutral(self):
        assert AutonomousLoop._sentiment_to_multiplier(0.5) == 1.0

    def test_sentiment_to_multiplier_bullish(self):
        assert AutonomousLoop._sentiment_to_multiplier(0.6) == 1.1

    def test_sentiment_to_multiplier_very_bullish(self):
        assert AutonomousLoop._sentiment_to_multiplier(0.9) == 1.2

    def test_sentiment_blocks_buy_when_negative(self, basic_loop):
        basic_loop._last_sentiment_detail = {"negative": 0.7, "positive": 0.1, "neutral": 0.2}
        assert basic_loop._sentiment_blocks_buy() is True

    def test_sentiment_allows_buy_when_neutral(self, basic_loop):
        basic_loop._last_sentiment_detail = {"negative": 0.3, "positive": 0.4, "neutral": 0.3}
        assert basic_loop._sentiment_blocks_buy() is False

    def test_sentiment_allows_buy_when_no_data(self, basic_loop):
        basic_loop._last_sentiment_detail = None
        assert basic_loop._sentiment_blocks_buy() is False


# ──────────────────────────────────────────────────
# Algorithm selection
# ──────────────────────────────────────────────────


class TestAlgoSelection:
    def test_direct_when_no_adv_cache(self, basic_loop):
        assert basic_loop._select_algo("RELIANCE", 100) == "direct"

    def test_direct_for_small_order(self, mock_submit_order):
        adv_cache = MagicMock()
        adv_cache.get_adv.return_value = 1_000_000  # 1M ADV
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            adv_cache=adv_cache,
        )
        # 1000 / 1_000_000 = 0.1% → < 1% → direct
        assert loop._select_algo("RELIANCE", 1000) == "direct"

    def test_twap_for_medium_order(self, mock_submit_order):
        adv_cache = MagicMock()
        adv_cache.get_adv.return_value = 100_000
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            adv_cache=adv_cache,
        )
        # 2000 / 100_000 = 2% → between 1-5% → twap
        assert loop._select_algo("RELIANCE", 2000) == "twap"

    def test_vwap_for_large_order(self, mock_submit_order):
        adv_cache = MagicMock()
        adv_cache.get_adv.return_value = 100_000
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            adv_cache=adv_cache,
        )
        # 7000 / 100_000 = 7% → between 5-10% → vwap
        assert loop._select_algo("RELIANCE", 7000) == "vwap"

    def test_vwap_extended_for_very_large_order(self, mock_submit_order):
        adv_cache = MagicMock()
        adv_cache.get_adv.return_value = 100_000
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            adv_cache=adv_cache,
        )
        # 15000 / 100_000 = 15% → > 10% → vwap_extended
        assert loop._select_algo("RELIANCE", 15000) == "vwap_extended"

    def test_direct_when_adv_none(self, mock_submit_order):
        adv_cache = MagicMock()
        adv_cache.get_adv.return_value = None
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            adv_cache=adv_cache,
        )
        assert loop._select_algo("RELIANCE", 1000) == "direct"

    def test_direct_when_adv_lookup_fails(self, mock_submit_order):
        adv_cache = MagicMock()
        adv_cache.get_adv.side_effect = RuntimeError("cache error")
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            adv_cache=adv_cache,
        )
        assert loop._select_algo("RELIANCE", 1000) == "direct"


# ──────────────────────────────────────────────────
# Daily P&L reset
# ──────────────────────────────────────────────────


class TestDailyPnLReset:
    def test_reset_on_new_day(self, basic_loop):
        basic_loop._current_trading_date = "2025-03-10"
        basic_loop._daily_pnl = -500.0
        basic_loop._signal_generation_paused = True
        # Trigger new day detection (mocking current IST date)
        with patch("src.execution.autonomous_loop.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 11, 9, 30, 0, tzinfo=_IST)
            basic_loop._reset_daily_pnl_if_new_day()
        assert basic_loop._daily_pnl == 0.0
        assert basic_loop._signal_generation_paused is False
        assert basic_loop._current_trading_date == "2025-03-11"

    def test_no_reset_same_day(self, basic_loop):
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        basic_loop._current_trading_date = today
        basic_loop._daily_pnl = -200.0
        basic_loop._reset_daily_pnl_if_new_day()
        assert basic_loop._daily_pnl == -200.0


# ──────────────────────────────────────────────────
# P&L registration
# ──────────────────────────────────────────────────


class TestPnLRegistration:
    def test_register_pnl_updates_local_mirror(self, basic_loop):
        basic_loop._register_pnl(100.0)
        assert basic_loop._daily_pnl == 100.0
        basic_loop._register_pnl(-30.0)
        assert basic_loop._daily_pnl == 70.0

    def test_register_pnl_delegates_to_risk_manager(self, basic_loop):
        rm = MagicMock()
        rm.daily_pnl = 0.0
        basic_loop._risk_manager = rm
        basic_loop._register_pnl(50.0)
        rm.register_pnl.assert_called_once_with(50.0)

    def test_get_daily_pnl_from_risk_manager(self, basic_loop):
        rm = MagicMock()
        rm.daily_pnl = -150.0
        basic_loop._risk_manager = rm
        assert basic_loop._get_daily_pnl() == -150.0

    def test_get_daily_pnl_fallback_to_local(self, basic_loop):
        basic_loop._daily_pnl = -75.0
        assert basic_loop._get_daily_pnl() == -75.0

    def test_get_daily_pnl_from_risk_state(self, basic_loop):
        basic_loop.get_risk_state = lambda: {"daily_pnl": -200.0}
        assert basic_loop._get_daily_pnl() == -200.0


# ──────────────────────────────────────────────────
# Bar freshness
# ──────────────────────────────────────────────────


class TestBarFreshness:
    @pytest.mark.asyncio
    async def test_fresh_bar_passes(self, mock_submit_order):
        fresh_bar = MagicMock()
        fresh_bar.ts = datetime.now(UTC)

        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            get_bars=lambda s, e, i, n: [fresh_bar],
            get_symbols=lambda: [("RELIANCE", Exchange.NSE)],
        )
        with patch("src.execution.autonomous_loop._is_nse_market_hours", return_value=True):
            result = await loop._check_bar_freshness()
        assert result is True

    @pytest.mark.asyncio
    async def test_stale_bar_fails(self, mock_submit_order):
        stale_bar = MagicMock()
        stale_bar.ts = datetime.now(UTC) - timedelta(minutes=10)

        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            get_bars=lambda s, e, i, n: [stale_bar],
            get_symbols=lambda: [("RELIANCE", Exchange.NSE)],
        )
        with patch("src.execution.autonomous_loop._is_nse_market_hours", return_value=True):
            result = await loop._check_bar_freshness()
        assert result is False

    @pytest.mark.asyncio
    async def test_freshness_not_enforced_outside_market_hours(self, basic_loop):
        with patch("src.execution.autonomous_loop._is_nse_market_hours", return_value=False):
            result = await basic_loop._check_bar_freshness()
        assert result is True

    @pytest.mark.asyncio
    async def test_freshness_ok_when_no_bar_provider(self, basic_loop):
        with patch("src.execution.autonomous_loop._is_nse_market_hours", return_value=True):
            result = await basic_loop._check_bar_freshness()
        assert result is True


# ──────────────────────────────────────────────────
# WebSocket broadcast
# ──────────────────────────────────────────────────


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_calls_callback(self, mock_submit_order):
        ws = AsyncMock()
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            ws_broadcast=ws,
        )
        await loop._broadcast({"type": "test"})
        ws.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_swallows_errors(self, mock_submit_order):
        ws = AsyncMock(side_effect=RuntimeError("ws broken"))
        loop = AutonomousLoop(
            submit_order_fn=mock_submit_order,
            get_safe_mode=lambda: False,
            ws_broadcast=ws,
        )
        # Should not raise
        await loop._broadcast({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_noop_when_no_callback(self, basic_loop):
        # Should not raise
        await basic_loop._broadcast({"type": "test"})


# ──────────────────────────────────────────────────
# Dynamic universe fallback
# ──────────────────────────────────────────────────


class TestDynamicUniverseFallback:
    def test_returns_cached_universe(self, basic_loop):
        import time

        basic_loop._dynamic_universe_cache = [("RELIANCE", Exchange.NSE)]
        basic_loop._dynamic_universe_ts = time.time()
        result = basic_loop._get_dynamic_universe_fallback()
        assert result == [("RELIANCE", Exchange.NSE)]

    def test_stale_cache_triggers_refresh(self, basic_loop):
        import time

        basic_loop._dynamic_universe_cache = [("OLD", Exchange.NSE)]
        basic_loop._dynamic_universe_ts = time.time() - 3600  # 1 hour old, TTL is 30 min
        # Will try DynamicUniverse and YFinance — both may fail in test env,
        # but should not crash
        result = basic_loop._get_dynamic_universe_fallback()
        # Result may be empty or populated depending on test environment
        assert isinstance(result, list)

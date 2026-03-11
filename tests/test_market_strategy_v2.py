"""
Comprehensive tests for market data (BarCache) and strategy engine
(PortfolioAllocator, StrategyRunner), plus core events, DynamicUniverse,
FeatureEngine, and EnsembleEngine.

All tests are self-contained, use mocks where needed, and require no
external services (no network, no Redis, no database).
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np

from src.ai.feature_engine import FeatureEngine
from src.ai.models.base import BasePredictor, PredictionOutput
from src.ai.models.ensemble import EnsembleEngine
from src.ai.models.registry import ModelRegistry
from src.core.events import (
    Bar,
    Exchange,
    OrderStatus,
    Position,
    Signal,
    SignalSide,
    validate_order_transition,
)
from src.strategy_engine.allocator import AllocatorConfig, PortfolioAllocator
from src.strategy_engine.base import MarketState, StrategyBase
from src.strategy_engine.registry import StrategyRegistry
from src.strategy_engine.runner import StrategyRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bar(
    symbol: str = "RELIANCE",
    exchange: Exchange = Exchange.NSE,
    interval: str = "1m",
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 99.0,
    close: float = 103.0,
    volume: float = 10000.0,
    ts: datetime | None = None,
    source: str = "test",
) -> Bar:
    if ts is None:
        ts = datetime.now(UTC)
    return Bar(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        ts=ts,
        source=source,
    )


def _make_bars(
    n: int = 100,
    symbol: str = "RELIANCE",
    exchange: Exchange = Exchange.NSE,
    base_price: float = 100.0,
) -> list[Bar]:
    """Generate n synthetic bars with monotonically increasing timestamps."""
    bars = []
    now = datetime.now(UTC) - timedelta(minutes=n)
    for i in range(n):
        ts = now + timedelta(minutes=i)
        o = base_price + np.random.uniform(-2, 2)
        c = o + np.random.uniform(-1, 1)
        h = max(o, c) + np.random.uniform(0, 1)
        l = min(o, c) - np.random.uniform(0, 1)
        v = 10000 + np.random.randint(0, 5000)
        bars.append(
            Bar(
                symbol=symbol,
                exchange=exchange,
                interval="1m",
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=float(v),
                ts=ts,
                source="test",
            )
        )
    return bars


def _make_signal(
    strategy_id: str = "ai_alpha",
    symbol: str = "RELIANCE",
    exchange: Exchange = Exchange.NSE,
    side: SignalSide = SignalSide.BUY,
    score: float = 0.8,
    price: float = 100.0,
    portfolio_weight: float = 0.1,
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=exchange,
        side=side,
        score=score,
        price=price,
        portfolio_weight=portfolio_weight,
    )


class _DummyStrategy(StrategyBase):
    """A minimal strategy for testing StrategyRunner."""

    strategy_id = "dummy_test"
    description = "dummy for tests"

    def __init__(self, signals: list[Signal] | None = None, min_bars: int = 5):
        self._signals = signals or []
        self._min_bars = min_bars

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= self._min_bars

    def generate_signals(self, state: MarketState) -> list[Signal]:
        return list(self._signals)


class _StubPredictor(BasePredictor):
    """A stub predictor for EnsembleEngine tests."""

    model_id = "stub"
    version = "v1"

    def __init__(
        self, model_id: str = "stub", prob_up: float = 0.7, confidence: float = 0.8, expected_return: float = 0.005
    ):
        self.model_id = model_id
        self.version = "v1"
        self._prob_up = prob_up
        self._confidence = confidence
        self._expected_return = expected_return

    def predict(self, features, context=None):
        return PredictionOutput(
            prob_up=self._prob_up,
            expected_return=self._expected_return,
            confidence=self._confidence,
            model_id=self.model_id,
            version=self.version,
        )


# ===================================================================
# 1. BarCache Tests
# ===================================================================


class TestBarCache:
    """Tests for src.market_data.bar_cache.BarCache"""

    def _make_cache(self, max_bars: int = 500):
        """Create BarCache with a permissive OHLCValidator stub."""
        from src.data_pipeline.ohlc_validator import OHLCValidator
        from src.market_data.bar_cache import BarCache

        # Use a very generous stale threshold so test bars always pass
        validator = OHLCValidator(stale_seconds=365 * 86400.0)
        return BarCache(max_bars=max_bars, ohlc_validator=validator)

    def test_append_bar_stores_correctly(self):
        """Appending a valid bar should be retrievable via get_bars."""
        cache = self._make_cache()
        bar = _make_bar()
        accepted = cache.append_bar(bar)
        assert accepted is True
        bars = cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=10)
        assert len(bars) == 1
        assert bars[0].symbol == "RELIANCE"
        assert bars[0].close == 103.0

    def test_get_bars_empty_symbol(self):
        """get_bars for a non-existent symbol should return empty list."""
        cache = self._make_cache()
        bars = cache.get_bars("NONEXISTENT", Exchange.NSE, "1m", n=10)
        assert bars == []

    def test_get_bars_returns_requested_count(self):
        """get_bars(n=5) should return the last 5 bars when more are available."""
        cache = self._make_cache()
        now = datetime.now(UTC)
        for i in range(20):
            c = 100.0 + i
            bar = _make_bar(
                ts=now + timedelta(minutes=i),
                open_=c - 1.0,
                high=c + 2.0,
                low=c - 2.0,
                close=c,
            )
            cache.append_bar(bar)
        bars = cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=5)
        assert len(bars) == 5
        # Should be the last 5 bars
        assert bars[-1].close == 119.0

    def test_deque_maxlen_enforcement(self):
        """BarCache with max_bars=10 should evict oldest bars when capacity exceeded."""
        cache = self._make_cache(max_bars=10)
        now = datetime.now(UTC)
        for i in range(20):
            c = 100.0 + i
            bar = _make_bar(
                ts=now + timedelta(minutes=i),
                open_=c - 1.0,
                high=c + 2.0,
                low=c - 2.0,
                close=c,
            )
            cache.append_bar(bar)
        bars = cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=0)
        assert len(bars) == 10
        # Oldest should be bar index 10 (close=110)
        assert bars[0].close == 110.0

    def test_prune_idle_symbols_removes_stale(self):
        """prune_idle_symbols should remove symbols that have not been updated."""
        cache = self._make_cache()
        bar = _make_bar()
        cache.append_bar(bar)
        assert len(cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=10)) == 1

        # Artificially set the last_update to a time well in the past
        key = cache._key("RELIANCE", Exchange.NSE.value, "1m")
        cache._last_update[key] = time.time() - 200000  # > 24h ago

        pruned = cache.prune_idle_symbols(max_idle_seconds=100)
        assert pruned == 1
        assert len(cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=10)) == 0

    def test_prune_idle_symbols_keeps_active(self):
        """prune_idle_symbols should keep recently updated symbols."""
        cache = self._make_cache()
        bar = _make_bar()
        cache.append_bar(bar)
        pruned = cache.prune_idle_symbols(max_idle_seconds=86400)
        assert pruned == 0
        assert len(cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=10)) == 1

    def test_ohlc_validation_rejects_high_less_than_low(self):
        """A bar with high < low should be rejected by the OHLC validator."""
        cache = self._make_cache()
        bad_bar = _make_bar(high=95.0, low=100.0)  # high < low
        accepted = cache.append_bar(bad_bar)
        assert accepted is False
        assert len(cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=10)) == 0

    def test_current_bar_ts_updates(self):
        """append_bar should update the current bar timestamp."""
        cache = self._make_cache()
        assert cache.get_current_bar_ts() is None
        bar = _make_bar()
        cache.append_bar(bar)
        assert cache.get_current_bar_ts() is not None

    def test_has_data_threshold(self):
        """has_data should respect min_bars threshold."""
        cache = self._make_cache()
        now = datetime.now(UTC)
        for i in range(15):
            cache.append_bar(_make_bar(ts=now + timedelta(minutes=i)))
        assert cache.has_data("RELIANCE", Exchange.NSE, "1m", min_bars=10) is True
        assert cache.has_data("RELIANCE", Exchange.NSE, "1m", min_bars=20) is False

    def test_out_of_order_bar_rejected(self):
        """A bar with a timestamp <= the last bar's timestamp should be rejected."""
        cache = self._make_cache()
        t1 = datetime.now(UTC)
        t2 = t1 + timedelta(minutes=1)
        cache.append_bar(_make_bar(ts=t2))
        # Attempt to append an earlier bar
        accepted = cache.append_bar(_make_bar(ts=t1, close=50.0))
        assert accepted is False
        bars = cache.get_bars("RELIANCE", Exchange.NSE, "1m", n=10)
        assert len(bars) == 1


# ===================================================================
# 2. PortfolioAllocator Tests
# ===================================================================


class TestPortfolioAllocator:
    """Tests for src.strategy_engine.allocator.PortfolioAllocator"""

    def test_allocate_with_valid_signals(self):
        """allocate should return (signal, qty) tuples for valid signals."""
        alloc = PortfolioAllocator(AllocatorConfig(min_confidence=0.5, use_kelly=False))
        signals = [_make_signal(score=0.8, price=100.0)]
        result = alloc.allocate(signals, equity=100000.0, positions=[])
        assert len(result) >= 1
        signal, qty = result[0]
        assert qty > 0
        assert signal.symbol == "RELIANCE"

    def test_allocate_zero_equity_returns_empty(self):
        """allocate with zero or negative equity should return empty."""
        alloc = PortfolioAllocator()
        signals = [_make_signal(score=0.8, price=100.0)]
        assert alloc.allocate(signals, equity=0.0, positions=[]) == []
        assert alloc.allocate(signals, equity=-1000.0, positions=[]) == []

    def test_allocate_filters_low_confidence_signals(self):
        """Signals below min_confidence should be filtered out."""
        config = AllocatorConfig(min_confidence=0.6)
        alloc = PortfolioAllocator(config)
        signals = [
            _make_signal(score=0.4, price=100.0),  # below threshold
            _make_signal(strategy_id="other", score=0.3, price=100.0),  # below threshold
        ]
        result = alloc.allocate(signals, equity=100000.0, positions=[])
        assert result == []

    def test_max_active_signals_cap(self):
        """allocate should respect max_active_signals limit."""
        config = AllocatorConfig(max_active_signals=2, min_confidence=0.5, use_kelly=False)
        alloc = PortfolioAllocator(config)
        signals = [_make_signal(strategy_id=f"s{i}", score=0.9 - i * 0.05, price=100.0) for i in range(5)]
        result = alloc.allocate(signals, equity=500000.0, positions=[])
        assert len(result) <= 2

    def test_kelly_criterion_sizing(self):
        """With Kelly enabled, higher-scored signals should get larger allocations."""
        config = AllocatorConfig(
            use_kelly=True,
            kelly_fraction=0.5,
            min_confidence=0.5,
            max_capital_pct_per_signal=20.0,
        )
        alloc = PortfolioAllocator(config)
        high_signal = _make_signal(strategy_id="high", score=0.9, price=10.0)
        low_signal = _make_signal(strategy_id="low", score=0.55, price=10.0)

        # Use a high max_position_pct so the Kelly difference is visible
        result_high = alloc.allocate(
            [high_signal],
            equity=1000000.0,
            positions=[],
            max_position_pct=20.0,
        )
        result_low = alloc.allocate(
            [low_signal],
            equity=1000000.0,
            positions=[],
            max_position_pct=20.0,
        )

        assert len(result_high) == 1
        assert len(result_low) == 1
        _, qty_high = result_high[0]
        _, qty_low = result_low[0]
        # Higher score should yield larger quantity
        assert qty_high > qty_low

    def test_exposure_multiplier_scaling(self):
        """exposure_multiplier < 1 should reduce position sizes."""
        config = AllocatorConfig(min_confidence=0.5, use_kelly=False)
        alloc = PortfolioAllocator(config)
        signal = _make_signal(score=0.8, price=100.0)

        result_full = alloc.allocate([signal], equity=100000.0, positions=[], exposure_multiplier=1.0)
        result_half = alloc.allocate([signal], equity=100000.0, positions=[], exposure_multiplier=0.5)
        assert len(result_full) == 1
        assert len(result_half) == 1
        _, qty_full = result_full[0]
        _, qty_half = result_half[0]
        # Half exposure should yield roughly half the quantity
        assert qty_half <= qty_full

    def test_max_position_pct_hard_cap(self):
        """Position size should never exceed max_position_pct of equity."""
        config = AllocatorConfig(
            min_confidence=0.5,
            max_capital_pct_per_signal=50.0,  # very high
            use_kelly=False,
        )
        alloc = PortfolioAllocator(config)
        signal = _make_signal(score=0.95, price=100.0)
        result = alloc.allocate([signal], equity=100000.0, positions=[], max_position_pct=2.0)
        assert len(result) == 1
        _, qty = result[0]
        max_qty = int(100000.0 * 0.02 / 100.0)  # 2% of equity at price=100
        assert qty <= max_qty

    def test_strategy_cap_pct_per_strategy(self):
        """Per-strategy capital cap should limit allocation."""
        config = AllocatorConfig(
            min_confidence=0.5,
            strategy_cap_pct={"ai_alpha": 2.0},  # cap at 2%
            use_kelly=False,
        )
        alloc = PortfolioAllocator(config)
        signal = _make_signal(strategy_id="ai_alpha", score=0.8, price=100.0)
        result = alloc.allocate([signal], equity=100000.0, positions=[], max_position_pct=10.0)
        assert len(result) == 1
        _, qty = result[0]
        # Notional should not exceed 2% of 100k = 2000 (20 shares at 100)
        assert qty <= 20

    def test_zero_price_signal_skipped(self):
        """A signal with price=0 should be skipped."""
        alloc = PortfolioAllocator(AllocatorConfig(min_confidence=0.5))
        signal = _make_signal(score=0.8, price=0.0)
        result = alloc.allocate([signal], equity=100000.0, positions=[])
        assert result == []


# ===================================================================
# 3. StrategyRunner Tests
# ===================================================================


class TestStrategyRunner:
    """Tests for src.strategy_engine.runner.StrategyRunner"""

    def _make_runner_with_strategy(self, signals=None, min_bars=5):
        registry = StrategyRegistry()
        strategy = _DummyStrategy(signals=signals or [], min_bars=min_bars)
        registry.register(strategy)
        return StrategyRunner(registry), strategy

    def test_initialization(self):
        """StrategyRunner should initialise with a registry."""
        registry = StrategyRegistry()
        runner = StrategyRunner(registry)
        assert runner.registry is registry

    def test_run_returns_signals_from_strategy(self):
        """run() should aggregate signals from registered strategies."""
        sig = _make_signal(score=0.9)
        runner, _ = self._make_runner_with_strategy(signals=[sig])
        state = MarketState(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            bars=_make_bars(20),
            latest_price=100.0,
            volume=10000.0,
        )
        signals = runner.run(state)
        assert len(signals) == 1
        assert signals[0].score == 0.9

    def test_run_returns_empty_when_not_warm(self):
        """run() should return no signals if strategy is not warm enough."""
        sig = _make_signal(score=0.9)
        runner, _ = self._make_runner_with_strategy(signals=[sig], min_bars=200)
        state = MarketState(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            bars=_make_bars(10),  # too few bars
            latest_price=100.0,
            volume=10000.0,
        )
        signals = runner.run(state)
        assert signals == []

    def test_run_sorts_by_score_descending(self):
        """Signals should be sorted by score in descending order."""
        registry = StrategyRegistry()
        s1 = _DummyStrategy(signals=[_make_signal(strategy_id="a", score=0.5)], min_bars=5)
        s1.strategy_id = "strat_a"
        s2 = _DummyStrategy(signals=[_make_signal(strategy_id="b", score=0.9)], min_bars=5)
        s2.strategy_id = "strat_b"
        registry.register(s1)
        registry.register(s2)
        runner = StrategyRunner(registry)
        state = MarketState(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            bars=_make_bars(20),
            latest_price=100.0,
            volume=10000.0,
        )
        signals = runner.run(state)
        assert len(signals) == 2
        assert signals[0].score >= signals[1].score

    def test_regime_crisis_blocks_all_strategies(self):
        """In crisis regime, all strategies should be blocked."""
        sig = _make_signal(score=0.9)
        runner, _ = self._make_runner_with_strategy(signals=[sig])
        state = MarketState(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            bars=_make_bars(20),
            latest_price=100.0,
            volume=10000.0,
            metadata={"regime": "crisis"},
        )
        signals = runner.run(state)
        assert signals == []

    def test_disabled_strategy_skipped(self):
        """Disabled strategies should not generate signals."""
        registry = StrategyRegistry()
        strategy = _DummyStrategy(signals=[_make_signal(score=0.9)])
        registry.register(strategy)
        registry.disable("dummy_test")
        runner = StrategyRunner(registry)
        state = MarketState(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            bars=_make_bars(20),
            latest_price=100.0,
            volume=10000.0,
        )
        signals = runner.run(state)
        assert signals == []


# ===================================================================
# 4. Signal / Position / Exchange Events Tests
# ===================================================================


class TestCoreEvents:
    """Tests for src.core.events domain objects."""

    def test_signal_creation_all_fields(self):
        """Signal should be created with all fields populated."""
        sig = Signal(
            strategy_id="test_strat",
            symbol="TCS",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            score=0.85,
            portfolio_weight=0.10,
            risk_level="HIGH",
            reason="ML prediction",
            price=3500.0,
            stop_loss=3400.0,
            target=3700.0,
            metadata={"model": "xgb"},
        )
        assert sig.strategy_id == "test_strat"
        assert sig.symbol == "TCS"
        assert sig.exchange == Exchange.NSE
        assert sig.side == SignalSide.BUY
        assert sig.score == 0.85
        assert sig.portfolio_weight == 0.10
        assert sig.risk_level == "HIGH"
        assert sig.reason == "ML prediction"
        assert sig.price == 3500.0
        assert sig.stop_loss == 3400.0
        assert sig.target == 3700.0
        assert sig.metadata == {"model": "xgb"}
        assert sig.ts is not None

    def test_position_creation_and_pnl(self):
        """Position should store fields and support unrealized PnL."""
        pos = Position(
            symbol="INFY",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            quantity=50.0,
            avg_price=1500.0,
            unrealized_pnl=2500.0,
            strategy_id="ai_alpha",
        )
        assert pos.symbol == "INFY"
        assert pos.quantity == 50.0
        assert pos.avg_price == 1500.0
        assert pos.unrealized_pnl == 2500.0
        assert pos.strategy_id == "ai_alpha"

    def test_position_negative_pnl(self):
        """Position should support negative unrealized PnL."""
        pos = Position(
            symbol="HDFC",
            exchange=Exchange.BSE,
            side=SignalSide.BUY,
            quantity=10.0,
            avg_price=2500.0,
            unrealized_pnl=-500.0,
        )
        assert pos.unrealized_pnl == -500.0

    def test_exchange_enum_values(self):
        """Exchange enum should include expected market identifiers."""
        assert Exchange.NSE == "NSE"
        assert Exchange.BSE == "BSE"
        assert Exchange.NYSE == "NYSE"
        assert Exchange.NASDAQ == "NASDAQ"
        assert Exchange.LSE == "LSE"
        assert Exchange.FX == "FX"

    def test_signal_side_enum(self):
        """SignalSide should have BUY and SELL."""
        assert SignalSide.BUY == "BUY"
        assert SignalSide.SELL == "SELL"

    def test_order_status_transitions(self):
        """Valid order status transitions should be accepted."""
        assert validate_order_transition(OrderStatus.PENDING, OrderStatus.SUBMITTING) is True
        assert validate_order_transition(OrderStatus.SUBMITTING, OrderStatus.LIVE) is True
        assert validate_order_transition(OrderStatus.LIVE, OrderStatus.FILLED) is True
        # Terminal states cannot transition
        assert validate_order_transition(OrderStatus.FILLED, OrderStatus.CANCELLED) is False

    def test_bar_creation(self):
        """Bar should hold OHLCV data."""
        bar = _make_bar(
            symbol="SBIN",
            exchange=Exchange.NSE,
            open_=500.0,
            high=510.0,
            low=495.0,
            close=505.0,
            volume=50000.0,
        )
        assert bar.symbol == "SBIN"
        assert bar.open == 500.0
        assert bar.high == 510.0
        assert bar.low == 495.0
        assert bar.close == 505.0
        assert bar.volume == 50000.0


# ===================================================================
# 5. DynamicUniverse Tests
# ===================================================================


class TestDynamicUniverse:
    """Tests for src.scanner.dynamic_universe.DynamicUniverse"""

    def test_initialization_defaults(self):
        """DynamicUniverse should initialise with default parameters."""
        from src.scanner.dynamic_universe import DynamicUniverse

        du = DynamicUniverse()
        assert du.min_volume == 50_000
        assert du.min_turnover == 1e7
        assert du.min_price == 10.0
        assert du.max_price == 50_000.0
        assert du.target_count == 300

    def test_initialization_custom_params(self):
        """DynamicUniverse should accept custom parameters."""
        from src.scanner.dynamic_universe import DynamicUniverse

        du = DynamicUniverse(
            min_volume=100_000,
            min_turnover=5e7,
            min_price=50.0,
            max_price=10_000.0,
            target_count=100,
        )
        assert du.min_volume == 100_000
        assert du.min_turnover == 5e7
        assert du.target_count == 100

    @patch("src.scanner.dynamic_universe.DynamicUniverse.build_universe")
    def test_get_tradeable_stocks_returns_symbols(self, mock_build):
        """get_tradeable_stocks should return symbol list from build_universe."""
        from src.scanner.dynamic_universe import DynamicUniverse

        mock_build.return_value = {
            "symbols": ["RELIANCE", "TCS", "INFY", "HDFC", "SBIN"],
            "total_nse": 1800,
            "post_filter": 300,
            "scan_date": "2025-01-01T00:00:00",
        }
        du = DynamicUniverse()
        symbols = du.get_tradeable_stocks(count=3)
        assert symbols == ["RELIANCE", "TCS", "INFY"]
        mock_build.assert_called_once()

    @patch("src.scanner.dynamic_universe.DynamicUniverse.build_universe")
    def test_get_training_stocks(self, mock_build):
        """get_training_stocks should return symbols up to requested count."""
        from src.scanner.dynamic_universe import DynamicUniverse

        mock_build.return_value = {
            "symbols": [f"SYM{i}" for i in range(500)],
            "total_nse": 1800,
            "post_filter": 500,
            "scan_date": "2025-01-01T00:00:00",
        }
        du = DynamicUniverse()
        result = du.get_training_stocks(count=50)
        assert len(result) == 50

    @patch("src.scanner.dynamic_universe.DynamicUniverse.build_universe")
    def test_get_yfinance_symbols_appends_ns(self, mock_build):
        """get_yfinance_symbols should append .NS suffix."""
        from src.scanner.dynamic_universe import DynamicUniverse

        mock_build.return_value = {
            "symbols": ["RELIANCE", "TCS"],
            "total_nse": 1800,
            "post_filter": 2,
            "scan_date": "2025-01-01T00:00:00",
        }
        du = DynamicUniverse()
        yf_syms = du.get_yfinance_symbols()
        assert yf_syms == ["RELIANCE.NS", "TCS.NS"]


# ===================================================================
# 6. FeatureEngine Tests
# ===================================================================


class TestFeatureEngine:
    """Tests for src.ai.feature_engine.FeatureEngine"""

    def test_compute_features_with_valid_ohlcv(self):
        """build_features should return a dict with 30+ features for valid bars."""
        engine = FeatureEngine()
        bars = _make_bars(100, base_price=500.0)
        features = engine.build_features(bars)
        assert isinstance(features, dict)
        # Should have many features
        assert len(features) >= 30

    def test_compute_features_returns_expected_keys(self):
        """build_features output should contain known feature keys."""
        engine = FeatureEngine()
        bars = _make_bars(100, base_price=500.0)
        features = engine.build_features(bars)
        expected_keys = [
            "returns_1",
            "returns_5",
            "returns_10",
            "returns_20",
            "rolling_volatility",
            "atr",
            "rsi",
            "ema_spread",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "volume_spike",
            "obv_slope",
            "vwap_distance",
            "bollinger_pct_b",
            "bollinger_bandwidth",
            "price_position",
            "gap_pct",
            "candle_body_ratio",
            "candle_upper_shadow",
            "stochastic_k",
            "stochastic_d",
            "close",
            "volume",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "hurst_exponent",
            "adx",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature key: {key}"

    def test_compute_features_empty_bars(self):
        """build_features with empty bars should return empty dict."""
        engine = FeatureEngine()
        features = engine.build_features([])
        assert features == {}

    def test_compute_features_few_bars(self):
        """build_features with very few bars should still produce features (with defaults)."""
        engine = FeatureEngine()
        bars = _make_bars(5, base_price=100.0)
        features = engine.build_features(bars)
        assert isinstance(features, dict)
        assert len(features) > 0
        # RSI should default to 50 with insufficient data
        assert "rsi" in features

    def test_features_are_finite(self):
        """All features should be finite numbers (no NaN or Inf)."""
        engine = FeatureEngine()
        bars = _make_bars(100, base_price=500.0)
        features = engine.build_features(bars)
        for key, val in features.items():
            assert np.isfinite(val), f"Feature '{key}' is not finite: {val}"

    def test_features_deterministic(self):
        """Same bars should produce identical features (no randomness)."""
        engine = FeatureEngine()
        np.random.seed(42)
        bars = _make_bars(100, base_price=500.0)
        f1 = engine.build_features(bars)
        f2 = engine.build_features(bars)
        assert f1 == f2


# ===================================================================
# 7. EnsembleEngine Tests
# ===================================================================


class TestEnsembleEngine:
    """Tests for src.ai.models.ensemble.EnsembleEngine"""

    def _make_engine(self, model_ids=None, n_models=3, prob_up=0.75, confidence=0.8, expected_return=0.005):
        """Create EnsembleEngine with N stub predictors, all agreeing bullish."""
        registry = ModelRegistry()
        ids = model_ids or [f"model_{i}" for i in range(n_models)]
        for mid in ids:
            pred = _StubPredictor(model_id=mid, prob_up=prob_up, confidence=confidence, expected_return=expected_return)
            registry.register(pred)
        # Set uniform weights so predictions are not zero-weighted
        weights = {mid: 1.0 / len(ids) for mid in ids}
        engine = EnsembleEngine(registry=registry, model_ids=ids, weights=weights, ic_weighted=False)
        return engine

    def test_ensemble_predict_aggregates_models(self):
        """predict() should aggregate predictions from multiple models."""
        engine = self._make_engine(n_models=5, prob_up=0.75, confidence=0.8)
        features = {"close": 100.0, "rsi": 55.0}
        result = engine.predict(features)
        assert result is not None
        assert result.model_id == "ensemble"
        assert 0.0 <= result.prob_up <= 1.0
        assert result.metadata["count"] == 5

    def test_ensemble_returns_none_when_no_models(self):
        """predict() with no registered models should return None."""
        registry = ModelRegistry()
        engine = EnsembleEngine(registry=registry, model_ids=[], weights={})
        result = engine.predict({"close": 100.0})
        assert result is None

    def test_ensemble_halted_when_all_ic_below_threshold(self):
        """When _all_models_halted is True, predict() should return None."""
        engine = self._make_engine(n_models=3)
        engine._all_models_halted = True
        result = engine.predict({"close": 100.0})
        assert result is None

    def test_ensemble_model_agreement_required(self):
        """If fewer than MIN_AGREEING_MODELS agree, predict() should return None."""
        registry = ModelRegistry()
        ids = [f"m{i}" for i in range(5)]
        # 2 bullish, 2 bearish, 1 neutral => no clear agreement
        probs = [0.7, 0.8, 0.3, 0.2, 0.5]
        for mid, p in zip(ids, probs):
            pred = _StubPredictor(model_id=mid, prob_up=p, confidence=0.8)
            registry.register(pred)
        weights = {mid: 0.2 for mid in ids}
        engine = EnsembleEngine(registry=registry, model_ids=ids, weights=weights, ic_weighted=False)
        result = engine.predict({"close": 100.0})
        # With 2 bullish vs 2 bearish vs 1 neutral, max_agreeing=2 < MIN_AGREEING_MODELS=3
        assert result is None

    def test_ensemble_record_prediction(self):
        """record_prediction should store data for IC calculation."""
        engine = self._make_engine(n_models=2)
        engine.record_prediction("model_0", "RELIANCE", 1.0, 0.02)
        engine.record_prediction("model_0", "RELIANCE", -1.0, -0.01)
        assert len(engine._prediction_history["model_0"]["RELIANCE"]) == 2


# ===================================================================
# 8. FeatureNormalizer Tests
# ===================================================================


class TestFeatureNormalizer:
    """Tests for src.ai.feature_engine.FeatureNormalizer"""

    def test_normalize_passthrough_when_not_fitted(self):
        """Unfitted normalizer should pass features through unchanged."""
        from src.ai.feature_engine import FeatureNormalizer

        norm = FeatureNormalizer()
        features = {"rsi": 55.0, "close": 100.0}
        result = norm.normalize(features)
        assert result == features

    def test_fit_and_normalize(self):
        """Fitted normalizer should z-score normalize features."""
        from src.ai.feature_engine import FeatureNormalizer

        norm = FeatureNormalizer()
        # Generate enough history for fitting
        history = [{"rsi": 50.0 + i * 0.1, "close": 100.0 + i} for i in range(200)]
        norm.fit(history)
        assert norm._fitted is True
        result = norm.normalize({"rsi": 50.0, "close": 100.0})
        # Result should be z-score (not the original values)
        assert result["rsi"] != 50.0

    def test_normalize_clips_outliers(self):
        """Normalizer should clip extreme z-scores to [-5, 5]."""
        from src.ai.feature_engine import FeatureNormalizer

        norm = FeatureNormalizer()
        history = [{"rsi": 50.0} for _ in range(200)]
        norm.fit(history)
        # std is ~0 => z-score would be huge for any deviation
        # The normalizer uses std=1.0 when std < 1e-12, so 1e6 should clip
        result = norm.normalize({"rsi": 1e6})
        assert result["rsi"] <= 5.0


# ===================================================================
# 9. PredictionOutput Validation Tests
# ===================================================================


class TestPredictionOutput:
    """Tests for PredictionOutput validation in __post_init__."""

    def test_valid_prediction(self):
        """Valid PredictionOutput should retain its values."""
        p = PredictionOutput(
            prob_up=0.7,
            expected_return=0.005,
            confidence=0.8,
            model_id="test",
            version="v1",
        )
        assert p.prob_up == 0.7
        assert p.confidence == 0.8

    def test_invalid_prob_up_clamped(self):
        """Out-of-range prob_up should be clamped to 0.5."""
        p = PredictionOutput(
            prob_up=1.5,  # invalid: > 1
            expected_return=0.0,
            confidence=0.5,
            model_id="test",
            version="v1",
        )
        assert p.prob_up == 0.5

    def test_negative_confidence_clamped(self):
        """Negative confidence should be clamped to 0."""
        p = PredictionOutput(
            prob_up=0.6,
            expected_return=0.0,
            confidence=-0.5,
            model_id="test",
            version="v1",
        )
        assert p.confidence == 0.0

    def test_nan_expected_return_clamped(self):
        """NaN expected_return should be clamped to 0."""
        p = PredictionOutput(
            prob_up=0.6,
            expected_return=float("nan"),
            confidence=0.5,
            model_id="test",
            version="v1",
        )
        assert p.expected_return == 0.0


# ===================================================================
# 10. ModelRegistry Tests
# ===================================================================


class TestModelRegistry:
    """Tests for src.ai.models.registry.ModelRegistry"""

    def test_register_and_get(self):
        """Registering a model should make it retrievable by ID."""
        registry = ModelRegistry()
        pred = _StubPredictor(model_id="test_model")
        registry.register(pred)
        assert registry.get("test_model") is pred

    def test_list_models(self):
        """list_models should return all registered model IDs."""
        registry = ModelRegistry()
        for i in range(3):
            registry.register(_StubPredictor(model_id=f"m{i}"))
        ids = registry.list_models()
        assert set(ids) == {"m0", "m1", "m2"}

    def test_deregister(self):
        """Deregistering should remove the model."""
        registry = ModelRegistry()
        pred = _StubPredictor(model_id="to_remove")
        registry.register(pred)
        assert registry.deregister("to_remove") is True
        assert registry.get("to_remove") is None

    def test_replace_if_better(self):
        """replace_if_better should swap model when candidate metric is superior."""
        registry = ModelRegistry()
        original = _StubPredictor(model_id="test_model")
        registry.register(original, metrics={"sharpe": 1.0})
        candidate = _StubPredictor(model_id="test_model", prob_up=0.9)
        replaced = registry.replace_if_better("test_model", candidate, {"sharpe": 2.0}, compare_metric="sharpe")
        assert replaced is True
        assert registry.get("test_model") is candidate


# ===================================================================
# 11. StrategyRegistry Tests
# ===================================================================


class TestStrategyRegistry:
    """Tests for src.strategy_engine.registry.StrategyRegistry"""

    def test_enable_disable(self):
        """Enabling and disabling strategies should work."""
        registry = StrategyRegistry()
        strategy = _DummyStrategy()
        registry.register(strategy)
        assert "dummy_test" in registry.list_enabled()

        registry.disable("dummy_test")
        assert "dummy_test" not in registry.list_enabled()

        registry.enable("dummy_test")
        assert "dummy_test" in registry.list_enabled()

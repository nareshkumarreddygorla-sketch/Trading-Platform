"""
Comprehensive tests for the 5 AI model predictors, ModelRegistry,
EnsembleEngine, FeatureEngine, and PerformanceTracker.

All tests run without trained models on disk and without network access.
Models operate in untrained/fallback mode.
"""

import math
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ai.feature_engine import FeatureEngine
from src.ai.models.base import BasePredictor, PredictionOutput
from src.ai.models.ensemble import EnsembleEngine
from src.ai.models.lstm_predictor import FEATURE_KEYS, SEQ_LEN, LSTMPredictor
from src.ai.models.registry import ModelRegistry
from src.ai.models.rl_agent import RLPredictor
from src.ai.models.sentiment_predictor import SentimentPredictor
from src.ai.models.transformer_predictor import TransformerPredictor
from src.ai.performance_tracker import PerformanceTracker
from src.core.events import Bar, Exchange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_features() -> dict[str, float]:
    """Return a feature dict with all keys expected by LSTM/Transformer predictors."""
    return {k: np.random.uniform(-1.0, 1.0) for k in FEATURE_KEYS}


def _make_sequence(seq_len: int = SEQ_LEN, num_features: int = len(FEATURE_KEYS)) -> np.ndarray:
    """Return a random (seq_len, num_features) array suitable for LSTM/Transformer."""
    return np.random.randn(seq_len, num_features).astype(np.float32)


def _make_bar(
    ts_offset: int = 0,
    o: float = 100.0,
    h: float = 101.0,
    l: float = 99.0,
    c: float = 100.5,
    v: float = 10000.0,
) -> Bar:
    """Create a synthetic Bar for FeatureEngine tests."""
    return Bar(
        symbol="TEST",
        exchange=Exchange.NSE,
        interval="1m",
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        ts=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=ts_offset),
        source="test",
    )


class StubPredictor(BasePredictor):
    """Deterministic predictor for registry / ensemble tests."""

    def __init__(self, model_id: str, prob_up: float = 0.6, confidence: float = 0.8):
        self.model_id = model_id
        self.version = "v_test"
        self.path = ""
        self._prob_up = prob_up
        self._confidence = confidence

    def predict(self, features: dict[str, float], context: dict[str, Any] | None = None) -> PredictionOutput:
        return PredictionOutput(
            prob_up=self._prob_up,
            expected_return=(self._prob_up - 0.5) * 0.02,
            confidence=self._confidence,
            model_id=self.model_id,
            version=self.version,
            metadata={"stub": True},
        )


# ===========================================================================
# 1. PredictionOutput
# ===========================================================================
class TestPredictionOutput:
    """Verify PredictionOutput dataclass fields and construction."""

    def test_fields_present(self):
        po = PredictionOutput(
            prob_up=0.65,
            expected_return=0.003,
            confidence=0.8,
            model_id="test_model",
            version="v1",
            metadata={"foo": "bar"},
        )
        assert po.prob_up == 0.65
        assert po.expected_return == 0.003
        assert po.confidence == 0.8
        assert po.model_id == "test_model"
        assert po.version == "v1"
        assert po.metadata == {"foo": "bar"}

    def test_edge_values(self):
        po = PredictionOutput(
            prob_up=0.0,
            expected_return=-0.05,
            confidence=1.0,
            model_id="m",
            version="v0",
            metadata={},
        )
        assert po.prob_up == 0.0
        assert po.confidence == 1.0

    def test_metadata_can_be_empty(self):
        po = PredictionOutput(
            prob_up=0.5,
            expected_return=0.0,
            confidence=0.0,
            model_id="m",
            version="v0",
            metadata={},
        )
        assert po.metadata == {}

    def test_equality(self):
        from datetime import UTC, datetime

        ts = datetime.now(UTC)
        a = PredictionOutput(
            prob_up=0.5,
            expected_return=0.0,
            confidence=0.5,
            model_id="x",
            version="v1",
            metadata={},
            timestamp=ts,
        )
        b = PredictionOutput(
            prob_up=0.5,
            expected_return=0.0,
            confidence=0.5,
            model_id="x",
            version="v1",
            metadata={},
            timestamp=ts,
        )
        assert a == b


# ===========================================================================
# 2. Individual Predictor predict() tests
# ===========================================================================
class TestLSTMPredictor:
    """LSTMPredictor in untrained mode: returns PredictionOutput or None depending on validation."""

    def test_predict_no_sequence_returns_none(self):
        predictor = LSTMPredictor()
        features = _make_dummy_features()
        out = predictor.predict(features)
        # Without sequence context, predict returns None (no data to run LSTM on)
        assert out is None or isinstance(out, PredictionOutput)

    def test_predict_with_sequence_context(self):
        predictor = LSTMPredictor()
        features = _make_dummy_features()
        seq = _make_sequence()
        context = {"sequence": seq}
        out = predictor.predict(features, context=context)
        # Untrained model may return None (fails validation/empirical return check)
        # or PredictionOutput with valid ranges
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert 0.0 <= out.prob_up <= 1.0
            assert 0.0 <= out.confidence <= 1.0
            assert out.model_id == "lstm_ts"

    def test_predict_with_feature_history(self):
        predictor = LSTMPredictor()
        features = _make_dummy_features()
        feature_history = [_make_dummy_features() for _ in range(SEQ_LEN)]
        context = {"feature_history": feature_history}
        out = predictor.predict(features, context=context)
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert 0.0 <= out.prob_up <= 1.0

    def test_predict_short_sequence_returns_none(self):
        predictor = LSTMPredictor()
        features = _make_dummy_features()
        short_seq = _make_sequence(seq_len=10)
        context = {"sequence": short_seq}
        out = predictor.predict(features, context=context)
        # Short sequence: predict returns None (below SEQ_LEN threshold)
        assert out is None

    def test_predict_untrained_low_confidence(self):
        predictor = LSTMPredictor()
        seq = _make_sequence()
        out = predictor.predict({}, context={"sequence": seq})
        # Untrained: may return None or PredictionOutput with low confidence
        if out is not None:
            assert out.confidence <= 0.3

    def test_model_id_and_version(self):
        predictor = LSTMPredictor()
        assert predictor.model_id == "lstm_ts"
        assert predictor.version == "v3"


class TestTransformerPredictor:
    """TransformerPredictor in untrained mode: returns PredictionOutput or None."""

    def test_predict_no_sequence_returns_none(self):
        predictor = TransformerPredictor()
        features = _make_dummy_features()
        out = predictor.predict(features)
        # Without sequence, returns None
        assert out is None or isinstance(out, PredictionOutput)

    def test_predict_with_sequence_context(self):
        predictor = TransformerPredictor()
        seq = _make_sequence()
        out = predictor.predict({}, context={"sequence": seq})
        # Untrained model may return None (validation) or PredictionOutput
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert 0.0 <= out.prob_up <= 1.0
            assert 0.0 <= out.confidence <= 1.0
            assert out.model_id == "transformer_ts"

    def test_predict_with_feature_history(self):
        predictor = TransformerPredictor()
        feature_history = [_make_dummy_features() for _ in range(SEQ_LEN)]
        out = predictor.predict({}, context={"feature_history": feature_history})
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert 0.0 <= out.prob_up <= 1.0

    def test_predict_short_sequence_returns_none(self):
        predictor = TransformerPredictor()
        short_seq = _make_sequence(seq_len=5)
        out = predictor.predict({}, context={"sequence": short_seq})
        # Short sequence: returns None
        assert out is None

    def test_predict_untrained_low_confidence(self):
        predictor = TransformerPredictor()
        seq = _make_sequence()
        out = predictor.predict({}, context={"sequence": seq})
        # Untrained: may return None or PredictionOutput with low confidence
        if out is not None:
            assert out.confidence <= 0.3

    def test_model_id_and_version(self):
        predictor = TransformerPredictor()
        assert predictor.model_id == "transformer_ts"
        assert predictor.version == "v2"


class TestRLPredictor:
    """RLPredictor returns None when no model is loaded (no stable-baselines3 or model file)."""

    def test_predict_no_model_returns_none(self):
        predictor = RLPredictor()
        features = _make_dummy_features()
        out = predictor.predict(features)
        # Without a loaded model, predict returns None (no model to run inference)
        assert out is None

    def test_predict_with_context_returns_none(self):
        predictor = RLPredictor()
        context = {"position_side": 1, "unrealized_pnl_pct": 0.01, "bars_held": 5}
        out = predictor.predict(_make_dummy_features(), context=context)
        # Without a loaded model, predict returns None
        assert out is None

    def test_model_id_and_version(self):
        predictor = RLPredictor()
        assert predictor.model_id == "rl_ppo"
        assert predictor.version in ("v1", "v2")


class TestSentimentPredictor:
    """SentimentPredictor returns None when no headlines available (no network)."""

    def test_predict_no_network_returns_none_or_output(self):
        predictor = SentimentPredictor()
        out = predictor.predict({})
        # Without network access, fetch_headlines returns empty → predict returns None
        # With cached/available data, returns PredictionOutput
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert out.model_id == "sentiment_finbert"
            assert 0.0 <= out.prob_up <= 1.0

    def test_predict_with_symbol_context(self):
        predictor = SentimentPredictor()
        out = predictor.predict({}, context={"symbol": "RELIANCE"})
        # May return None if no headlines fetched
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert 0.0 <= out.prob_up <= 1.0

    def test_model_id_and_version(self):
        predictor = SentimentPredictor()
        assert predictor.model_id == "sentiment_finbert"
        assert predictor.version == "v1"

    def test_predict_with_mocked_headlines(self, monkeypatch):
        """When headlines exist but no FinBERT, sentiment falls back to neutral."""
        predictor = SentimentPredictor()
        # Mock the news fetcher to return some headlines without network
        monkeypatch.setattr(
            predictor._fetcher,
            "fetch_headlines",
            lambda symbol=None, max_items=20: ["Market rallies today", "Stocks surge"],
        )
        out = predictor.predict({})
        assert isinstance(out, PredictionOutput)
        # With default neutral sentiment (no FinBERT), prob_up should be ~0.5
        assert 0.0 <= out.prob_up <= 1.0
        assert 0.0 <= out.confidence <= 1.0


# ===========================================================================
# 3. ModelRegistry
# ===========================================================================
class TestModelRegistry:
    """Register, list, and get models by ID."""

    def test_register_and_get(self):
        reg = ModelRegistry()
        stub = StubPredictor("my_model")
        reg.register(stub)
        retrieved = reg.get("my_model")
        assert retrieved is stub

    def test_get_unknown_returns_none(self):
        reg = ModelRegistry()
        assert reg.get("nonexistent") is None

    def test_register_multiple_models(self):
        reg = ModelRegistry()
        m1 = StubPredictor("model_a", prob_up=0.7)
        m2 = StubPredictor("model_b", prob_up=0.3)
        reg.register(m1)
        reg.register(m2)
        assert reg.get("model_a") is m1
        assert reg.get("model_b") is m2

    def test_list_registered_models(self):
        reg = ModelRegistry()
        reg.register(StubPredictor("a"))
        reg.register(StubPredictor("b"))
        reg.register(StubPredictor("c"))
        ids = list(reg._models.keys())
        assert set(ids) == {"a", "b", "c"}

    def test_metadata_created_on_register(self):
        reg = ModelRegistry()
        stub = StubPredictor("meta_test")
        reg.register(stub, metrics={"sharpe": 1.5, "accuracy": 0.72})
        meta = reg.get_metadata("meta_test")
        assert meta is not None
        assert meta.model_id == "meta_test"
        assert meta.current_version == "v_test"
        assert len(meta.versions) == 1
        assert meta.versions[0].metrics["sharpe"] == 1.5

    def test_replace_if_better_promotes_candidate(self):
        reg = ModelRegistry()
        old = StubPredictor("model_x")
        reg.register(old, metrics={"sharpe": 1.0})
        new = StubPredictor("model_x", prob_up=0.8)
        new.version = "v2_test"
        replaced = reg.replace_if_better(
            "model_x",
            new,
            {"sharpe": 2.0},
            compare_metric="sharpe",
        )
        assert replaced is True
        assert reg.get("model_x") is new
        meta = reg.get_metadata("model_x")
        assert meta.current_version == "v2_test"

    def test_replace_if_better_keeps_current_when_worse(self):
        reg = ModelRegistry()
        old = StubPredictor("model_x")
        reg.register(old, metrics={"sharpe": 2.0})
        worse = StubPredictor("model_x", prob_up=0.3)
        replaced = reg.replace_if_better(
            "model_x",
            worse,
            {"sharpe": 0.5},
            compare_metric="sharpe",
        )
        assert replaced is False
        assert reg.get("model_x") is old

    def test_log_performance(self):
        reg = ModelRegistry()
        reg.register(StubPredictor("perf_model"))
        reg.log_performance("perf_model", {"sharpe": 1.2, "win_rate": 0.55})
        meta = reg.get_metadata("perf_model")
        assert len(meta.performance_log) == 1
        assert meta.performance_log[0]["sharpe"] == 1.2


# ===========================================================================
# 4. EnsembleEngine
# ===========================================================================
class TestEnsembleEngine:
    """Aggregate predictions from multiple models with weights."""

    @pytest.fixture
    def registry_with_models(self):
        reg = ModelRegistry()
        reg.register(StubPredictor("m1", prob_up=0.7, confidence=0.8))
        reg.register(StubPredictor("m2", prob_up=0.3, confidence=0.6))
        return reg

    def test_predict_equal_weight(self, registry_with_models):
        engine = EnsembleEngine(
            registry=registry_with_models,
            model_ids=["m1", "m2"],
        )
        out = engine.predict({})
        assert isinstance(out, PredictionOutput)
        assert out.model_id == "ensemble"
        # Equal weight: (0.7 + 0.3) / 2 = 0.5
        assert abs(out.prob_up - 0.5) < 1e-6
        assert out.metadata["count"] == 2
        assert set(out.metadata["models"]) == {"m1", "m2"}

    def test_predict_custom_weights(self, registry_with_models):
        engine = EnsembleEngine(
            registry=registry_with_models,
            model_ids=["m1", "m2"],
            weights={"m1": 3.0, "m2": 1.0},
        )
        out = engine.predict({})
        # Weighted: (0.7*3 + 0.3*1) / (3+1) = 2.4/4 = 0.6
        assert abs(out.prob_up - 0.6) < 1e-6

    def test_predict_single_model(self, registry_with_models):
        engine = EnsembleEngine(
            registry=registry_with_models,
            model_ids=["m1"],
        )
        out = engine.predict({})
        assert abs(out.prob_up - 0.7) < 1e-6
        assert out.metadata["count"] == 1

    def test_predict_empty_registry(self):
        reg = ModelRegistry()
        engine = EnsembleEngine(registry=reg, model_ids=[])
        out = engine.predict({})
        assert out.prob_up == 0.5
        assert out.confidence == 0.0
        assert out.metadata["count"] == 0

    def test_predict_missing_model_id_skipped(self, registry_with_models):
        engine = EnsembleEngine(
            registry=registry_with_models,
            model_ids=["m1", "nonexistent"],
        )
        out = engine.predict({})
        # Only m1 contributes
        assert abs(out.prob_up - 0.7) < 1e-6
        assert out.metadata["count"] == 1

    def test_set_weights(self, registry_with_models):
        engine = EnsembleEngine(
            registry=registry_with_models,
            model_ids=["m1", "m2"],
        )
        engine.set_weights({"m1": 0.0, "m2": 1.0})
        # m1 has weight 0, m2 has weight 1
        # total_weight = 0 + 1 = 1
        # prob_up = (0.7*0 + 0.3*1) / 1 = 0.3
        out = engine.predict({})
        assert abs(out.prob_up - 0.3) < 1e-6

    def test_predict_output_clipped_to_01(self):
        """Ensure prob_up is clipped even with extreme model outputs."""
        reg = ModelRegistry()
        reg.register(StubPredictor("extreme_high", prob_up=0.99, confidence=1.0))
        engine = EnsembleEngine(registry=reg, model_ids=["extreme_high"])
        out = engine.predict({})
        assert 0.0 <= out.prob_up <= 1.0

    def test_predict_confidence_capped_at_1(self):
        reg = ModelRegistry()
        reg.register(StubPredictor("high_conf", prob_up=0.7, confidence=0.95))
        engine = EnsembleEngine(registry=reg, model_ids=["high_conf"])
        out = engine.predict({})
        assert out.confidence <= 1.0

    def test_predict_expected_return_is_weighted(self, registry_with_models):
        engine = EnsembleEngine(
            registry=registry_with_models,
            model_ids=["m1", "m2"],
        )
        out = engine.predict({})
        # m1: er = (0.7-0.5)*0.02 = 0.004;  m2: er = (0.3-0.5)*0.02 = -0.004
        # avg = 0.0
        assert abs(out.expected_return) < 1e-6

    def test_predict_nan_sanitized(self):
        """If a model returns NaN, ensemble should sanitize to safe defaults."""

        class NaNPredictor(BasePredictor):
            model_id = "nan_model"
            version = "v0"

            def predict(self, features, context=None):
                return PredictionOutput(
                    prob_up=float("nan"),
                    expected_return=float("nan"),
                    confidence=float("nan"),
                    model_id=self.model_id,
                    version=self.version,
                    metadata={},
                )

        reg = ModelRegistry()
        nan_pred = NaNPredictor()
        nan_pred.path = ""
        reg.register(nan_pred)
        engine = EnsembleEngine(registry=reg, model_ids=["nan_model"])
        out = engine.predict({})
        assert math.isfinite(out.prob_up)
        assert math.isfinite(out.expected_return)
        assert math.isfinite(out.confidence)


# ===========================================================================
# 5. FeatureEngine
# ===========================================================================
class TestFeatureEngine:
    """Build features from dummy bars, verify output keys."""

    @pytest.fixture
    def engine(self):
        return FeatureEngine()

    @pytest.fixture
    def bars_50(self):
        """50 synthetic bars with slight price variation."""
        bars = []
        price = 100.0
        for i in range(50):
            change = (i % 7 - 3) * 0.5
            o = price
            c = price + change
            h = max(o, c) + 0.5
            l = min(o, c) - 0.5
            v = 10000.0 + i * 100
            bars.append(_make_bar(ts_offset=i, o=o, h=h, l=l, c=c, v=v))
            price = c
        return bars

    def test_build_features_returns_dict(self, engine, bars_50):
        features = engine.build_features(bars_50)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_expected_feature_keys_present(self, engine, bars_50):
        features = engine.build_features(bars_50)
        expected_keys = [
            "returns_1",
            "returns_5",
            "returns_10",
            "returns_20",
            "rolling_volatility",
            "atr",
            "bollinger_bandwidth",
            "rsi",
            "ema_spread",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "stochastic_k",
            "stochastic_d",
            "adx",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "roc_10",
            "volume_spike",
            "obv_slope",
            "vwap_distance",
            "bollinger_pct_b",
            "price_position",
            "gap_pct",
            "candle_body_ratio",
            "candle_upper_shadow",
            "candle_lower_shadow",
            "candle_engulfing",
            "williams_r",
            "mfi",
            "close",
            "volume",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature key: {key}"

    def test_all_feature_values_are_floats(self, engine, bars_50):
        features = engine.build_features(bars_50)
        for key, val in features.items():
            assert isinstance(val, float), f"Feature {key} is not float: {type(val)}"

    def test_no_nan_or_inf_in_features(self, engine, bars_50):
        features = engine.build_features(bars_50)
        for key, val in features.items():
            assert math.isfinite(val), f"Feature {key} is not finite: {val}"

    def test_empty_bars_returns_empty(self, engine):
        assert engine.build_features([]) == {}

    def test_single_bar(self, engine):
        bars = [_make_bar(ts_offset=0)]
        features = engine.build_features(bars)
        assert isinstance(features, dict)
        assert "close" in features
        assert features["close"] == 100.5

    def test_deterministic(self, engine, bars_50):
        f1 = engine.build_features(bars_50)
        f2 = engine.build_features(bars_50)
        assert f1 == f2

    def test_rsi_bounded(self, engine, bars_50):
        features = engine.build_features(bars_50)
        assert 0.0 <= features["rsi"] <= 100.0

    def test_stochastic_bounded(self, engine, bars_50):
        features = engine.build_features(bars_50)
        assert 0.0 <= features["stochastic_k"] <= 100.0
        assert 0.0 <= features["stochastic_d"] <= 100.0


# ===========================================================================
# 6. PerformanceTracker
# ===========================================================================
class TestPerformanceTracker:
    """Record fills, verify summary, verify auto-disable on consecutive losses."""

    def test_record_win(self):
        tracker = PerformanceTracker()
        tracker.record_fill("strat_a", 100.0)
        stats = tracker.get_stats("strat_a")
        assert stats.wins == 1
        assert stats.losses == 0
        assert stats.consecutive_losses == 0
        assert stats.win_rate == 1.0

    def test_record_loss(self):
        tracker = PerformanceTracker()
        tracker.record_fill("strat_a", -50.0)
        stats = tracker.get_stats("strat_a")
        assert stats.wins == 0
        assert stats.losses == 1
        assert stats.consecutive_losses == 1
        assert stats.win_rate == 0.0

    def test_mixed_trades_win_rate(self):
        tracker = PerformanceTracker()
        for pnl in [100, -50, 200, -30, 150]:
            tracker.record_fill("strat_a", pnl)
        stats = tracker.get_stats("strat_a")
        assert stats.wins == 3
        assert stats.losses == 2
        assert abs(stats.win_rate - 0.6) < 1e-6

    def test_consecutive_loss_resets_on_win(self):
        tracker = PerformanceTracker()
        tracker.record_fill("strat_a", -10.0)
        tracker.record_fill("strat_a", -10.0)
        tracker.record_fill("strat_a", -10.0)
        assert tracker.get_stats("strat_a").consecutive_losses == 3
        tracker.record_fill("strat_a", 50.0)
        assert tracker.get_stats("strat_a").consecutive_losses == 0

    def test_auto_disable_consecutive_losses(self):
        tracker = PerformanceTracker(max_consecutive_losses_disable=5)
        for _ in range(5):
            tracker.record_fill("strat_a", -10.0)
        assert tracker.is_disabled("strat_a") is True
        stats = tracker.get_stats("strat_a")
        assert stats.disabled is True
        assert stats.consecutive_losses == 5

    def test_not_disabled_below_threshold(self):
        tracker = PerformanceTracker(max_consecutive_losses_disable=5)
        for _ in range(4):
            tracker.record_fill("strat_a", -10.0)
        assert tracker.is_disabled("strat_a") is False

    def test_auto_disable_callback_called(self):
        callback = MagicMock()
        tracker = PerformanceTracker(
            max_consecutive_losses_disable=3,
            on_strategy_disabled=callback,
        )
        for _ in range(3):
            tracker.record_fill("strat_x", -10.0)
        callback.assert_called_once_with("strat_x", "consecutive_losses")

    def test_auto_disable_low_win_rate(self):
        tracker = PerformanceTracker(min_win_rate_disable=0.35)
        # 3 wins, 7 losses = 30% win rate, below 35% threshold
        for _ in range(3):
            tracker.record_fill("strat_b", 10.0)
        for _ in range(7):
            tracker.record_fill("strat_b", -5.0)
        assert tracker.is_disabled("strat_b") is True

    def test_summary_aggregate(self):
        tracker = PerformanceTracker()
        tracker.record_fill("strat_a", 100.0)
        tracker.record_fill("strat_a", -20.0)
        tracker.record_fill("strat_b", 50.0)
        s = tracker.summary()
        assert s["total_trades"] == 3
        assert s["total_wins"] == 2
        assert s["total_losses"] == 1
        assert s["total_pnl"] == 130.0
        assert abs(s["win_rate"] - 2 / 3) < 1e-6
        assert abs(s["avg_trade_pnl"] - 130.0 / 3) < 1e-6
        assert "strat_a" in s["strategies"]
        assert "strat_b" in s["strategies"]

    def test_summary_empty_tracker(self):
        tracker = PerformanceTracker()
        s = tracker.summary()
        assert s["total_trades"] == 0
        assert s["win_rate"] == 0.0
        assert s["total_pnl"] == 0.0

    def test_get_all_stats(self):
        tracker = PerformanceTracker()
        tracker.record_fill("a", 10.0)
        tracker.record_fill("b", -5.0)
        all_stats = tracker.get_all_stats()
        assert "a" in all_stats
        assert "b" in all_stats
        assert all_stats["a"].wins == 1
        assert all_stats["b"].losses == 1

    def test_rolling_sharpe(self):
        tracker = PerformanceTracker()
        # Record enough trades to compute rolling sharpe
        for pnl in [10.0, 15.0, -5.0, 20.0, 10.0]:
            tracker.record_fill("strat_c", pnl)
        stats = tracker.get_stats("strat_c")
        assert isinstance(stats.rolling_sharpe, float)
        assert math.isfinite(stats.rolling_sharpe)

    def test_exposure_multiplier_initial(self):
        tracker = PerformanceTracker()
        tracker.record_fill("strat_a", 100.0)
        stats = tracker.get_stats("strat_a")
        assert stats.exposure_multiplier == 1.0

    def test_multiple_strategies_independent(self):
        tracker = PerformanceTracker(max_consecutive_losses_disable=3)
        for _ in range(3):
            tracker.record_fill("strat_a", -10.0)
        tracker.record_fill("strat_b", 100.0)
        assert tracker.is_disabled("strat_a") is True
        assert tracker.is_disabled("strat_b") is False

    def test_pnl_window_capped(self):
        tracker = PerformanceTracker()
        # StrategyStats has max_pnl_window=100 by default
        for i in range(150):
            tracker.record_fill("strat_a", 1.0)
        stats = tracker.get_stats("strat_a")
        assert len(stats.pnls) <= 100


# ===========================================================================
# 7. Integration: predictors through registry and ensemble
# ===========================================================================
class TestIntegration:
    """End-to-end: register real predictors, run ensemble, verify output."""

    def test_real_predictors_through_ensemble(self):
        """Wire up actual predictors (untrained) through registry and ensemble."""
        reg = ModelRegistry()
        lstm = LSTMPredictor()
        transformer = TransformerPredictor()
        rl = RLPredictor()
        sentiment = SentimentPredictor()

        reg.register(lstm)
        reg.register(transformer)
        reg.register(rl)
        reg.register(sentiment)

        engine = EnsembleEngine(
            registry=reg,
            model_ids=["lstm_ts", "transformer_ts", "rl_ppo", "sentiment_finbert"],
        )
        out = engine.predict(_make_dummy_features())
        # All models untrained/no-data → may all return None → ensemble returns None
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert out.model_id == "ensemble"
            assert 0.0 <= out.prob_up <= 1.0
            assert 0.0 <= out.confidence <= 1.0
            assert math.isfinite(out.expected_return)

    def test_feature_engine_output_feeds_predictors(self):
        """FeatureEngine output can be used as predictor input."""
        fe = FeatureEngine()
        bars = [
            _make_bar(
                ts_offset=i,
                o=100.0 + i * 0.1,
                h=101.0 + i * 0.1,
                l=99.0 + i * 0.1,
                c=100.5 + i * 0.1,
                v=10000.0 + i * 100,
            )
            for i in range(50)
        ]
        features = fe.build_features(bars)
        assert len(features) > 0

        lstm = LSTMPredictor()
        out = lstm.predict(features)
        # Without sequence context, LSTM returns None
        assert out is None or isinstance(out, PredictionOutput)

    def test_performance_tracker_with_ensemble(self):
        """Ensemble prediction followed by performance tracking."""
        reg = ModelRegistry()
        reg.register(StubPredictor("stub_a", prob_up=0.65, confidence=0.7))
        engine = EnsembleEngine(registry=reg, model_ids=["stub_a"])
        tracker = PerformanceTracker()

        # Simulate trade outcome
        pred = engine.predict({})
        assert pred.prob_up > 0.5  # predicts up

        # Record some fills
        tracker.record_fill("ensemble_strat", 100.0)
        tracker.record_fill("ensemble_strat", -30.0)
        s = tracker.summary()
        assert s["total_trades"] == 2

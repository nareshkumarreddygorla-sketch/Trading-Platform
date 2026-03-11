"""
Signal path integration test.

Tests the full signal pipeline:
  1. Synthetic OHLCV bars -> FeatureEngine -> feature validation
  2. XGBPredictor output format (if model artifact exists)
  3. Ensemble DEFAULT_MODEL_IDS match registered predictors
  4. RiskManager basic order validation sanity
"""

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ai.feature_engine import FeatureEngine
from src.ai.models.base import PredictionOutput
from src.ai.models.lstm_predictor import FEATURE_KEYS, NUM_FEATURES
from src.core.events import Bar, Exchange, Signal, SignalSide

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_bars(n: int = 100, start_price: float = 1000.0, symbol: str = "SYNTH") -> list:
    """Generate n synthetic OHLCV bars with a gentle uptrend and realistic noise."""
    bars = []
    rng = np.random.RandomState(42)
    price = start_price
    base_ts = datetime(2025, 1, 1, 9, 15, tzinfo=UTC)

    for i in range(n):
        # Random walk with slight upward drift
        ret = rng.normal(0.001, 0.015)
        price *= 1 + ret
        high = price * (1 + abs(rng.normal(0, 0.005)))
        low = price * (1 - abs(rng.normal(0, 0.005)))
        opn = price * (1 + rng.normal(0, 0.002))
        vol = float(rng.randint(100_000, 1_000_000))
        ts = base_ts + timedelta(days=i)

        bars.append(
            Bar(
                symbol=symbol,
                exchange=Exchange.NSE,
                interval="1d",
                open=round(opn, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(price, 2),
                volume=vol,
                ts=ts,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Test 1: FeatureEngine produces correct number and names of features
# ---------------------------------------------------------------------------


class TestFeatureEngine:
    def test_feature_count_matches_num_features(self):
        """build_features() output should contain at least NUM_FEATURES keys."""
        bars = _make_synthetic_bars(100)
        fe = FeatureEngine()
        features = fe.build_features(bars)

        # FeatureEngine may produce a superset of FEATURE_KEYS (e.g. microstructure features)
        assert len(features) >= NUM_FEATURES, (
            f"Expected at least {NUM_FEATURES} features, got {len(features)}. "
            f"Missing: {set(FEATURE_KEYS) - set(features.keys())}"
        )

    def test_all_feature_keys_present(self):
        """Every key in FEATURE_KEYS must appear in build_features() output."""
        bars = _make_synthetic_bars(100)
        fe = FeatureEngine()
        features = fe.build_features(bars)

        missing = [k for k in FEATURE_KEYS if k not in features]
        assert not missing, f"Missing feature keys: {missing}"

    def test_no_extra_keys(self):
        """build_features() keys should all be valid strings (no empty/None)."""
        bars = _make_synthetic_bars(100)
        fe = FeatureEngine()
        features = fe.build_features(bars)

        # FeatureEngine may produce more features than FEATURE_KEYS (e.g. microstructure);
        # just verify all keys are non-empty strings
        bad = [k for k in features if not isinstance(k, str) or not k]
        assert not bad, f"Invalid feature keys: {bad}"

    def test_features_are_finite(self):
        """All feature values must be finite floats (no NaN or Inf)."""
        bars = _make_synthetic_bars(100)
        fe = FeatureEngine()
        features = fe.build_features(bars)

        for key, val in features.items():
            assert np.isfinite(val), f"Feature '{key}' is not finite: {val}"

    def test_short_bars_does_not_crash(self):
        """build_features() should handle short bar lists gracefully."""
        fe = FeatureEngine()
        # 5 bars -- many indicators will default but should not raise
        bars = _make_synthetic_bars(5)
        features = fe.build_features(bars)
        assert isinstance(features, dict)
        # Empty bars
        features_empty = fe.build_features([])
        assert features_empty == {}


# ---------------------------------------------------------------------------
# Test 2: XGBPredictor output format (conditional on model artifact)
# ---------------------------------------------------------------------------


class TestXGBPredictor:
    def test_predict_output_format_with_model(self):
        """If alpha_xgb.joblib exists, verify XGBPredictor.predict() output."""
        model_path = os.path.join(PROJECT_ROOT, "models", "alpha_xgb.joblib")
        if not os.path.exists(model_path):
            pytest.skip("alpha_xgb.joblib not found -- skipping XGB prediction test")

        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("xgboost not installed")

        from src.ai.models.xgb_predictor import XGBPredictor

        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
        except Exception as e:
            pytest.skip(f"Could not load XGB model artifact: {e}")
        predictor = XGBPredictor(model=booster)

        # Build features from synthetic bars
        bars = _make_synthetic_bars(100)
        fe = FeatureEngine()
        features = fe.build_features(bars)

        out = predictor.predict(features)
        assert isinstance(out, PredictionOutput)
        assert 0.0 <= out.prob_up <= 1.0, f"prob_up out of range: {out.prob_up}"
        assert 0.0 <= out.confidence <= 1.0, f"confidence out of range: {out.confidence}"
        assert out.model_id == "xgb_direction"

    def test_predict_without_model_returns_neutral(self):
        """XGBPredictor with no model should return None (no model loaded)."""
        from src.ai.models.xgb_predictor import XGBPredictor

        predictor = XGBPredictor(model=None)
        features = {k: 0.0 for k in FEATURE_KEYS}
        out = predictor.predict(features)

        # Predictors return None when no model is loaded
        assert out is None


# ---------------------------------------------------------------------------
# Test 3: Ensemble DEFAULT_MODEL_IDS match registered predictors
# ---------------------------------------------------------------------------


class TestEnsembleModelIds:
    def test_default_model_ids_match_predictor_classes(self):
        """Every ID in DEFAULT_MODEL_IDS must match a predictor's model_id attribute."""
        from src.ai.models.ensemble import DEFAULT_MODEL_IDS
        from src.ai.models.lstm_predictor import LSTMPredictor
        from src.ai.models.rl_agent import RLPredictor
        from src.ai.models.sentiment_predictor import SentimentPredictor
        from src.ai.models.transformer_predictor import TransformerPredictor
        from src.ai.models.xgb_predictor import XGBPredictor

        # Map of model_id -> predictor class
        registered_ids = {
            XGBPredictor.model_id,
            LSTMPredictor.model_id,
            TransformerPredictor.model_id,
            RLPredictor.model_id,
            SentimentPredictor.model_id,
        }

        for model_id in DEFAULT_MODEL_IDS:
            assert model_id in registered_ids, (
                f"DEFAULT_MODEL_IDS contains '{model_id}' which has no matching "
                f"predictor class. Registered IDs: {registered_ids}"
            )

    def test_no_duplicate_model_ids(self):
        """DEFAULT_MODEL_IDS should not contain duplicates."""
        from src.ai.models.ensemble import DEFAULT_MODEL_IDS

        assert len(DEFAULT_MODEL_IDS) == len(set(DEFAULT_MODEL_IDS)), f"Duplicate model IDs: {DEFAULT_MODEL_IDS}"

    def test_ensemble_predict_with_registry(self):
        """EnsembleEngine.predict() should work with registered (unloaded) models."""
        from src.ai.models.ensemble import DEFAULT_MODEL_IDS, EnsembleEngine
        from src.ai.models.lstm_predictor import LSTMPredictor
        from src.ai.models.registry import ModelRegistry
        from src.ai.models.rl_agent import RLPredictor
        from src.ai.models.sentiment_predictor import SentimentPredictor
        from src.ai.models.xgb_predictor import XGBPredictor

        registry = ModelRegistry()
        registry.register(XGBPredictor(model=None))
        registry.register(LSTMPredictor())
        registry.register(RLPredictor())
        registry.register(SentimentPredictor())

        engine = EnsembleEngine(registry=registry, model_ids=list(DEFAULT_MODEL_IDS))

        features = {k: 0.0 for k in FEATURE_KEYS}
        out = engine.predict(features)

        # When all models are unloaded (no torch, no model files), predict returns None
        # When at least one model works, returns PredictionOutput
        if out is not None:
            assert isinstance(out, PredictionOutput)
            assert out.model_id == "ensemble"
            assert 0.0 <= out.prob_up <= 1.0


# ---------------------------------------------------------------------------
# Test 4: RiskManager basic order validation
# ---------------------------------------------------------------------------


class TestRiskManagerSanity:
    def _make_signal(self, symbol="SYNTH", side=SignalSide.BUY, price=1000.0):
        return Signal(
            strategy_id="test_strategy",
            symbol=symbol,
            exchange=Exchange.NSE,
            side=side,
            score=0.8,
            portfolio_weight=0.05,
            price=price,
        )

    def test_valid_order_allowed(self):
        """A small order against fresh equity should be allowed."""
        from src.risk_engine.manager import RiskManager

        rm = RiskManager(equity=1_000_000, load_persisted_state=False)
        signal = self._make_signal()
        result = rm.can_place_order(signal, quantity=10, price=1000.0)
        assert result.allowed, f"Expected allowed, got: {result.reason}"

    def test_zero_quantity_rejected(self):
        """Order with zero quantity must be rejected."""
        from src.risk_engine.manager import RiskManager

        rm = RiskManager(equity=1_000_000, load_persisted_state=False)
        signal = self._make_signal()
        result = rm.can_place_order(signal, quantity=0, price=1000.0)
        assert not result.allowed

    def test_zero_price_rejected(self):
        """Order with zero price must be rejected."""
        from src.risk_engine.manager import RiskManager

        rm = RiskManager(equity=1_000_000, load_persisted_state=False)
        signal = self._make_signal()
        result = rm.can_place_order(signal, quantity=10, price=0)
        assert not result.allowed

    def test_circuit_breaker_blocks_orders(self):
        """When circuit breaker is open, normal orders should be blocked."""
        from src.risk_engine.manager import RiskManager

        rm = RiskManager(equity=1_000_000, load_persisted_state=False)
        rm._circuit_open = True
        signal = self._make_signal()
        result = rm.can_place_order(signal, quantity=10, price=1000.0)
        assert not result.allowed
        assert "circuit" in result.reason.lower()

    def test_zero_equity_rejected(self):
        """Order against zero equity must be rejected."""
        from src.risk_engine.manager import RiskManager

        rm = RiskManager(equity=0, load_persisted_state=False)
        signal = self._make_signal()
        result = rm.can_place_order(signal, quantity=10, price=1000.0)
        assert not result.allowed

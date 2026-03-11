"""
AI signal engine: load pre-trained model, predict(features) -> probability, convert to Signal.
Integrates via StrategyRunner; rule-based and AI strategies coexist.

CRITICAL: Prediction uses saved feature_names from training metadata to guarantee
correct feature ordering at inference. This prevents the misalignment bug where
sorted(keys()) would produce different ordering than training.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.events import Exchange, Signal, SignalSide
from src.strategy_engine.base import MarketState, StrategyBase

logger = logging.getLogger(__name__)


def _get_feature_engine():
    from src.ai.feature_engine import FeatureEngine
    return FeatureEngine()


class AlphaModel:
    """
    Load pre-trained model; predict(features) -> probability; convert to Signal.
    strategy_id identifies this model in the registry.

    Uses saved feature_names from training metadata to ensure exact feature
    ordering at inference time — prevents the critical misalignment bug.
    """

    def __init__(self, strategy_id: str = "ai_alpha", model_path: Optional[str] = None):
        self.strategy_id = strategy_id
        self.model_path = model_path
        self._model = None
        self._feature_names: Optional[List[str]] = None  # from training metadata
        if model_path:
            self.load(model_path)

    def load(self, path: str) -> bool:
        """Load pre-trained model + metadata from path. Returns True on success."""
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning("AlphaModel load: path not found %s", path)
                return False
            import joblib
            # Patch __main__ so pickle can find EnsembleClassifier saved by training script
            import sys
            _main = sys.modules.get("__main__")
            _needs_patch = _main is not None and not hasattr(_main, "EnsembleClassifier")
            if _needs_patch:
                try:
                    from scripts.train_alpha_model import EnsembleClassifier
                    _main.EnsembleClassifier = EnsembleClassifier
                except ImportError:
                    # Define a minimal compatible class as fallback
                    class EnsembleClassifier:
                        """Minimal unpickle stub for soft-voting ensemble."""
                        def __init__(self, xgb_model=None, lgb_model=None, xgb_weight=0.6, lgb_weight=0.4):
                            self.xgb_model = xgb_model
                            self.lgb_model = lgb_model
                            self.xgb_weight = xgb_weight if lgb_model else 1.0
                            self.lgb_weight = lgb_weight if lgb_model else 0.0
                        def predict_proba(self, X):
                            xgb_proba = self.xgb_model.predict_proba(X)
                            if self.lgb_model is not None:
                                lgb_proba = self.lgb_model.predict_proba(X)
                                return self.xgb_weight * xgb_proba + self.lgb_weight * lgb_proba
                            return xgb_proba
                        def predict(self, X):
                            proba = self.predict_proba(X)
                            return (proba[:, 1] >= 0.5).astype(int)
                    _main.EnsembleClassifier = EnsembleClassifier
            self._model = joblib.load(path)
            self.model_path = path

            # Load feature metadata for correct ordering
            meta_path = path_obj.parent / (path_obj.stem + "_meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._feature_names = meta.get("feature_names")
                logger.info("AlphaModel loaded %s (%d features from metadata)",
                            path, len(self._feature_names) if self._feature_names else 0)
            else:
                logger.warning("AlphaModel: no metadata at %s, using sorted keys", meta_path)

            return True
        except Exception as e:
            logger.exception("AlphaModel load failed: %s", e)
            return False

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict probability (0-1) from feature dict.

        CRITICAL: Uses self._feature_names (from training metadata) for exact
        feature ordering. If metadata unavailable, falls back to sorted keys.
        """
        if self._model is not None:
            try:
                import numpy as np

                # Use training feature order if available (prevents misalignment)
                if self._feature_names is not None:
                    keys = self._feature_names
                else:
                    keys = sorted(features.keys())

                vec = np.array(
                    [[features.get(k, 0.0) for k in keys]],
                    dtype=np.float64
                )

                # Handle NaN/Inf from features
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

                pred = self._model.predict_proba(vec)
                if pred is not None and pred.shape[1] >= 2:
                    return float(pred[0][1])
                if hasattr(self._model, "predict"):
                    out = self._model.predict(vec)
                    return float(out[0]) if out is not None else 0.5
            except Exception as e:
                logger.exception("AlphaModel predict failed: %s", e)
                return 0.5

        # No model: deterministic fallback from RSI/ema_spread for testing
        rsi = features.get("rsi", 50.0)
        ema_spread = features.get("ema_spread", 0.0)
        macd_hist = features.get("macd_histogram", 0.0)
        stoch_k = features.get("stochastic_k", 50.0)
        adx = features.get("adx", 25.0)
        vwap_dist = features.get("vwap_distance", 0.0)

        # Multi-factor heuristic fallback
        score = 0.0
        if rsi < 30:
            score += 0.2
        elif rsi > 70:
            score -= 0.2

        if macd_hist > 0:
            score += 0.1
        elif macd_hist < 0:
            score -= 0.1

        if stoch_k < 20:
            score += 0.1
        elif stoch_k > 80:
            score -= 0.1

        if adx > 25:
            score += 0.05 * (1 if ema_spread > 0 else -1)

        score += 0.1 * max(-1, min(1, vwap_dist * 10))

        prob = 0.5 + score
        return max(0.0, min(1.0, prob))

    def to_signal(
        self,
        probability: float,
        symbol: str,
        exchange: Exchange,
        price: float,
        *,
        side: Optional[SignalSide] = None,
        strategy_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Signal:
        """
        Convert probability to Signal. P > 0.5 -> BUY, P < 0.5 -> SELL;
        confidence = |P - 0.5| * 2.
        """
        confidence = abs(probability - 0.5) * 2.0
        confidence = max(0.0, min(1.0, confidence))
        if side is None:
            side = SignalSide.BUY if probability >= 0.5 else SignalSide.SELL
        return Signal(
            strategy_id=strategy_id or self.strategy_id,
            symbol=symbol,
            exchange=exchange,
            side=side,
            score=confidence,
            portfolio_weight=confidence * 0.1,
            risk_level="NORMAL",
            reason="ai_alpha",
            price=price,
            ts=timestamp or datetime.now(timezone.utc),
            metadata={"probability": probability},
        )


class AlphaStrategy(StrategyBase):
    """
    Strategy plugin that uses FeatureEngine + AlphaModel. Register with StrategyRegistry
    so StrategyRunner runs it alongside rule-based strategies.
    """
    strategy_id: str = "ai_alpha"
    description: str = "AI alpha model (features -> probability -> signal)"

    def __init__(
        self,
        alpha_model: Optional[AlphaModel] = None,
        feature_engine: Optional[Any] = None,
        min_confidence: float = 0.5,
    ):
        self.alpha_model = alpha_model or AlphaModel(strategy_id=self.strategy_id)
        self.feature_engine = feature_engine or _get_feature_engine()
        self.min_confidence = min_confidence

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= 30  # Need 30 bars for all indicators

    def generate_signals(self, state: MarketState) -> List[Signal]:
        """Build features, predict, convert to Signal. Returns list of 0 or 1 signal."""
        if not self.warm(state):
            return []
        try:
            features = self.feature_engine.build_features(state.bars)

            # Inject market context if available
            try:
                from src.ai.market_context import fetch_market_context
                ctx = fetch_market_context()
                features.update(ctx)
            except Exception:
                pass  # Market context is optional

            prob = self.alpha_model.predict(features)
            signal = self.alpha_model.to_signal(
                prob, state.symbol, state.exchange, state.latest_price or features.get("close", 0.0),
                strategy_id=self.strategy_id,
            )
            if signal.score >= self.min_confidence:
                return [signal]
        except Exception as e:
            logger.exception("AlphaStrategy generate_signals failed: %s", e)
        return []

"""ML/AI strategy hooks: predictive models, RL agent, meta-optimizer. Real implementations."""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.events import Exchange, Signal, SignalSide
from .base import MarketState, StrategyBase

logger = logging.getLogger(__name__)


class MLPredictorStrategy(StrategyBase):
    """
    Ensemble ML predictor: LSTM + Transformer + XGBoost + Sentiment.
    Uses EnsembleEngine to aggregate predictions, emits signals when confidence > threshold.
    """
    strategy_id = "ml_predictor"
    description = "ML ensemble price direction predictor"

    def __init__(self, ensemble_engine=None, feature_engine=None,
                 confidence_threshold: float = 0.55, prob_threshold: float = 0.58):
        self._ensemble = ensemble_engine
        self._feature_engine = feature_engine
        self.confidence_threshold = confidence_threshold
        self.prob_threshold = prob_threshold
        # Feature history per symbol for sequence building
        self._feature_history: Dict[str, List[Dict[str, float]]] = {}

    def set_ensemble(self, ensemble_engine) -> None:
        self._ensemble = ensemble_engine

    def set_feature_engine(self, feature_engine) -> None:
        self._feature_engine = feature_engine

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= 60

    def generate_signals(self, state: MarketState) -> List[Signal]:
        if not self.warm(state) or self._ensemble is None:
            return []

        # Build features
        if self._feature_engine is not None:
            features = self._feature_engine.build_features(state.bars)
        elif state.metadata and "features" in state.metadata:
            features = state.metadata["features"]
        else:
            return []

        if not features:
            return []

        # Track feature history for sequence-based models (LSTM, Transformer)
        symbol = state.symbol
        if symbol not in self._feature_history:
            self._feature_history[symbol] = []
        self._feature_history[symbol].append(features)
        # Keep last 100 entries
        if len(self._feature_history[symbol]) > 100:
            self._feature_history[symbol] = self._feature_history[symbol][-100:]

        # Build context for models
        context = {
            "symbol": symbol,
            "feature_history": self._feature_history[symbol],
        }

        # Get ensemble prediction
        try:
            prediction = self._ensemble.predict(features, context)
        except Exception as e:
            logger.debug("Ensemble predict failed for %s: %s", symbol, e)
            return []

        # Check confidence threshold
        if prediction.confidence < self.confidence_threshold:
            return []

        # Determine signal direction
        signals = []
        if prediction.prob_up > self.prob_threshold:
            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                exchange=state.exchange,
                side=SignalSide.BUY,
                score=prediction.confidence,
                price=state.latest_price,
                portfolio_weight=min(0.15, prediction.confidence * 0.2),
                risk_level="NORMAL",
                metadata={
                    "prob_up": prediction.prob_up,
                    "expected_return": prediction.expected_return,
                    "models": prediction.metadata.get("models", []),
                    "model_count": prediction.metadata.get("count", 0),
                },
            ))
        elif prediction.prob_up < (1.0 - self.prob_threshold):
            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                exchange=state.exchange,
                side=SignalSide.SELL,
                score=prediction.confidence,
                price=state.latest_price,
                portfolio_weight=min(0.15, prediction.confidence * 0.2),
                risk_level="NORMAL",
                metadata={
                    "prob_up": prediction.prob_up,
                    "expected_return": prediction.expected_return,
                    "models": prediction.metadata.get("models", []),
                    "model_count": prediction.metadata.get("count", 0),
                },
            ))

        return signals


class RLAgentStrategy(StrategyBase):
    """
    Reinforcement learning agent (entry/exit).
    State: bars + position + PnL; action: hold/buy/sell.
    Uses trained PPO model from stable-baselines3.
    """
    strategy_id = "rl_agent"
    description = "RL dynamic entry/exit agent"

    def __init__(self, rl_predictor=None, feature_engine=None,
                 confidence_threshold: float = 0.5):
        self._rl = rl_predictor
        self._feature_engine = feature_engine
        self.confidence_threshold = confidence_threshold
        # Track positions per symbol for RL state
        self._positions: Dict[str, Dict[str, Any]] = {}

    def set_rl_predictor(self, rl_predictor) -> None:
        self._rl = rl_predictor

    def set_feature_engine(self, feature_engine) -> None:
        self._feature_engine = feature_engine

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= 50

    def generate_signals(self, state: MarketState) -> List[Signal]:
        if not self.warm(state) or self._rl is None:
            return []

        # Build features
        if self._feature_engine is not None:
            features = self._feature_engine.build_features(state.bars)
        elif state.metadata and "features" in state.metadata:
            features = state.metadata["features"]
        else:
            return []

        if not features:
            return []

        # Build context with position info
        symbol = state.symbol
        pos_info = self._positions.get(symbol, {})
        context = {
            "symbol": symbol,
            "position_side": pos_info.get("side", 0),
            "unrealized_pnl_pct": pos_info.get("unrealized_pnl_pct", 0.0),
            "bars_held": pos_info.get("bars_held", 0),
        }

        try:
            prediction = self._rl.predict(features, context)
        except Exception as e:
            logger.debug("RL predict failed for %s: %s", symbol, e)
            return []

        if prediction.confidence < self.confidence_threshold:
            return []

        action = prediction.metadata.get("action", 0)
        signals = []

        if action == 1:  # Buy
            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                exchange=state.exchange,
                side=SignalSide.BUY,
                score=prediction.confidence,
                price=state.latest_price,
                portfolio_weight=0.08,
                risk_level="NORMAL",
                metadata={"action": "buy", "rl_confidence": prediction.confidence},
            ))
            self._positions[symbol] = {"side": 1, "entry_price": state.latest_price, "bars_held": 0}

        elif action == 2:  # Sell
            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                exchange=state.exchange,
                side=SignalSide.SELL,
                score=prediction.confidence,
                price=state.latest_price,
                portfolio_weight=0.08,
                risk_level="NORMAL",
                metadata={"action": "sell", "rl_confidence": prediction.confidence},
            ))
            self._positions[symbol] = {"side": -1, "entry_price": state.latest_price, "bars_held": 0}

        else:  # Hold - increment bars_held
            if symbol in self._positions:
                self._positions[symbol]["bars_held"] = self._positions[symbol].get("bars_held", 0) + 1

        return signals


class MetaOptimizerStrategy(StrategyBase):
    """
    Meta-optimizer: tunes strategy parameters over rolling window.
    Outputs weights for ensemble of child strategies based on recent performance.
    """
    strategy_id = "meta_optimizer"
    description = "Parameter tuning over rolling window"

    def __init__(self, performance_tracker=None, strategy_registry=None):
        self._tracker = performance_tracker
        self._registry = strategy_registry

    def generate_signals(self, state: MarketState) -> List[Signal]:
        # Meta-optimizer doesn't generate direct signals.
        # It adjusts weights of other strategies via the performance tracker.
        if self._tracker and self._registry:
            try:
                stats = self._tracker.get_all_stats()
                for sid, s in stats.items():
                    win_rate = s.get("win_rate", 0.5)
                    if win_rate < 0.35:
                        self._registry.disable(sid)
                        logger.info("MetaOptimizer: disabled %s (win_rate=%.2f)", sid, win_rate)
            except Exception as e:
                logger.debug("MetaOptimizer failed: %s", e)
        return []

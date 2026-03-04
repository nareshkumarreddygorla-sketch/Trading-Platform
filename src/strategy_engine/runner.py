"""Run enabled strategies on market state; aggregate and rank signals.
Regime-aware: filters strategies by market regime when regime data is available.
"""
import logging
from typing import Dict, List, Optional, Set

from src.core.events import Signal

from .base import MarketState, StrategyBase
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)

# Regime → strategy mapping: which strategies are allowed in each regime
REGIME_STRATEGY_MAP: Dict[str, Set[str]] = {
    "trending_up": {"ai_alpha", "ema_crossover", "macd", "momentum_breakout", "ml_predictor", "rl_agent"},
    "trending_down": {"ai_alpha", "ema_crossover", "macd", "ml_predictor", "rl_agent"},
    "sideways": {"ai_alpha", "rsi", "mean_reversion", "ml_predictor"},
    "high_volatility": {"ai_alpha", "ml_predictor", "rl_agent"},
    "low_volatility": {"ai_alpha", "ema_crossover", "macd", "rsi", "mean_reversion", "ml_predictor"},
    "crisis": set(),  # No strategies in crisis mode
}


class StrategyRunner:
    """Consume market state; run all enabled strategies; return combined sorted signals.
    When regime data is present in state.metadata, filters strategies by regime compatibility.
    """

    def __init__(self, registry: StrategyRegistry):
        self.registry = registry

    def run(self, state: MarketState) -> List[Signal]:
        regime = (state.metadata or {}).get("regime", None) if state.metadata else None
        allowed_ids: Optional[Set[str]] = None
        if regime and regime in REGIME_STRATEGY_MAP:
            allowed_ids = REGIME_STRATEGY_MAP[regime]
            if not allowed_ids:
                logger.warning("Regime '%s' — all strategies blocked (crisis mode)", regime)
                return []

        signals: List[Signal] = []
        for strategy in self.registry.get_enabled_strategies():
            # Skip strategies not compatible with current regime
            if allowed_ids is not None and strategy.strategy_id not in allowed_ids:
                continue
            try:
                if strategy.warm(state):
                    out = strategy.generate_signals(state)
                    signals.extend(out)
            except Exception as e:
                logger.exception("Strategy %s failed: %s", strategy.strategy_id, e)
        # Sort by score desc; optional: apply ensemble weight from meta_optimizer
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

"""Run enabled strategies on market state; aggregate and rank signals."""
import logging
from typing import List

from src.core.events import Signal

from .base import MarketState, StrategyBase
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)


class StrategyRunner:
    """Consume market state; run all enabled strategies; return combined sorted signals."""

    def __init__(self, registry: StrategyRegistry):
        self.registry = registry

    def run(self, state: MarketState) -> List[Signal]:
        signals: List[Signal] = []
        for strategy in self.registry.get_enabled_strategies():
            try:
                if strategy.warm(state):
                    out = strategy.generate_signals(state)
                    signals.extend(out)
            except Exception as e:
                logger.exception("Strategy %s failed: %s", strategy.strategy_id, e)
        # Sort by score desc; optional: apply ensemble weight from meta_optimizer
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

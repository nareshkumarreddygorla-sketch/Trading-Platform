"""Strategy registry: plugin discovery and enable/disable."""
import logging
from typing import Dict, List, Optional, Type

from .base import StrategyBase

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Register and resolve strategies by id. Support enable/disable per strategy."""

    def __init__(self):
        self._strategies: Dict[str, StrategyBase] = {}
        self._enabled: Dict[str, bool] = {}

    def register(self, strategy: StrategyBase) -> None:
        self._strategies[strategy.strategy_id] = strategy
        self._enabled[strategy.strategy_id] = True

    def register_class(self, cls: Type[StrategyBase]) -> None:
        instance = cls()
        self.register(instance)

    def get(self, strategy_id: str) -> Optional[StrategyBase]:
        return self._strategies.get(strategy_id)

    def list_all(self) -> List[str]:
        return list(self._strategies.keys())

    def list_enabled(self) -> List[str]:
        return [sid for sid, on in self._enabled.items() if on]

    def enable(self, strategy_id: str) -> None:
        if strategy_id in self._strategies:
            self._enabled[strategy_id] = True

    def disable(self, strategy_id: str) -> None:
        if strategy_id in self._strategies:
            self._enabled[strategy_id] = False

    def get_enabled_strategies(self) -> List[StrategyBase]:
        return [self._strategies[sid] for sid in self.list_enabled()]

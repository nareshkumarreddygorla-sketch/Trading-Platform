"""Strategy registry: plugin discovery and enable/disable."""

import logging

from .base import StrategyBase

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Register and resolve strategies by id. Support enable/disable per strategy."""

    def __init__(self):
        self._strategies: dict[str, StrategyBase] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, strategy: StrategyBase) -> None:
        self._strategies[strategy.strategy_id] = strategy
        self._enabled[strategy.strategy_id] = True

    def register_class(self, cls: type[StrategyBase]) -> None:
        instance = cls()
        self.register(instance)

    def get(self, strategy_id: str) -> StrategyBase | None:
        return self._strategies.get(strategy_id)

    def list_all(self) -> list[str]:
        return list(self._strategies.keys())

    def list_enabled(self) -> list[str]:
        return [sid for sid, on in self._enabled.items() if on]

    def enable(self, strategy_id: str) -> None:
        if strategy_id in self._strategies:
            self._enabled[strategy_id] = True

    def disable(self, strategy_id: str) -> None:
        if strategy_id in self._strategies:
            self._enabled[strategy_id] = False

    def get_enabled_strategies(self) -> list[StrategyBase]:
        return [self._strategies[sid] for sid in self.list_enabled()]

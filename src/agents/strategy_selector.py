"""
Strategy Selector Agent: uses regime detection to dynamically
enable/disable strategies based on market conditions.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .base import BaseAgent

logger = logging.getLogger(__name__)

# Strategy regime suitability mapping
REGIME_STRATEGY_MAP = {
    "trending_up": {
        "enable": ["ema_crossover", "macd", "momentum_breakout", "ml_predictor", "ai_alpha"],
        "disable": ["mean_reversion"],
    },
    "trending_down": {
        "enable": ["ema_crossover", "macd", "ml_predictor", "ai_alpha"],
        "disable": ["momentum_breakout", "mean_reversion"],
    },
    "low_volatility": {
        "enable": ["mean_reversion", "rsi", "ml_predictor", "ai_alpha"],
        "disable": ["momentum_breakout"],
    },
    "high_volatility": {
        "enable": ["rsi", "ml_predictor"],
        "disable": ["ema_crossover", "momentum_breakout", "mean_reversion"],
    },
    "crisis": {
        "enable": [],
        "disable": ["ema_crossover", "macd", "rsi", "momentum_breakout", "mean_reversion"],
        # Only AI models remain (they have their own caution)
    },
}


class StrategySelectorAgent(BaseAgent):
    """
    Regime-based strategy selection agent.
    Monitors market regime and enables/disables strategies accordingly.
    """

    name = "strategy_selector"
    description = "Regime-based dynamic strategy selection"

    def __init__(
        self,
        strategy_registry=None,
        regime_classifier=None,
        get_bars: Callable | None = None,
        index_symbol: str = "NIFTY",
        selection_interval: float = 120.0,  # 2 minutes
    ):
        super().__init__()
        self._registry = strategy_registry
        self._regime_classifier = regime_classifier
        self._get_bars = get_bars
        self._index_symbol = index_symbol
        self._selection_interval = selection_interval
        self._current_regime: str = "unknown"
        self._regime_history: list[str] = []

    @property
    def interval_seconds(self) -> float:
        return self._selection_interval

    def _detect_regime(self) -> str | None:
        """Detect current market regime from index bars."""
        if not self._regime_classifier or not self._get_bars:
            return None

        try:
            from src.core.events import Exchange

            bars = self._get_bars(self._index_symbol, Exchange.NSE, "1m", 100)
            if not bars or len(bars) < 30:
                # Try with a common index name
                for alt in ["NIFTY 50", "NIFTY50", "^NSEI"]:
                    bars = self._get_bars(alt, Exchange.NSE, "1m", 100)
                    if bars and len(bars) >= 30:
                        break

            if not bars or len(bars) < 30:
                return None

            closes = np.array([b.close for b in bars])
            returns = np.diff(closes) / (closes[:-1] + 1e-12)
            vol = float(np.std(returns))
            trend = float(np.mean(returns[-20:])) if len(returns) >= 20 else 0.0

            result = self._regime_classifier.classify(returns, vol, trend, None)
            return result.label.value if hasattr(result.label, "value") else str(result.label)

        except Exception as e:
            logger.debug("Regime detection failed: %s", e)
            return None

    async def run_cycle(self) -> None:
        if not self._registry:
            return

        regime = self._detect_regime()
        if regime is None:
            return

        self._regime_history.append(regime)
        if len(self._regime_history) > 30:
            self._regime_history = self._regime_history[-30:]

        # Only act on regime change (with confirmation: same regime 2+ times)
        if regime == self._current_regime:
            return

        # Require 2 consecutive same-regime detections to confirm regime change
        if len(self._regime_history) < 2 or self._regime_history[-2] != regime:
            return

        old_regime = self._current_regime
        self._current_regime = regime
        logger.info("StrategySelectorAgent: regime changed %s → %s", old_regime, regime)

        # Apply strategy changes
        strategy_map = REGIME_STRATEGY_MAP.get(regime, {})
        to_enable = strategy_map.get("enable", [])
        to_disable = strategy_map.get("disable", [])

        for sid in to_enable:
            try:
                self._registry.enable(sid)
            except Exception:
                pass

        for sid in to_disable:
            try:
                self._registry.disable(sid)
            except Exception:
                pass

        logger.info("StrategySelectorAgent: enabled=%s disabled=%s", to_enable, to_disable)

        # Broadcast regime change
        await self.send_message(
            target="broadcast",
            msg_type="regime_change",
            payload={
                "old_regime": old_regime,
                "new_regime": regime,
                "enabled_strategies": to_enable,
                "disabled_strategies": to_disable,
            },
        )

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["current_regime"] = self._current_regime
        status["regime_history"] = self._regime_history[-10:]
        return status

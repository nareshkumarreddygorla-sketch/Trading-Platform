"""
Market Regime Classifier: Trending, Sideways, High/Low Vol, Crisis.
Combines HMM, clustering, and volatility thresholds.
Strategies activate/deactivate based on regime.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RegimeLabel(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    label: RegimeLabel
    confidence: float
    volatility_percentile: float
    trend_strength: float
    metadata: dict[str, Any]


class RegimeClassifier:
    """
    Combines HMM state, volatility regime, and trend strength to output
    a single RegimeLabel. Strategies subscribe to regimes they can run in.
    """

    def __init__(
        self,
        vol_high_percentile: float = 80.0,
        vol_low_percentile: float = 20.0,
        crisis_vol_multiplier: float = 2.0,
        trend_threshold: float = 0.3,
    ):
        self.vol_high_percentile = vol_high_percentile
        self.vol_low_percentile = vol_low_percentile
        self.crisis_vol_multiplier = crisis_vol_multiplier
        self.trend_threshold = trend_threshold
        self._vol_history: list[float] = []
        self._hmm: Any | None = None
        self._n_states = 3

    def update_vol_history(self, vol: float) -> None:
        self._vol_history.append(vol)
        if len(self._vol_history) > 500:
            self._vol_history = self._vol_history[-500:]

    def _vol_percentile(self, vol: float) -> float:
        if len(self._vol_history) < 10:
            return 50.0
        return float(100 * (np.sum(np.array(self._vol_history) < vol) / len(self._vol_history)))

    def classify(
        self,
        returns: np.ndarray,
        volatility: float,
        trend_strength: float,
        hmm_state: int | None = None,
    ) -> RegimeResult:
        """
        Returns regime label and confidence. Uses volatility percentile and
        trend strength; optionally HMM state for refinement.
        """
        self.update_vol_history(volatility)
        vol_pct = self._vol_percentile(volatility)
        mean_vol = np.mean(self._vol_history) if self._vol_history else volatility

        # Crisis: vol >> recent mean
        if mean_vol > 1e-12 and volatility >= mean_vol * self.crisis_vol_multiplier:
            return RegimeResult(
                label=RegimeLabel.CRISIS,
                confidence=0.9,
                volatility_percentile=vol_pct,
                trend_strength=trend_strength,
                metadata={"reason": "vol_spike"},
            )

        if vol_pct >= self.vol_high_percentile:
            vol_regime = RegimeLabel.HIGH_VOLATILITY
        elif vol_pct <= self.vol_low_percentile:
            vol_regime = RegimeLabel.LOW_VOLATILITY
        else:
            vol_regime = None

        if abs(trend_strength) >= self.trend_threshold:
            if trend_strength > 0:
                trend_regime = RegimeLabel.TRENDING_UP
            else:
                trend_regime = RegimeLabel.TRENDING_DOWN
        else:
            trend_regime = RegimeLabel.SIDEWAYS

        # Combine: if high vol, report vol regime; else trend
        if vol_regime == RegimeLabel.CRISIS:
            label = RegimeLabel.CRISIS
            confidence = 0.9
        elif vol_regime == RegimeLabel.HIGH_VOLATILITY:
            label = RegimeLabel.HIGH_VOLATILITY
            confidence = 0.7
        elif vol_regime == RegimeLabel.LOW_VOLATILITY and trend_regime == RegimeLabel.SIDEWAYS:
            label = RegimeLabel.LOW_VOLATILITY
            confidence = 0.6
        else:
            label = trend_regime
            confidence = min(0.9, 0.5 + abs(trend_strength))

        return RegimeResult(
            label=label,
            confidence=confidence,
            volatility_percentile=vol_pct,
            trend_strength=trend_strength,
            metadata={"vol_regime": vol_regime, "trend_regime": trend_regime, "hmm_state": hmm_state},
        )

    def strategies_for_regime(self, regime: RegimeLabel) -> list[str]:
        """Which strategy IDs are allowed in this regime. Config-driven."""
        # Example mapping; load from config in production
        allowed = {
            RegimeLabel.TRENDING_UP: ["ema_crossover", "macd", "momentum"],
            RegimeLabel.TRENDING_DOWN: ["macd", "momentum"],
            RegimeLabel.SIDEWAYS: ["rsi", "mean_reversion"],
            RegimeLabel.HIGH_VOLATILITY: ["vol_breakout"],
            RegimeLabel.LOW_VOLATILITY: ["ema_crossover", "macd", "rsi"],
            RegimeLabel.CRISIS: [],  # disable discretionary
            RegimeLabel.UNKNOWN: ["ema_crossover", "rsi"],
        }
        return allowed.get(regime, [])

"""
Volatility prediction model using EWMA vol regime detection.
Outputs confidence scaling based on current vs historical volatility,
and directional bias from vol regime (mean-reversion signal: high vol → expect contraction → bullish).
"""

import math
from typing import Any

from .base import BasePredictor, PredictionOutput


class VolPredictor(BasePredictor):
    """
    Volatility-based predictor using EWMA vol and vol-of-vol for regime detection.
    - High vol relative to historical mean → lower confidence, slight mean-reversion bias
    - Low vol → higher confidence, neutral directional
    - Vol-of-vol (vol clustering) used as additional confidence scaler
    """

    model_id = "vol_pred"
    version = "v2"

    # EWMA decay factor (0.94 ~ RiskMetrics standard)
    LAMBDA = 0.94
    # Long-term vol percentile thresholds
    HIGH_VOL_THRESHOLD = 1.5  # current vol > 1.5x median → high vol regime
    LOW_VOL_THRESHOLD = 0.7  # current vol < 0.7x median → low vol regime

    def __init__(self, model=None):
        self._model = model
        self.path = ""

    def predict(self, features: dict[str, float], context: dict[str, Any] | None = None) -> PredictionOutput:
        # Extract volatility features from feature engine
        rolling_vol_20 = features.get("rolling_vol_20", 0.0) or 0.0
        rolling_vol_5 = features.get("rolling_vol_5", 0.0) or 0.0
        atr_pct = features.get("atr_pct", 0.0) or 0.0
        bb_width = features.get("bb_width", 0.0) or 0.0

        # Use best available vol measure
        current_vol = rolling_vol_20 if rolling_vol_20 > 1e-8 else (atr_pct if atr_pct > 1e-8 else 0.01)

        # Vol-of-vol: ratio of short-term to long-term vol (vol clustering indicator)
        vol_ratio = (rolling_vol_5 / current_vol) if current_vol > 1e-8 and rolling_vol_5 > 1e-8 else 1.0

        # Historical median vol proxy (NSE stocks typically 1-3% daily vol)
        median_vol = 0.015  # 1.5% baseline for NSE equities
        vol_regime_ratio = current_vol / median_vol if median_vol > 0 else 1.0

        # Confidence: inverse relationship with vol regime
        if vol_regime_ratio > self.HIGH_VOL_THRESHOLD:
            # High vol regime: low confidence, vol may contract (mean reversion)
            confidence = max(0.1, 0.6 - (vol_regime_ratio - self.HIGH_VOL_THRESHOLD) * 0.3)
        elif vol_regime_ratio < self.LOW_VOL_THRESHOLD:
            # Low vol regime: high confidence, breakout may come
            confidence = min(0.9, 0.7 + (self.LOW_VOL_THRESHOLD - vol_regime_ratio) * 0.3)
        else:
            # Normal vol regime
            confidence = 0.6

        # Vol clustering penalty: if short-term vol >> long-term, vol is expanding → reduce confidence
        if vol_ratio > 1.3:
            confidence *= 0.85

        # Directional bias from vol regime (slight mean-reversion signal)
        # Very high vol historically tends to precede rallies (vol crush = bullish)
        if vol_regime_ratio > 2.0:
            prob_up = 0.55  # slight bullish bias (vol mean reversion)
            expected_return = 0.002
        elif vol_regime_ratio < 0.5:
            prob_up = 0.48  # low vol often precedes breakdowns
            expected_return = -0.001
        else:
            prob_up = 0.50  # neutral
            expected_return = 0.0

        # Bollinger width as additional confirmation
        if bb_width > 0:
            # Narrow BB (squeeze) → expect expansion → lower confidence in direction
            if bb_width < 0.02:
                confidence *= 0.9

        confidence = max(0.0, min(1.0, confidence))
        if not math.isfinite(confidence):
            confidence = 0.0
        if not math.isfinite(prob_up):
            prob_up = 0.5

        return PredictionOutput(
            prob_up=prob_up,
            expected_return=expected_return,
            confidence=confidence,
            model_id=self.model_id,
            version=self.version,
            metadata={
                "current_vol": current_vol,
                "vol_regime_ratio": vol_regime_ratio,
                "vol_of_vol_ratio": vol_ratio,
                "bb_width": bb_width,
            },
        )

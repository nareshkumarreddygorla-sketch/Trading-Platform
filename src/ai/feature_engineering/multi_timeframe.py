"""
Multi-Timeframe Feature Engine:
Compute features across 1m, 5m, 15m, 1h, daily bars simultaneously.
Higher timeframes provide trend context; lower timeframes provide entry timing.

Rule: Trade in the direction of the higher timeframe trend,
      timed by the lower timeframe signal.
"""

import logging
from typing import Any

import numpy as np

from src.ai.feature_engine import FeatureEngine
from src.core.events import Bar

logger = logging.getLogger(__name__)

# Timeframe hierarchy: higher = stronger trend signal
TIMEFRAME_HIERARCHY = ["1m", "5m", "15m", "1h", "1d"]
TIMEFRAME_WEIGHTS = {
    "1m": 0.05,
    "5m": 0.10,
    "15m": 0.20,
    "1h": 0.30,
    "1d": 0.35,
}


class MultiTimeframeEngine:
    """
    Builds feature vectors from multiple timeframes.
    Each timeframe's features are prefixed (e.g., tf_1h_rsi, tf_1d_ema_spread).
    Also computes cross-timeframe alignment features.
    """

    def __init__(self, timeframes: list[str] = None):
        self.timeframes = timeframes or ["5m", "15m", "1h", "1d"]
        self._engine = FeatureEngine()

    def build_multi_tf_features(
        self,
        bars_by_tf: dict[str, list[Bar]],
    ) -> dict[str, float]:
        """
        Build comprehensive multi-timeframe feature dict.

        Args:
            bars_by_tf: Dict of interval -> List[Bar]
                e.g., {"5m": [...], "15m": [...], "1h": [...], "1d": [...]}

        Returns:
            Dict of feature_name -> value with tf_ prefixes + alignment features.
        """
        features: dict[str, float] = {}

        # Per-timeframe features
        tf_features: dict[str, dict[str, float]] = {}
        for tf in self.timeframes:
            bars = bars_by_tf.get(tf, [])
            if len(bars) < 20:
                continue
            tf_feats = self._engine.build_features(bars)
            tf_features[tf] = tf_feats
            # Add with prefix
            for name, value in tf_feats.items():
                features[f"tf_{tf}_{name}"] = value

        # Cross-timeframe alignment features
        if len(tf_features) >= 2:
            alignment = self._compute_alignment(tf_features)
            features.update(alignment)

        # Multi-timeframe composite signal
        composite = self._compute_composite_signal(tf_features)
        features.update(composite)

        return features

    def _compute_alignment(
        self,
        tf_features: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Compute cross-timeframe alignment features.
        Alignment = how many timeframes agree on direction.
        """
        alignment: dict[str, float] = {}

        # Trend alignment: count TFs where EMA spread > 0 (bullish)
        bullish_count = 0
        total_tfs = 0
        for tf, feats in tf_features.items():
            ema_spread = feats.get("ema_spread", 0)
            if ema_spread != 0:
                total_tfs += 1
                if ema_spread > 0:
                    bullish_count += 1

        alignment["mtf_trend_alignment"] = bullish_count / total_tfs if total_tfs > 0 else 0.5

        # Momentum alignment: count TFs with positive momentum_5
        mom_bullish = 0
        mom_total = 0
        for tf, feats in tf_features.items():
            mom = feats.get("momentum_5", 0)
            if mom != 0:
                mom_total += 1
                if mom > 0:
                    mom_bullish += 1

        alignment["mtf_momentum_alignment"] = mom_bullish / mom_total if mom_total > 0 else 0.5

        # RSI divergence: higher TF overbought but lower TF oversold (or vice versa)
        rsis = {}
        for tf in TIMEFRAME_HIERARCHY:
            if tf in tf_features:
                rsis[tf] = tf_features[tf].get("rsi", 50)
        if len(rsis) >= 2:
            rsi_values = list(rsis.values())
            alignment["mtf_rsi_spread"] = max(rsi_values) - min(rsi_values)
            alignment["mtf_rsi_divergence"] = float(
                (rsi_values[-1] - rsi_values[0]) / 100  # higher TF - lower TF
            )
        else:
            alignment["mtf_rsi_spread"] = 0.0
            alignment["mtf_rsi_divergence"] = 0.0

        # Volatility regime agreement
        vol_levels = []
        for tf, feats in tf_features.items():
            vol = feats.get("rolling_volatility", 0)
            vol_levels.append(vol)
        if vol_levels:
            alignment["mtf_vol_mean"] = float(np.mean(vol_levels))
            alignment["mtf_vol_dispersion"] = float(np.std(vol_levels))
        else:
            alignment["mtf_vol_mean"] = 0.0
            alignment["mtf_vol_dispersion"] = 0.0

        # MACD alignment
        macd_bullish = 0
        macd_total = 0
        for tf, feats in tf_features.items():
            macd_hist = feats.get("macd_histogram", 0)
            if macd_hist != 0:
                macd_total += 1
                if macd_hist > 0:
                    macd_bullish += 1
        alignment["mtf_macd_alignment"] = macd_bullish / macd_total if macd_total > 0 else 0.5

        return alignment

    def _compute_composite_signal(
        self,
        tf_features: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Compute a weighted composite signal across timeframes.
        Higher TFs get more weight for trend, lower TFs for timing.
        """
        composite: dict[str, float] = {}

        # Weighted trend score
        trend_score = 0.0
        weight_sum = 0.0
        for tf, feats in tf_features.items():
            w = TIMEFRAME_WEIGHTS.get(tf, 0.1)
            ema_spread = feats.get("ema_spread", 0)
            trend_score += ema_spread * w
            weight_sum += w

        composite["mtf_weighted_trend"] = trend_score / (weight_sum + 1e-12)

        # Weighted momentum score
        mom_score = 0.0
        mom_weight_sum = 0.0
        for tf, feats in tf_features.items():
            w = TIMEFRAME_WEIGHTS.get(tf, 0.1)
            mom = feats.get("momentum_10", 0)
            mom_score += mom * w
            mom_weight_sum += w

        composite["mtf_weighted_momentum"] = mom_score / (mom_weight_sum + 1e-12)

        # Composite confidence: how aligned are all TFs?
        signals = []
        for tf, feats in tf_features.items():
            ema = feats.get("ema_spread", 0)
            rsi = feats.get("rsi", 50)
            macd = feats.get("macd_histogram", 0)
            # Simple direction: +1 bullish, -1 bearish
            direction = 0
            if ema > 0:
                direction += 1
            elif ema < 0:
                direction -= 1
            if rsi > 55:
                direction += 1
            elif rsi < 45:
                direction -= 1
            if macd > 0:
                direction += 1
            elif macd < 0:
                direction -= 1
            signals.append(direction)

        if signals:
            # Agreement ratio: all same sign = high confidence
            avg_signal = np.mean(signals)
            composite["mtf_signal_strength"] = float(avg_signal / 3.0)  # normalize to [-1, 1]
            composite["mtf_signal_agreement"] = float(np.mean([1 if s * avg_signal > 0 else 0 for s in signals]))
        else:
            composite["mtf_signal_strength"] = 0.0
            composite["mtf_signal_agreement"] = 0.0

        return composite

    def get_entry_timing(
        self,
        bars_by_tf: dict[str, list[Bar]],
    ) -> dict[str, Any]:
        """
        Determine optimal entry timing based on multi-TF analysis.

        Returns:
            Dict with: direction (1/-1/0), confidence (0-1), entry_tf, reason
        """
        features = self.build_multi_tf_features(bars_by_tf)

        trend = features.get("mtf_weighted_trend", 0)
        momentum = features.get("mtf_weighted_momentum", 0)
        alignment = features.get("mtf_signal_agreement", 0)
        strength = features.get("mtf_signal_strength", 0)

        # Direction from higher timeframes
        direction = 1 if trend > 0 else (-1 if trend < 0 else 0)

        # Confidence from alignment
        confidence = min(1.0, abs(strength) * alignment)

        # Entry TF: use the lowest available for precise timing
        entry_tf = self.timeframes[0] if self.timeframes else "15m"

        reason = []
        if abs(trend) > 0.001:
            reason.append(f"trend={'bullish' if trend > 0 else 'bearish'}")
        if alignment > 0.7:
            reason.append(f"alignment={alignment:.0%}")
        if abs(momentum) > 0.005:
            reason.append(f"momentum={'positive' if momentum > 0 else 'negative'}")

        return {
            "direction": direction,
            "confidence": confidence,
            "entry_tf": entry_tf,
            "reason": " | ".join(reason) if reason else "neutral",
            "features": features,
        }

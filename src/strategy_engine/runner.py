"""Run enabled strategies on market state; aggregate and rank signals.
Regime-aware: filters strategies by market regime when regime data is available.
Multi-timeframe aware: optionally enriches strategy context with cross-TF alignment signals.
"""

import logging
from collections.abc import Callable
from typing import Any

from src.core.events import Bar, Exchange, Signal

from .base import MarketState
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)

# Regime → strategy mapping: which strategies are allowed in each regime
REGIME_STRATEGY_MAP: dict[str, set[str]] = {
    "trending_up": {"ai_alpha", "ema_crossover", "macd", "momentum_breakout", "ml_predictor", "rl_agent"},
    "trending_down": {"ai_alpha", "ema_crossover", "macd", "ml_predictor", "rl_agent"},
    "sideways": {"ai_alpha", "rsi", "mean_reversion", "ml_predictor"},
    "high_volatility": {"ai_alpha", "ml_predictor", "rl_agent"},
    "low_volatility": {"ai_alpha", "ema_crossover", "macd", "rsi", "mean_reversion", "ml_predictor"},
    "crisis": set(),  # No strategies in crisis mode
}

# Timeframes used for multi-TF analysis (ascending granularity)
_MTF_INTERVALS: list[tuple[str, int]] = [
    ("1m", 100),
    ("5m", 100),
    ("15m", 60),
    ("1h", 50),
    ("1d", 30),
]


class StrategyRunner:
    """Consume market state; run all enabled strategies; return combined sorted signals.
    When regime data is present in state.metadata, filters strategies by regime compatibility.
    Optionally enriches state.metadata with multi-timeframe alignment via ``run_with_multi_timeframe``.
    """

    def __init__(self, registry: StrategyRegistry):
        self.registry = registry
        self._mtf_engine = None  # lazy-init to avoid import cost when not used

    # ── public API (unchanged) ──────────────────────────────────────────

    def run(self, state: MarketState) -> list[Signal]:
        regime = (state.metadata or {}).get("regime", None) if state.metadata else None
        allowed_ids: set[str] | None = None
        if regime and regime in REGIME_STRATEGY_MAP:
            allowed_ids = REGIME_STRATEGY_MAP[regime]
            if not allowed_ids:
                logger.warning("Regime '%s' — all strategies blocked (crisis mode)", regime)
                return []

        signals: list[Signal] = []
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

    # ── multi-timeframe entry point ─────────────────────────────────────

    def run_with_multi_timeframe(
        self,
        state: MarketState,
        get_bars: Callable[[str, Exchange, str, int], list[Bar]],
    ) -> list[Signal]:
        """Run strategies with multi-timeframe context injected into ``state.metadata``.

        1. Fetches bars across 1m / 5m / 15m / 1h / 1d via *get_bars*.
        2. Builds MTF features & entry-timing signals via ``MultiTimeframeEngine``.
        3. Merges alignment/timing data into ``state.metadata["mtf"]``.
        4. Falls back to the plain ``run()`` path when fewer than 2 timeframes
           have enough data (>= 20 bars each).

        Parameters
        ----------
        state : MarketState
            The per-symbol market state (bars, price, etc.).
        get_bars : callable
            ``(symbol, exchange, interval, count) -> List[Bar]`` — typically
            the bar-cache accessor wired through ``AutonomousLoop``.

        Returns
        -------
        List[Signal]
            Ranked signals, identical contract to ``run()``.
        """
        mtf_context = self._build_mtf_context(state.symbol, state.exchange, get_bars)
        if mtf_context is not None:
            if state.metadata is None:
                state.metadata = {}
            state.metadata["mtf"] = mtf_context
            logger.debug(
                "MTF context injected for %s: direction=%s confidence=%.2f",
                state.symbol,
                mtf_context.get("direction"),
                mtf_context.get("confidence", 0),
            )
        else:
            logger.debug("MTF: insufficient data for %s, falling back to single-TF", state.symbol)

        return self.run(state)

    # ── internals ───────────────────────────────────────────────────────

    def _get_mtf_engine(self):
        """Lazy-initialise the ``MultiTimeframeEngine`` on first use."""
        if self._mtf_engine is None:
            try:
                from src.ai.feature_engineering.multi_timeframe import MultiTimeframeEngine

                self._mtf_engine = MultiTimeframeEngine()
            except Exception:
                logger.exception("Failed to import MultiTimeframeEngine; MTF features disabled")
        return self._mtf_engine

    def _build_mtf_context(
        self,
        symbol: str,
        exchange: Exchange,
        get_bars: Callable[[str, Exchange, str, int], list[Bar]],
    ) -> dict[str, Any] | None:
        """Fetch multi-TF bars and compute alignment context dict.

        Returns ``None`` when insufficient data is available so the caller
        can gracefully fall back to single-timeframe execution.
        """
        engine = self._get_mtf_engine()
        if engine is None:
            return None

        bars_by_tf: dict[str, list[Bar]] = {}
        for interval, count in _MTF_INTERVALS:
            try:
                bars = get_bars(symbol, exchange, interval, count)
                if bars:
                    bars_by_tf[interval] = bars
            except Exception:
                logger.debug("MTF: could not fetch %s bars for %s", interval, symbol)

        # Need at least 2 timeframes with >= 20 bars each for meaningful alignment
        usable_tfs = [tf for tf, bars in bars_by_tf.items() if len(bars) >= 20]
        if len(usable_tfs) < 2:
            return None

        try:
            timing = engine.get_entry_timing(bars_by_tf)
        except Exception:
            logger.exception("MTF feature computation failed for %s", symbol)
            return None

        return {
            "direction": timing["direction"],
            "confidence": timing["confidence"],
            "entry_tf": timing["entry_tf"],
            "reason": timing["reason"],
            "trend_alignment": timing["features"].get("mtf_trend_alignment", 0.5),
            "momentum_alignment": timing["features"].get("mtf_momentum_alignment", 0.5),
            "signal_strength": timing["features"].get("mtf_signal_strength", 0.0),
            "signal_agreement": timing["features"].get("mtf_signal_agreement", 0.0),
            "weighted_trend": timing["features"].get("mtf_weighted_trend", 0.0),
            "features": timing["features"],
        }

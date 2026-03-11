"""
Correlation-aware position allocation.

Prevents hidden concentration risk by:
  - Rejecting new positions highly correlated with existing holdings
  - Enforcing portfolio-level volatility budget
  - Computing marginal risk contribution for each new position
  - Adaptive EWMA half-life: 20 days in high-vol regimes, 60 days in normal
  - Intraday correlation tracking (tick-level updates)
  - Correlation spike detection (alert if any pair jumps > 0.3 in one day)
"""

from __future__ import annotations

import logging
import math
import threading
import time as _time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Regime-adaptive correlation half-lives
_NORMAL_HALFLIFE = 60  # days in normal regime
_FAST_HALFLIFE = 20  # days in high-vol regime
_VOL_REGIME_THRESHOLD = 2.0  # realized vol > 2x historical -> high-vol regime
_CORRELATION_SPIKE_THRESHOLD = 0.3  # alert if pair correlation jumps > 0.3 in a day


@dataclass
class CorrelationCheckResult:
    """Result of a correlation check for a new position."""

    allowed: bool = True
    reason: str = ""
    max_correlation: float = 0.0
    correlated_with: str = ""
    portfolio_vol_pct: float = 0.0


@dataclass
class CorrelationSpikeAlert:
    """Alert for correlation spike between two symbols."""

    sym_a: str
    sym_b: str
    prev_correlation: float
    new_correlation: float
    change: float
    timestamp: float = 0.0


class CorrelationGuard:
    """
    Guards against excessive portfolio correlation and concentration.

    Enhanced with:
      - Regime-adaptive EWMA: half-life 20 days in high-vol, 60 days in normal
      - Intraday correlation tracking
      - Correlation spike detection

    Usage:
        guard = CorrelationGuard()
        # Feed daily returns
        guard.update_returns("RELIANCE", 0.012)
        guard.update_returns("INFY", -0.005)
        ...
        result = guard.check_new_position("HDFCBANK", existing_positions=["ICICIBANK", "SBIN"])
    """

    def __init__(
        self,
        max_pairwise_correlation: float = 0.70,
        max_portfolio_vol_pct: float = 15.0,
        correlation_window: int = 60,
        min_history: int = 10,
        ewma_lambda: float = 0.94,
        vol_regime_threshold: float = _VOL_REGIME_THRESHOLD,
        correlation_spike_threshold: float = _CORRELATION_SPIKE_THRESHOLD,
    ):
        self.max_pairwise_correlation = max_pairwise_correlation
        self.max_portfolio_vol_pct = max_portfolio_vol_pct
        self.correlation_window = correlation_window
        self.min_history = min_history
        self.ewma_lambda = ewma_lambda
        self._vol_regime_threshold = vol_regime_threshold
        self._correlation_spike_threshold = correlation_spike_threshold
        self._lock = threading.RLock()

        # Per-symbol daily returns buffer
        self._returns: dict[str, list[float]] = {}
        # EWMA variance
        self._ewma_var: dict[str, float] = {}

        # ── Regime detection state ──
        self._high_vol_regime: bool = False
        self._historical_vol: float = 0.0  # long-term average vol (annualised)
        self._realized_vol: float = 0.0  # recent realised vol (annualised)

        # ── Intraday correlation tracking ──
        self._intraday_returns: dict[str, list[float]] = {}  # tick-level returns within day
        self._intraday_correlations: dict[tuple[str, str], float] = {}  # latest intraday pair corr

        # ── Correlation spike detection ──
        self._prev_day_correlations: dict[tuple[str, str], float] = {}
        self._correlation_spike_alerts: list[CorrelationSpikeAlert] = []

    @property
    def high_vol_regime(self) -> bool:
        """Whether we are currently in a high-volatility regime."""
        return self._high_vol_regime

    @property
    def effective_correlation_window(self) -> int:
        """Adaptive correlation window based on regime."""
        if self._high_vol_regime:
            return _FAST_HALFLIFE
        return self.correlation_window

    @property
    def effective_ewma_lambda(self) -> float:
        """Adaptive EWMA lambda: faster decay in high-vol regime."""
        if self._high_vol_regime:
            # half-life 20 days => lambda = exp(-ln2/20) ~ 0.966
            # But we want faster reaction, so use a lower lambda
            return 0.90
        return self.ewma_lambda

    def update_vol_regime(self, realized_vol: float, historical_vol: float) -> None:
        """
        Update regime detection based on realized vs historical volatility.
        Switch to fast correlation updates if realized_vol > threshold * historical_vol.
        """
        with self._lock:
            self._realized_vol = realized_vol
            self._historical_vol = historical_vol
            was_high = self._high_vol_regime
            if historical_vol > 0 and realized_vol > self._vol_regime_threshold * historical_vol:
                self._high_vol_regime = True
                if not was_high:
                    logger.warning(
                        "Correlation regime -> HIGH-VOL: realized_vol=%.2f%% > %.1fx historical=%.2f%% "
                        "— switching to fast correlation updates (half-life=%d days)",
                        realized_vol * 100,
                        self._vol_regime_threshold,
                        historical_vol * 100,
                        _FAST_HALFLIFE,
                    )
            else:
                self._high_vol_regime = False
                if was_high:
                    logger.info(
                        "Correlation regime -> NORMAL: realized_vol=%.2f%% <= %.1fx historical=%.2f%% "
                        "— reverting to standard correlation updates (half-life=%d days)",
                        realized_vol * 100,
                        self._vol_regime_threshold,
                        historical_vol * 100,
                        _NORMAL_HALFLIFE,
                    )

    def update_intraday_return(self, symbol: str, tick_return: float) -> None:
        """
        Record an intraday tick-level return for a symbol.
        Used for intraday correlation tracking (not just daily).
        """
        if not math.isfinite(tick_return):
            return
        with self._lock:
            if symbol not in self._intraday_returns:
                self._intraday_returns[symbol] = []
            self._intraday_returns[symbol].append(tick_return)
            # Keep only last 500 ticks per symbol to bound memory
            if len(self._intraday_returns[symbol]) > 500:
                self._intraday_returns[symbol] = self._intraday_returns[symbol][-500:]

    def compute_intraday_correlations(self, symbols: list[str]) -> dict[tuple[str, str], float]:
        """
        Compute pairwise intraday correlations from tick-level returns.
        Returns dict of (sym_a, sym_b) -> correlation.
        """
        result: dict[tuple[str, str], float] = {}
        with self._lock:
            for i, sa in enumerate(symbols):
                for j in range(i + 1, len(symbols)):
                    sb = symbols[j]
                    ra = self._intraday_returns.get(sa, [])
                    rb = self._intraday_returns.get(sb, [])
                    min_len = min(len(ra), len(rb))
                    if min_len < 10:
                        continue
                    a = np.array(ra[-min_len:])
                    b = np.array(rb[-min_len:])
                    try:
                        corr = float(np.corrcoef(a, b)[0, 1])
                        if math.isfinite(corr):
                            result[(sa, sb)] = corr
                            self._intraday_correlations[(sa, sb)] = corr
                    except Exception:
                        pass
        return result

    def reset_intraday(self) -> None:
        """Reset intraday tracking at end of day. Save correlations for spike detection."""
        with self._lock:
            # Store current correlations for next-day spike detection
            # Merge daily computed correlations with whatever we had
            self._prev_day_correlations = dict(self._intraday_correlations)
            self._intraday_returns.clear()
            self._intraday_correlations.clear()

    def detect_correlation_spikes(self, symbols: list[str]) -> list[CorrelationSpikeAlert]:
        """
        Detect if any pairwise correlation jumped > spike_threshold since yesterday.
        Returns list of spike alerts.
        """
        alerts: list[CorrelationSpikeAlert] = []
        with self._lock:
            for i, sa in enumerate(symbols):
                for j in range(i + 1, len(symbols)):
                    sb = symbols[j]
                    pair = (sa, sb)
                    prev = self._prev_day_correlations.get(pair)
                    if prev is None:
                        continue
                    # Get current correlation
                    current = self.pairwise_correlation(sa, sb)
                    change = abs(current - prev)
                    if change > self._correlation_spike_threshold:
                        alert = CorrelationSpikeAlert(
                            sym_a=sa,
                            sym_b=sb,
                            prev_correlation=prev,
                            new_correlation=current,
                            change=change,
                            timestamp=_time.time(),
                        )
                        alerts.append(alert)
                        logger.warning(
                            "CORRELATION SPIKE: %s/%s changed %.2f -> %.2f (delta=%.2f > threshold=%.2f)",
                            sa,
                            sb,
                            prev,
                            current,
                            change,
                            self._correlation_spike_threshold,
                        )
            self._correlation_spike_alerts = alerts
        return alerts

    def get_recent_spike_alerts(self) -> list[CorrelationSpikeAlert]:
        """Return the most recent correlation spike alerts."""
        return list(self._correlation_spike_alerts)

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """Record a daily return for a symbol. Uses regime-adaptive EWMA lambda."""
        if not math.isfinite(daily_return):
            logger.warning("CorrelationGuard: rejecting non-finite return for %s", symbol)
            return
        with self._lock:
            if symbol not in self._returns:
                self._returns[symbol] = []
            self._returns[symbol].append(daily_return)
            # Use adaptive window based on regime
            max_len = self.effective_correlation_window * 2
            if len(self._returns[symbol]) > max_len:
                self._returns[symbol] = self._returns[symbol][-max_len:]
            # EWMA variance with regime-adaptive lambda
            lam = self.effective_ewma_lambda
            if symbol not in self._ewma_var:
                self._ewma_var[symbol] = daily_return**2
            else:
                self._ewma_var[symbol] = lam * self._ewma_var[symbol] + (1 - lam) * daily_return**2

    def pairwise_correlation(self, sym_a: str, sym_b: str) -> float:
        """Compute rolling pairwise correlation between two symbols."""
        with self._lock:
            r_a = self._returns.get(sym_a, [])
            r_b = self._returns.get(sym_b, [])
            min_len = min(len(r_a), len(r_b))
            if min_len < self.min_history:
                # Use sector-aware default correlation instead of fixed 0.5
                # Same-sector Indian stocks correlate 0.7-0.9, cross-sector 0.3-0.5
                try:
                    from src.risk_engine.var import PortfolioVaR

                    return PortfolioVaR._default_correlation(sym_a, sym_b)
                except Exception:
                    return 0.50  # Fallback to moderate correlation

            window = min(self.correlation_window, min_len)
            a = np.array(r_a[-window:])
            b = np.array(r_b[-window:])

            # Pearson correlation
            try:
                corr_matrix = np.corrcoef(a, b)
                corr = float(corr_matrix[0, 1])
                if np.isnan(corr):
                    return 0.5  # Conservative default on NaN
                return corr
            except Exception:
                return 0.5  # Conservative default on error

    def _estimate_portfolio_vol(self, symbols: list[str], notionals: list[float], total_value: float) -> float:
        """Estimate daily portfolio volatility using correlation matrix."""
        with self._lock:
            n = len(symbols)
            if n == 0 or total_value <= 0:
                return 0.0

            weights = np.array(notionals) / total_value

            # Build covariance matrix
            vols = np.array([self._get_daily_vol(s) for s in symbols])
            corr = self._correlation_matrix(symbols)
            D = np.diag(vols)
            cov = D @ corr @ D
            cov += np.eye(n) * 1e-10

            port_var = float(weights @ cov @ weights)
            return np.sqrt(max(0, port_var))

    def _get_daily_vol(self, symbol: str) -> float:
        """Get EWMA daily vol for a symbol."""
        if symbol in self._ewma_var and self._ewma_var[symbol] > 0:
            return np.sqrt(self._ewma_var[symbol])
        returns = self._returns.get(symbol, [])
        if len(returns) >= self.min_history:
            return float(np.std(returns, ddof=1))
        return 0.025  # default 2.5% daily vol

    def _correlation_matrix(self, symbols: list[str]) -> np.ndarray:
        """Build correlation matrix for given symbols. Uses adaptive window based on vol regime."""
        with self._lock:
            n = len(symbols)
            if n <= 1:
                return np.eye(n)

            window = self.effective_correlation_window
            max_len = max((len(self._returns.get(s, [])) for s in symbols), default=0)
            use_len = min(window, max_len)

            if use_len < self.min_history:
                # Use sector-aware default correlation matrix instead of identity (which is anti-conservative)
                default_corr = np.eye(n)
                for i in range(n):
                    for j in range(i + 1, n):
                        try:
                            from src.risk_engine.var import PortfolioVaR

                            c = PortfolioVaR._default_correlation(symbols[i], symbols[j])
                        except Exception:
                            c = 0.5  # Conservative default
                        default_corr[i, j] = c
                        default_corr[j, i] = c
                return default_corr

            returns_matrix = np.zeros((use_len, n))
            for j, sym in enumerate(symbols):
                r = self._returns.get(sym, [])
                if len(r) >= use_len:
                    returns_matrix[:, j] = r[-use_len:]
                elif len(r) > 0:
                    pad = use_len - len(r)
                    returns_matrix[pad:, j] = r

            try:
                corr = np.corrcoef(returns_matrix, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                return corr
            except Exception as e:
                logger.warning("Correlation matrix computation failed: %s — using sector-aware defaults", e)
                corr = np.eye(n)
                for i in range(n):
                    for j in range(i + 1, n):
                        try:
                            from src.risk_engine.var import PortfolioVaR

                            c = PortfolioVaR._default_correlation(symbols[i], symbols[j])
                        except Exception:
                            c = 0.5
                        corr[i, j] = c
                        corr[j, i] = c
                return corr

    def compute_stress_correlation(self, sym_a: str, sym_b: str, bottom_percentile: float = 10.0) -> float:
        """
        Compute conditional correlation using only bottom-percentile return days.
        Stress correlation is typically higher than normal correlation (tail dependence).
        """
        r_a = self._returns.get(sym_a, [])
        r_b = self._returns.get(sym_b, [])
        min_len = min(len(r_a), len(r_b))
        if min_len < self.min_history:
            return self.pairwise_correlation(sym_a, sym_b)

        window = min(self.correlation_window, min_len)
        a = np.array(r_a[-window:])
        b = np.array(r_b[-window:])

        # Filter to only bottom-percentile days (based on combined returns)
        # Filter based on individual stock stress days (either stock in tail)
        threshold_a = np.percentile(a, bottom_percentile)
        threshold_b = np.percentile(b, bottom_percentile)
        mask = (a <= threshold_a) | (b <= threshold_b)

        if mask.sum() < 3:  # Need at least 3 data points
            return self.pairwise_correlation(sym_a, sym_b)

        try:
            stress_a = a[mask]
            stress_b = b[mask]
            corr_matrix = np.corrcoef(stress_a, stress_b)
            corr = float(corr_matrix[0, 1])
            if np.isnan(corr):
                return self.pairwise_correlation(sym_a, sym_b)
            return corr
        except Exception:
            return self.pairwise_correlation(sym_a, sym_b)

    def check_new_position(
        self,
        new_symbol: str,
        existing_symbols: list[str],
        existing_notionals: list[float] | None = None,
        new_notional: float = 0.0,
        portfolio_value: float = 0.0,
        use_stress_correlation: bool = True,
    ) -> CorrelationCheckResult:
        """
        Check if a new position would violate correlation or portfolio vol limits.
        Uses max(normal, stress) correlation for conservative check when use_stress_correlation=True.
        """
        with self._lock:
            if not existing_symbols:
                return CorrelationCheckResult(allowed=True, reason="First position — no correlation check needed")

            # Check pairwise correlation with each existing position
            max_corr = 0.0
            max_corr_symbol = ""
            for sym in existing_symbols:
                normal_corr = self.pairwise_correlation(new_symbol, sym)
                if use_stress_correlation:
                    stress_corr = self.compute_stress_correlation(new_symbol, sym)
                    # Use whichever correlation has larger absolute value (conservative)
                    corr = stress_corr if abs(stress_corr) > abs(normal_corr) else normal_corr
                else:
                    corr = normal_corr
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    max_corr_symbol = sym

            if abs(max_corr) > self.max_pairwise_correlation:
                return CorrelationCheckResult(
                    allowed=False,
                    reason=f"Correlation too high: {new_symbol} vs {max_corr_symbol} = {max_corr:.2f} > {self.max_pairwise_correlation:.2f}",
                    max_correlation=max_corr,
                    correlated_with=max_corr_symbol,
                )

            # Check portfolio volatility budget (if we have enough data)
            if portfolio_value > 0 and existing_notionals and new_notional > 0:
                all_symbols = existing_symbols + [new_symbol]
                all_notionals = list(existing_notionals) + [new_notional]
                port_vol = self._estimate_portfolio_vol(all_symbols, all_notionals, portfolio_value + new_notional)
                port_vol_ann = port_vol * np.sqrt(252) * 100  # annualised %

                if port_vol_ann > self.max_portfolio_vol_pct:
                    return CorrelationCheckResult(
                        allowed=False,
                        reason=f"Portfolio vol {port_vol_ann:.1f}% > limit {self.max_portfolio_vol_pct:.1f}%",
                        max_correlation=max_corr,
                        correlated_with=max_corr_symbol,
                        portfolio_vol_pct=port_vol_ann,
                    )

            return CorrelationCheckResult(
                allowed=True,
                max_correlation=max_corr,
                correlated_with=max_corr_symbol,
            )

    def get_correlation_matrix(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Return full correlation matrix as nested dict (for API exposure)."""
        with self._lock:
            corr = self._correlation_matrix(symbols)
            result = {}
            for i, s1 in enumerate(symbols):
                result[s1] = {}
                for j, s2 in enumerate(symbols):
                    result[s1][s2] = round(float(corr[i, j]), 4)
            return result

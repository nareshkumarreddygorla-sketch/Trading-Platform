"""
Correlation-aware position allocation.

Prevents hidden concentration risk by:
  - Rejecting new positions highly correlated with existing holdings
  - Enforcing portfolio-level volatility budget
  - Computing marginal risk contribution for each new position
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationCheckResult:
    """Result of a correlation check for a new position."""
    allowed: bool = True
    reason: str = ""
    max_correlation: float = 0.0
    correlated_with: str = ""
    portfolio_vol_pct: float = 0.0


class CorrelationGuard:
    """
    Guards against excessive portfolio correlation and concentration.

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
    ):
        self.max_pairwise_correlation = max_pairwise_correlation
        self.max_portfolio_vol_pct = max_portfolio_vol_pct
        self.correlation_window = correlation_window
        self.min_history = min_history
        self.ewma_lambda = ewma_lambda

        # Per-symbol daily returns buffer
        self._returns: Dict[str, List[float]] = {}
        # EWMA variance
        self._ewma_var: Dict[str, float] = {}

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """Record a daily return for a symbol."""
        import math
        if not math.isfinite(daily_return):
            return  # Reject NaN/inf silently
        if symbol not in self._returns:
            self._returns[symbol] = []
        self._returns[symbol].append(daily_return)
        max_len = self.correlation_window * 2
        if len(self._returns[symbol]) > max_len:
            self._returns[symbol] = self._returns[symbol][-max_len:]
        # EWMA variance
        lam = self.ewma_lambda
        if symbol not in self._ewma_var:
            self._ewma_var[symbol] = daily_return ** 2
        else:
            self._ewma_var[symbol] = lam * self._ewma_var[symbol] + (1 - lam) * daily_return ** 2

    def pairwise_correlation(self, sym_a: str, sym_b: str) -> float:
        """Compute rolling pairwise correlation between two symbols."""
        r_a = self._returns.get(sym_a, [])
        r_b = self._returns.get(sym_b, [])
        min_len = min(len(r_a), len(r_b))
        if min_len < self.min_history:
            return 0.5  # Unknown = conservative moderate correlation (not 0.0 which is unsafe)

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

    def _estimate_portfolio_vol(self, symbols: List[str], notionals: List[float], total_value: float) -> float:
        """Estimate daily portfolio volatility using correlation matrix."""
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
            return float(np.std(returns))
        return 0.025  # default 2.5% daily vol

    def _correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Build correlation matrix for given symbols."""
        n = len(symbols)
        if n <= 1:
            return np.eye(n)

        window = self.correlation_window
        max_len = max((len(self._returns.get(s, [])) for s in symbols), default=0)
        use_len = min(window, max_len)

        if use_len < self.min_history:
            return np.eye(n)

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
        except Exception:
            return np.eye(n)

    def compute_stress_correlation(self, sym_a: str, sym_b: str, bottom_percentile: float = 10.0) -> float:
        """
        Compute conditional correlation using only bottom-percentile return days.
        Stress correlation is typically higher than normal correlation (tail dependence).
        """
        r_a = self._returns.get(sym_a, [])
        r_b = self._returns.get(sym_b, [])
        min_len = min(len(r_a), len(r_b))
        if min_len < self.min_history:
            return 0.0

        window = min(self.correlation_window, min_len)
        a = np.array(r_a[-window:])
        b = np.array(r_b[-window:])

        # Filter to only bottom-percentile days (based on combined returns)
        combined = a + b
        threshold = np.percentile(combined, bottom_percentile)
        mask = combined <= threshold

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
        existing_symbols: List[str],
        existing_notionals: Optional[List[float]] = None,
        new_notional: float = 0.0,
        portfolio_value: float = 0.0,
        use_stress_correlation: bool = True,
    ) -> CorrelationCheckResult:
        """
        Check if a new position would violate correlation or portfolio vol limits.
        Uses max(normal, stress) correlation for conservative check when use_stress_correlation=True.
        """
        if not existing_symbols:
            return CorrelationCheckResult(allowed=True, reason="First position — no correlation check needed")

        # Check pairwise correlation with each existing position
        max_corr = 0.0
        max_corr_symbol = ""
        for sym in existing_symbols:
            normal_corr = self.pairwise_correlation(new_symbol, sym)
            if use_stress_correlation:
                stress_corr = self.compute_stress_correlation(new_symbol, sym)
                corr = max(abs(normal_corr), abs(stress_corr))
                # Preserve sign from whichever was larger
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

    def get_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Return full correlation matrix as nested dict (for API exposure)."""
        corr = self._correlation_matrix(symbols)
        result = {}
        for i, s1 in enumerate(symbols):
            result[s1] = {}
            for j, s2 in enumerate(symbols):
                result[s1][s2] = round(float(corr[i, j]), 4)
        return result

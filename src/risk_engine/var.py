"""
Portfolio Value-at-Risk (VaR) calculator.

Implements:
  - Parametric VaR (variance-covariance method) at 95% and 99% confidence
  - Per-stock EWMA volatility estimation
  - Rolling pairwise correlation matrix
  - Portfolio volatility via w' * Sigma * w
  - Marginal VaR contribution per position
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Z-scores for confidence levels
Z_95 = 1.6449
Z_99 = 2.3263


@dataclass
class VaRResult:
    """Portfolio VaR output."""
    var_95: float = 0.0         # 95% VaR in currency (INR)
    var_99: float = 0.0         # 99% VaR in currency (INR)
    var_95_pct: float = 0.0     # 95% VaR as % of portfolio
    var_99_pct: float = 0.0     # 99% VaR as % of portfolio
    portfolio_vol: float = 0.0  # annualised portfolio volatility
    portfolio_vol_daily: float = 0.0  # daily portfolio volatility
    horizon_days: int = 1
    n_positions: int = 0
    per_position_var: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "var_95_pct": round(self.var_95_pct, 4),
            "var_99_pct": round(self.var_99_pct, 4),
            "portfolio_vol": round(self.portfolio_vol, 4),
            "portfolio_vol_daily": round(self.portfolio_vol_daily, 6),
            "horizon_days": self.horizon_days,
            "n_positions": self.n_positions,
            "per_position_var": {k: round(v, 2) for k, v in self.per_position_var.items()},
        }


class PortfolioVaR:
    """
    Computes parametric (variance-covariance) VaR for a portfolio of positions.

    Usage:
        var_calc = PortfolioVaR()
        # Feed daily returns as they arrive
        var_calc.update_returns("RELIANCE", 0.012)
        var_calc.update_returns("INFY", -0.005)
        ...
        result = var_calc.compute(positions, portfolio_value)
    """

    def __init__(
        self,
        ewma_lambda: float = 0.94,
        correlation_window: int = 60,
        min_history: int = 5,
        horizon_days: int = 1,
    ):
        self.ewma_lambda = ewma_lambda
        self.correlation_window = correlation_window
        self.min_history = min_history
        self.horizon_days = horizon_days

        # Per-symbol daily returns buffer: symbol -> list of daily returns
        self._returns: Dict[str, List[float]] = {}
        # EWMA variance cache: symbol -> current variance estimate
        self._ewma_var: Dict[str, float] = {}
        # Default vol for stocks with no history
        self._default_daily_vol = 0.025  # 2.5% daily vol (conservative)

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """Record a daily return observation for a symbol."""
        if not math.isfinite(daily_return):
            logger.warning("VaR: rejecting non-finite return for %s: %s", symbol, daily_return)
            return
        if symbol not in self._returns:
            self._returns[symbol] = []
        self._returns[symbol].append(daily_return)
        # Keep only last correlation_window * 2 observations
        max_len = self.correlation_window * 2
        if len(self._returns[symbol]) > max_len:
            self._returns[symbol] = self._returns[symbol][-max_len:]
        # Update EWMA variance
        self._update_ewma_var(symbol, daily_return)

    def _update_ewma_var(self, symbol: str, daily_return: float) -> None:
        lam = self.ewma_lambda
        if symbol not in self._ewma_var:
            # Initialize with squared return
            self._ewma_var[symbol] = daily_return ** 2
        else:
            self._ewma_var[symbol] = lam * self._ewma_var[symbol] + (1 - lam) * daily_return ** 2

    def get_daily_vol(self, symbol: str) -> float:
        """Get EWMA daily volatility estimate for a symbol."""
        if symbol in self._ewma_var and self._ewma_var[symbol] > 0:
            return math.sqrt(self._ewma_var[symbol])
        # Fallback: use rolling std if available
        returns = self._returns.get(symbol, [])
        if len(returns) >= self.min_history:
            return float(np.std(returns))
        return self._default_daily_vol

    def _correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Compute rolling pairwise correlation matrix."""
        n = len(symbols)
        if n == 0:
            return np.empty((0, 0))

        # Build return matrix: each column is a symbol's return series
        window = self.correlation_window
        max_len = max((len(self._returns.get(s, [])) for s in symbols), default=0)
        use_len = min(window, max_len)

        if use_len < self.min_history:
            # Not enough data — use conservative moderate correlation (0.5) instead of identity
            # Identity (uncorrelated) underestimates portfolio risk; 0.5 is a safer default
            return np.full((n, n), 0.5) + np.eye(n) * 0.5  # diagonal=1.0, off-diagonal=0.5

        returns_matrix = np.zeros((use_len, n))
        for j, sym in enumerate(symbols):
            r = self._returns.get(sym, [])
            if len(r) >= use_len:
                returns_matrix[:, j] = r[-use_len:]
            elif len(r) > 0:
                # Pad with zeros (conservative: reduces correlation)
                pad = use_len - len(r)
                returns_matrix[pad:, j] = r
            # else: all zeros (uncorrelated)

        # Compute correlation matrix
        try:
            corr = np.corrcoef(returns_matrix, rowvar=False)
            # Handle NaN (can happen if a column is all zeros)
            corr = np.nan_to_num(corr, nan=0.0)
            # Ensure diagonal is 1
            np.fill_diagonal(corr, 1.0)
            return corr
        except Exception as e:
            logger.error("Correlation matrix computation failed: %s — using conservative 0.5 default", e)
            return np.full((n, n), 0.5) + np.eye(n) * 0.5  # Conservative default

    def _covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """Build covariance matrix from EWMA vols and correlation matrix."""
        n = len(symbols)
        corr = self._correlation_matrix(symbols)
        vols = np.array([self.get_daily_vol(s) for s in symbols])
        # Cov = diag(vol) @ corr @ diag(vol)
        D = np.diag(vols)
        cov = D @ corr @ D
        # Regularize: add small epsilon to diagonal for numerical stability
        cov += np.eye(n) * 1e-10
        return cov

    def compute(
        self,
        positions: List[dict],
        portfolio_value: float,
    ) -> VaRResult:
        """
        Compute portfolio VaR.

        Args:
            positions: List of dicts with keys: symbol, notional (qty * price)
            portfolio_value: total portfolio equity

        Returns:
            VaRResult with VaR at 95% and 99% confidence
        """
        if not positions or portfolio_value <= 0:
            return VaRResult(horizon_days=self.horizon_days)

        symbols = [p["symbol"] for p in positions]
        notionals = np.array([p["notional"] for p in positions], dtype=float)
        n = len(symbols)

        if n == 0:
            return VaRResult(horizon_days=self.horizon_days)

        # Weights: notional / portfolio_value
        weights = notionals / portfolio_value if portfolio_value > 0 else np.zeros(n)

        # Covariance matrix
        cov = self._covariance_matrix(symbols)

        # Portfolio variance: w' * Sigma * w
        port_var = float(weights @ cov @ weights)
        port_vol_daily = math.sqrt(max(0, port_var))
        port_vol_annual = port_vol_daily * math.sqrt(252)

        # VaR = portfolio_value * z * portfolio_vol_daily * sqrt(horizon)
        sqrt_h = math.sqrt(self.horizon_days)
        var_95 = portfolio_value * Z_95 * port_vol_daily * sqrt_h
        var_99 = portfolio_value * Z_99 * port_vol_daily * sqrt_h
        var_95_pct = (var_95 / portfolio_value) * 100 if portfolio_value > 0 else 0
        var_99_pct = (var_99 / portfolio_value) * 100 if portfolio_value > 0 else 0

        # Per-position marginal VaR (component VaR)
        per_pos_var = {}
        if port_vol_daily > 0:
            # Marginal VaR_i = (Sigma @ w)_i / port_vol * w_i * VaR_95
            sigma_w = cov @ weights
            for i, sym in enumerate(symbols):
                marginal = (sigma_w[i] / port_vol_daily) * weights[i] * Z_95 * portfolio_value * sqrt_h
                per_pos_var[sym] = float(marginal)

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            var_95_pct=var_95_pct,
            var_99_pct=var_99_pct,
            portfolio_vol=port_vol_annual,
            portfolio_vol_daily=port_vol_daily,
            horizon_days=self.horizon_days,
            n_positions=n,
            per_position_var=per_pos_var,
        )

    def marginal_var_for_new_position(
        self,
        current_positions: List[dict],
        new_symbol: str,
        new_notional: float,
        portfolio_value: float,
    ) -> float:
        """
        Estimate VaR increase if a new position is added.
        Returns the marginal VaR (positive = increases risk).
        """
        if portfolio_value <= 0:
            return 0.0

        # Current VaR
        current_var = self.compute(current_positions, portfolio_value)

        # VaR with new position (use same portfolio_value base for consistent comparison)
        new_positions = current_positions + [{"symbol": new_symbol, "notional": new_notional}]
        new_var = self.compute(new_positions, portfolio_value)

        return new_var.var_95 - current_var.var_95

    def compute_cvar(
        self,
        positions: List[dict],
        portfolio_value: float,
        n_simulations: int = 10000,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute CVaR (Expected Shortfall) via historical simulation with correlation.

        Uses correlation-aware sampling from historical returns.
        CVaR = mean of worst (1-confidence)% portfolio returns.

        Returns CVaR as % of portfolio (positive = loss).
        """
        if not positions or portfolio_value <= 0:
            return 0.0

        symbols = [p["symbol"] for p in positions]
        notionals = np.array([p["notional"] for p in positions], dtype=float)
        n = len(symbols)

        if n == 0:
            return 0.0

        weights = notionals / portfolio_value if portfolio_value > 0 else np.zeros(n)

        # Build covariance matrix for correlated simulation
        cov = self._covariance_matrix(symbols)

        try:
            # Generate correlated random returns
            rng = np.random.default_rng(42)
            mean = np.zeros(n)
            simulated_returns = rng.multivariate_normal(mean, cov, size=n_simulations)

            # Portfolio returns: weighted sum
            portfolio_returns = simulated_returns @ weights

            # CVaR: mean of worst (1-confidence)% outcomes
            cutoff_idx = int(n_simulations * (1 - confidence))
            cutoff_idx = max(1, cutoff_idx)
            sorted_returns = np.sort(portfolio_returns)
            worst_returns = sorted_returns[:cutoff_idx]
            cvar = -float(np.mean(worst_returns))  # positive = loss

            return cvar * 100  # as percentage
        except Exception as e:
            logger.warning("CVaR Monte Carlo failed: %s — using historical fallback", e)
            # Fallback: attempt historical simulation from actual returns
            try:
                all_returns = []
                for sym in symbols:
                    r = self._returns.get(sym, [])
                    if r:
                        all_returns.append(np.array(r[-60:]))
                if all_returns:
                    min_len = min(len(r) for r in all_returns)
                    if min_len >= 10:
                        port_returns = sum(w * r[-min_len:] for w, r in zip(weights, all_returns))
                        sorted_losses = np.sort(port_returns)
                        cutoff = max(1, int(len(sorted_losses) * 0.05))
                        cvar = -float(np.mean(sorted_losses[:cutoff])) * 100
                        return cvar
            except Exception:
                pass
            # Ultimate fallback: VaR * 1.4
            var_result = self.compute(positions, portfolio_value)
            return var_result.var_95_pct * 1.4

    def check_var_limit(
        self,
        positions: List[dict],
        portfolio_value: float,
        max_var_pct: float = 5.0,
    ) -> Tuple[bool, float]:
        """
        Check if current portfolio VaR is within limit.

        Returns:
            (allowed, current_var_pct)
        """
        result = self.compute(positions, portfolio_value)
        return result.var_95_pct <= max_var_pct, result.var_95_pct

    def check_cvar_limit(
        self,
        positions: List[dict],
        portfolio_value: float,
        max_cvar_pct: float = 8.0,
    ) -> Tuple[bool, float]:
        """Check if CVaR is within limit. Returns (allowed, cvar_pct)."""
        cvar_pct = self.compute_cvar(positions, portfolio_value)
        return cvar_pct <= max_cvar_pct, cvar_pct

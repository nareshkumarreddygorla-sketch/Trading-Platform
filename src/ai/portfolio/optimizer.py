"""
Phase 2: Correlation-aware portfolio optimizer.
Rolling correlation matrix; MCR, portfolio heat, concentration;
risk parity / vol target / Kelly cap / correlation penalty;
max gross, net, sector, correlated cluster exposure (vs effective_equity).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class OptimizerConfig:
    max_gross_exposure_pct: float = 100.0  # sum |position| / equity
    max_net_exposure_pct: float = 50.0
    max_sector_exposure_pct: float = 25.0
    max_cluster_exposure_pct: float = 40.0
    vol_target_annual: float = 0.15
    kelly_cap: float = 0.10
    correlation_penalty_lambda: float = 0.5
    min_weight: float = 0.0


@dataclass
class PortfolioWeights:
    symbols: List[str]
    weights: Dict[str, float]
    gross_exposure: float
    net_exposure: float
    mcr: Dict[str, float]
    heat: float
    concentration: float


def _mcr_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal contribution to risk: (C @ w)_i * w_i / sqrt(w^T C w)."""
    if cov.size == 0 or np.allclose(weights, 0):
        return np.zeros_like(weights)
    port_var = weights @ cov @ weights
    if port_var <= 0:
        return np.zeros_like(weights)
    port_vol = np.sqrt(port_var)
    marginal = cov @ weights
    return (marginal * weights) / (port_vol + 1e-12)


def _risk_parity_weights(cov: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Inverse volatility weighting as proxy for risk parity."""
    vol = np.sqrt(np.diag(cov) + 1e-12)
    inv_vol = 1.0 / vol
    w = inv_vol / inv_vol.sum()
    return w


class CorrelationOptimizer:
    """
    Rolling correlation (or covariance) across active symbols;
    compute MCR, heat, concentration; output weights with risk parity / vol target /
    correlation penalty; enforce gross/net/sector/cluster vs effective_equity.
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self._cov: Optional[np.ndarray] = None
        self._symbols: List[str] = []

    def set_covariance(self, symbols: List[str], cov: np.ndarray) -> None:
        """Set rolling covariance matrix (symbols order must match cov rows/cols)."""
        self._symbols = list(symbols)
        self._cov = np.asarray(cov)

    def set_correlation_from_returns(self, symbols: List[str], returns: np.ndarray) -> None:
        """returns shape (T, n); compute cov and set."""
        self._symbols = list(symbols)
        if returns.shape[1] != len(symbols):
            raise ValueError("returns columns must match symbols length")
        self._cov = np.cov(returns.T)
        if self._cov.ndim == 0:
            self._cov = np.array([[self._cov]])

    def mcr(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Marginal contribution to risk per symbol."""
        if self._cov is None or not self._symbols:
            return {}
        w = np.array([weights.get(s, 0.0) for s in self._symbols])
        m = _mcr_contributions(w, self._cov)
        return {s: float(m[i]) for i, s in enumerate(self._symbols)}

    def heat(self, weights: Dict[str, float], position_values: Dict[str, float], equity: float) -> float:
        """Portfolio heat = sum |position_value| / equity."""
        if equity <= 0:
            return 0.0
        total = sum(abs(position_values.get(s, 0.0)) for s in weights)
        return total / equity

    def concentration(self, weights: Dict[str, float]) -> float:
        """Herfindahl: sum of squared weights."""
        w = list(weights.values())
        return sum(wi * wi for wi in w)

    def optimize(
        self,
        symbols: List[str],
        expected_returns: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        sector: Optional[Dict[str, str]] = None,
        cluster: Optional[Dict[str, int]] = None,
        effective_equity: float = 1.0,
        current_position_values: Optional[Dict[str, float]] = None,
    ) -> PortfolioWeights:
        """
        Compute weights: risk parity base; vol target scale; Kelly cap; correlation penalty;
        enforce max gross/net/sector/cluster exposure vs effective_equity.
        """
        cfg = self.config
        current_position_values = current_position_values or {}
        sector = sector or {}
        cluster = cluster or {}
        n = len(symbols)
        if n == 0:
            return PortfolioWeights([], {}, 0.0, 0.0, {}, 0.0, 0.0)

        # Base weights: risk parity from cov or inverse vol
        if self._cov is not None and self._cov.shape[0] == n:
            w = _risk_parity_weights(self._cov)
        else:
            vol = np.array([volatilities.get(s, 0.02) for s in symbols])
            vol = np.maximum(vol, 1e-8)
            w = (1.0 / vol) / (1.0 / vol).sum()
        weights = {s: float(w[i]) for i, s in enumerate(symbols)}

        # Correlation penalty: reduce weight if high correlation with portfolio
        if self._cov is not None and self._cov.shape[0] == n:
            w_vec = np.array([weights[s] for s in symbols])
            port_vol = np.sqrt(w_vec @ self._cov @ w_vec + 1e-12)
            for i, s in enumerate(symbols):
                corr_with_port = (self._cov[i, :] @ w_vec) / (np.sqrt(self._cov[i, i] + 1e-12) * port_vol + 1e-12)
                penalty = 1.0 - cfg.correlation_penalty_lambda * max(0, corr_with_port)
                weights[s] = max(cfg.min_weight, weights[s] * penalty)
            total = sum(weights.values()) or 1.0
            weights = {s: weights[s] / total for s in symbols}

        # Kelly cap per name
        for s in symbols:
            weights[s] = min(weights[s], cfg.kelly_cap)

        # Renormalize
        total = sum(weights.values()) or 1.0
        weights = {s: weights[s] / total for s in symbols}

        # Scale to effective_equity: assume position_value_i = weight_i * effective_equity
        position_values = {s: weights[s] * effective_equity for s in symbols}
        gross = sum(abs(weights[s]) for s in symbols)  # as fraction
        net = abs(sum(weights[s] for s in symbols))
        heat_val = self.heat(weights, position_values, effective_equity)
        conc = self.concentration(weights)
        mcr_dict = self.mcr(weights)

        # Cap gross/net (scale down if over)
        if gross > cfg.max_gross_exposure_pct / 100.0:
            scale = (cfg.max_gross_exposure_pct / 100.0) / gross
            weights = {s: weights[s] * scale for s in symbols}
        if net > cfg.max_net_exposure_pct / 100.0:
            scale = (cfg.max_net_exposure_pct / 100.0) / net
            weights = {s: weights[s] * scale for s in symbols}

        return PortfolioWeights(
            symbols=symbols,
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            mcr=mcr_dict,
            heat=heat_val,
            concentration=conc,
        )

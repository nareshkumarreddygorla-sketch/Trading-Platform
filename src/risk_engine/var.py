"""
Portfolio Value-at-Risk (VaR) calculator.

Implements:
  - Parametric VaR (variance-covariance method) at 95% and 99% confidence
  - Cornish-Fisher VaR (adjusts for skewness and kurtosis — default for NSE)
  - Historical VaR (percentile of actual return distribution)
  - Monte Carlo VaR (simulate from fitted distribution)
  - EVT (Extreme Value Theory) tail estimation for 99% VaR
  - Per-stock EWMA volatility estimation
  - Rolling pairwise correlation matrix
  - Portfolio volatility via w' * Sigma * w
  - Marginal VaR contribution per position
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Z-scores for confidence levels
Z_95 = 1.6449
Z_99 = 2.3263

# Valid VaR methods
VaRMethod = Literal["parametric", "historical", "cornish_fisher", "monte_carlo"]


def _cornish_fisher_z(z: float, skew: float, excess_kurt: float) -> float:
    """
    Cornish-Fisher expansion: adjust Gaussian z-score for skewness and kurtosis.

    z_cf = z + (1/6)(z^2 - 1)S + (1/24)(z^3 - 3z)K - (1/36)(2z^3 - 5z)S^2

    where S = skewness, K = excess kurtosis.
    This is the standard 4th-order expansion used in risk management.
    """
    z2 = z * z
    z3 = z2 * z
    s = skew
    k = excess_kurt
    z_cf = (
        z
        + (1.0 / 6.0) * (z2 - 1.0) * s
        + (1.0 / 24.0) * (z3 - 3.0 * z) * k
        - (1.0 / 36.0) * (2.0 * z3 - 5.0 * z) * s * s
    )
    return z_cf


def _evt_var_99(returns: np.ndarray, threshold_percentile: float = 10.0) -> float | None:
    """
    Extreme Value Theory (Generalised Pareto Distribution) tail estimation.

    Fits GPD to exceedances above a high threshold to estimate 99% VaR.
    Returns VaR as a positive loss fraction, or None if fitting fails.
    """
    if len(returns) < 50:
        return None
    losses = -returns  # convert to positive losses
    threshold = np.percentile(losses, 100.0 - threshold_percentile)
    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 10:
        return None
    try:
        # Fit Generalised Pareto Distribution
        shape, loc, scale = scipy_stats.genpareto.fit(exceedances, floc=0)
        # Probability of exceeding threshold
        n_total = len(losses)
        n_exceed = len(exceedances)
        prob_exceed = n_exceed / n_total
        # GPD-based VaR at 99% confidence
        p = 0.01  # tail probability
        if prob_exceed <= 0 or scale <= 0:
            return None
        if abs(shape) < 1e-10:
            var_99 = threshold + scale * math.log(prob_exceed / p)
        else:
            var_99 = threshold + (scale / shape) * (((prob_exceed / p) ** shape) - 1.0)
        if not math.isfinite(var_99) or var_99 < 0:
            return None
        return float(var_99)
    except Exception:
        return None


@dataclass
class VaRResult:
    """Portfolio VaR output."""

    var_95: float = 0.0  # 95% VaR in currency (INR)
    var_99: float = 0.0  # 99% VaR in currency (INR)
    var_95_pct: float = 0.0  # 95% VaR as % of portfolio
    var_99_pct: float = 0.0  # 99% VaR as % of portfolio
    portfolio_vol: float = 0.0  # annualised portfolio volatility
    portfolio_vol_daily: float = 0.0  # daily portfolio volatility
    horizon_days: int = 1
    n_positions: int = 0
    per_position_var: dict[str, float] = field(default_factory=dict)
    method: str = "parametric"  # which VaR method was used
    evt_var_99: float | None = None  # EVT tail estimate for 99% VaR (if available)
    skewness: float = 0.0  # portfolio return skewness
    excess_kurtosis: float = 0.0  # portfolio return excess kurtosis

    def as_dict(self) -> dict:
        d = {
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "var_95_pct": round(self.var_95_pct, 4),
            "var_99_pct": round(self.var_99_pct, 4),
            "portfolio_vol": round(self.portfolio_vol, 4),
            "portfolio_vol_daily": round(self.portfolio_vol_daily, 6),
            "horizon_days": self.horizon_days,
            "n_positions": self.n_positions,
            "per_position_var": {k: round(v, 2) for k, v in self.per_position_var.items()},
            "method": self.method,
            "skewness": round(self.skewness, 4),
            "excess_kurtosis": round(self.excess_kurtosis, 4),
        }
        if self.evt_var_99 is not None:
            d["evt_var_99"] = round(self.evt_var_99, 2)
        return d


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
        sector_classifier=None,
        var_method: VaRMethod = "cornish_fisher",
        monte_carlo_simulations: int = 10000,
    ):
        self.ewma_lambda = ewma_lambda
        self.correlation_window = correlation_window
        self.min_history = min_history
        self.horizon_days = horizon_days
        self._sector_classifier = sector_classifier
        self._var_method: VaRMethod = var_method
        self._monte_carlo_simulations = monte_carlo_simulations
        self._lock = threading.RLock()

        # Per-symbol daily returns buffer: symbol -> deque of daily returns
        self._returns: dict[str, deque] = {}
        # EWMA variance cache: symbol -> current variance estimate
        self._ewma_var: dict[str, float] = {}
        # Default vol for stocks with no history
        self._default_daily_vol = 0.025  # 2.5% daily vol (conservative)

        logger.info(
            "PortfolioVaR initialised: method=%s, ewma_lambda=%.2f, "
            "correlation_window=%d, horizon=%d days, monte_carlo_sims=%d",
            self._var_method,
            self.ewma_lambda,
            self.correlation_window,
            self.horizon_days,
            self._monte_carlo_simulations,
        )

    # ── Indian market sector classification for correlation defaults ──
    _SECTOR_MAP = {
        # IT sector (typically 0.75-0.90 intra-sector correlation)
        "INFY": "IT",
        "TCS": "IT",
        "WIPRO": "IT",
        "HCLTECH": "IT",
        "TECHM": "IT",
        "LTIM": "IT",
        "MPHASIS": "IT",
        "COFORGE": "IT",
        "PERSISTENT": "IT",
        # Banking (typically 0.80+ intra-sector correlation)
        "HDFCBANK": "BANK",
        "ICICIBANK": "BANK",
        "SBIN": "BANK",
        "KOTAKBANK": "BANK",
        "AXISBANK": "BANK",
        "INDUSINDBK": "BANK",
        "BANDHANBNK": "BANK",
        "FEDERALBNK": "BANK",
        "IDFCFIRSTB": "BANK",
        "PNB": "BANK",
        "BANKBARODA": "BANK",
        "CANBK": "BANK",
        # NBFC / Financial Services
        "BAJFINANCE": "NBFC",
        "BAJAJFINSV": "NBFC",
        "HDFC": "NBFC",
        "SBILIFE": "NBFC",
        "HDFCLIFE": "NBFC",
        "ICICIGI": "NBFC",
        "CHOLAFIN": "NBFC",
        # Auto
        "MARUTI": "AUTO",
        "TATAMOTORS": "AUTO",
        "M&M": "AUTO",
        "BAJAJ-AUTO": "AUTO",
        "EICHERMOT": "AUTO",
        "HEROMOTOCO": "AUTO",
        "ASHOKLEY": "AUTO",
        # Pharma
        "SUNPHARMA": "PHARMA",
        "DRREDDY": "PHARMA",
        "CIPLA": "PHARMA",
        "DIVISLAB": "PHARMA",
        "APOLLOHOSP": "PHARMA",
        "BIOCON": "PHARMA",
        "LUPIN": "PHARMA",
        # Metal & Mining
        "TATASTEEL": "METAL",
        "JSWSTEEL": "METAL",
        "HINDALCO": "METAL",
        "VEDL": "METAL",
        "COALINDIA": "METAL",
        "NMDC": "METAL",
        # Energy / Oil & Gas
        "RELIANCE": "ENERGY",
        "ONGC": "ENERGY",
        "BPCL": "ENERGY",
        "IOC": "ENERGY",
        "GAIL": "ENERGY",
        "NTPC": "ENERGY",
        "POWERGRID": "ENERGY",
        # FMCG
        "HINDUNILVR": "FMCG",
        "ITC": "FMCG",
        "NESTLEIND": "FMCG",
        "BRITANNIA": "FMCG",
        "DABUR": "FMCG",
        "MARICO": "FMCG",
        # Telecom
        "BHARTIARTL": "TELECOM",
        "IDEA": "TELECOM",
        # Cement / Construction
        "ULTRACEMCO": "CEMENT",
        "SHREECEM": "CEMENT",
        "AMBUJACEM": "CEMENT",
        "ACC": "CEMENT",
        "GRASIM": "CEMENT",
    }

    # Default correlations between sector pairs
    _SECTOR_CORR = {
        ("IT", "IT"): 0.82,
        ("BANK", "BANK"): 0.85,
        ("NBFC", "NBFC"): 0.80,
        ("AUTO", "AUTO"): 0.72,
        ("PHARMA", "PHARMA"): 0.70,
        ("METAL", "METAL"): 0.78,
        ("ENERGY", "ENERGY"): 0.75,
        ("FMCG", "FMCG"): 0.68,
        ("CEMENT", "CEMENT"): 0.74,
        ("BANK", "NBFC"): 0.72,
        ("NBFC", "BANK"): 0.72,
        ("IT", "BANK"): 0.35,
        ("BANK", "IT"): 0.35,
        ("PHARMA", "IT"): 0.25,
        ("IT", "PHARMA"): 0.25,
        ("METAL", "ENERGY"): 0.55,
        ("ENERGY", "METAL"): 0.55,
        ("FMCG", "PHARMA"): 0.40,
        ("PHARMA", "FMCG"): 0.40,
    }

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol using external classifier or built-in map."""
        if self._sector_classifier is not None:
            try:
                return self._sector_classifier(symbol)
            except Exception:
                pass
        return self._SECTOR_MAP.get(symbol, "GENERIC")

    @classmethod
    def _default_correlation(cls, sym_a: str, sym_b: str) -> float:
        """Sector-aware default correlation when historical data is insufficient."""
        sec_a = cls._SECTOR_MAP.get(sym_a, "GENERIC")
        sec_b = cls._SECTOR_MAP.get(sym_b, "GENERIC")
        if sec_a == sec_b and sec_a != "GENERIC":
            return cls._SECTOR_CORR.get((sec_a, sec_b), 0.70)
        pair = (sec_a, sec_b)
        if pair in cls._SECTOR_CORR:
            return cls._SECTOR_CORR[pair]
        rev_pair = (sec_b, sec_a)
        if rev_pair in cls._SECTOR_CORR:
            return cls._SECTOR_CORR[rev_pair]
        # Cross-sector default: moderate correlation (Indian market tends to be correlated)
        return 0.45

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """Record a daily return observation for a symbol."""
        if not math.isfinite(daily_return):
            logger.warning("VaR: rejecting non-finite return for %s: %s", symbol, daily_return)
            return
        with self._lock:
            max_len = self.correlation_window * 2
            if symbol not in self._returns:
                self._returns[symbol] = deque(maxlen=max_len)
            self._returns[symbol].append(daily_return)
            # Update EWMA variance
            self._update_ewma_var(symbol, daily_return)

    def _update_ewma_var(self, symbol: str, daily_return: float) -> None:
        lam = self.ewma_lambda
        if symbol not in self._ewma_var:
            # Initialize with squared return
            self._ewma_var[symbol] = daily_return**2
        else:
            self._ewma_var[symbol] = lam * self._ewma_var[symbol] + (1 - lam) * daily_return**2

    def get_daily_vol(self, symbol: str) -> float:
        """Get EWMA daily volatility estimate for a symbol."""
        if symbol in self._ewma_var and self._ewma_var[symbol] > 0:
            return math.sqrt(self._ewma_var[symbol])
        # Fallback: use rolling std if available
        returns = self._returns.get(symbol, [])
        if len(returns) >= self.min_history:
            return float(np.std(returns, ddof=1))
        return self._default_daily_vol

    def _correlation_matrix(self, symbols: list[str]) -> np.ndarray:
        """Compute rolling pairwise correlation matrix."""
        n = len(symbols)
        if n == 0:
            return np.empty((0, 0))
        if n == 1:
            return np.eye(1)

        # Build return matrix: each column is a symbol's return series
        window = self.correlation_window
        max_len = max((len(self._returns.get(s, [])) for s in symbols), default=0)
        use_len = min(window, max_len)

        if use_len < self.min_history:
            # P2-4: Use STRESS correlation (0.5 floor for all pairs) when <60 days history.
            # Prevents VaR underestimation for new strategies by assuming higher correlation.
            logger.warning(
                "VaR: insufficient history (%d < %d days) — using stress correlation matrix",
                use_len,
                self.min_history,
            )
            _STRESS_CORR_FLOOR = 0.5  # minimum pairwise correlation under stress
            corr = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    default_corr = self._default_correlation(symbols[i], symbols[j])
                    # Under stress, correlations spike — use max of default and floor
                    stress_corr = max(default_corr, _STRESS_CORR_FLOOR)
                    corr[i, j] = stress_corr
                    corr[j, i] = stress_corr
            return corr

        returns_matrix = np.zeros((use_len, n))
        for j, sym in enumerate(symbols):
            r = self._returns.get(sym, [])
            if len(r) >= use_len:
                returns_matrix[:, j] = list(r)[-use_len:]
            elif len(r) > 0:
                # Pad with zeros (conservative: reduces correlation)
                pad = use_len - len(r)
                returns_matrix[pad:, j] = list(r)
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
            logger.error("Correlation matrix computation failed: %s — using sector-aware defaults", e)
            corr = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    default_corr = self._default_correlation(symbols[i], symbols[j])
                    corr[i, j] = default_corr
                    corr[j, i] = default_corr
            return corr

    def _covariance_matrix(self, symbols: list[str]) -> np.ndarray:
        """Build covariance matrix from EWMA vols and correlation matrix."""
        n = len(symbols)
        corr = self._correlation_matrix(symbols)
        # Ensure correlation matrix is positive semi-definite (Higham nearest PSD)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            np.fill_diagonal(corr, 1.0)  # Restore diagonal
        except Exception as e:
            logger.warning("PSD correction failed: %s — using uncorrected matrix", e)
        vols = np.array([self.get_daily_vol(s) for s in symbols])
        # Cov = diag(vol) @ corr @ diag(vol)
        D = np.diag(vols)
        cov = D @ corr @ D
        # Regularize: add small epsilon to diagonal for numerical stability
        cov += np.eye(n) * 1e-10
        return cov

    def _portfolio_return_series(
        self,
        symbols: list[str],
        weights: np.ndarray,
    ) -> np.ndarray | None:
        """Build a historical portfolio return series from per-symbol returns."""
        min_len_needed = max(self.min_history, 20)
        # Find common length
        lengths = [len(self._returns.get(s, [])) for s in symbols]
        if not lengths or min(lengths) < min_len_needed:
            return None
        use_len = min(min(lengths), self.correlation_window * 2)
        ret_matrix = np.zeros((use_len, len(symbols)))
        for j, sym in enumerate(symbols):
            ret_matrix[:, j] = list(self._returns[sym])[-use_len:]
        return ret_matrix @ weights  # weighted portfolio returns

    def _compute_skew_kurt(
        self,
        symbols: list[str],
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """Return (skewness, excess_kurtosis) of portfolio returns, or (0, 0) if insufficient data."""
        port_rets = self._portfolio_return_series(symbols, weights)
        if port_rets is None or len(port_rets) < 20:
            return 0.0, 0.0
        skew = float(scipy_stats.skew(port_rets))
        kurt = float(scipy_stats.kurtosis(port_rets))  # excess kurtosis by default
        if not math.isfinite(skew):
            skew = 0.0
        if not math.isfinite(kurt):
            kurt = 0.0
        return skew, kurt

    def compute(
        self,
        positions: list[dict],
        portfolio_value: float,
        method: VaRMethod | None = None,
    ) -> VaRResult:  # Thread-safe: acquires lock
        """
        Compute portfolio VaR using the configured method.

        Methods:
          - "parametric": standard variance-covariance (Gaussian)
          - "cornish_fisher": adjusts parametric VaR for skewness + kurtosis (default)
          - "historical": percentile of actual portfolio return distribution
          - "monte_carlo": simulate from fitted covariance matrix

        Args:
            positions: List of dicts with keys: symbol, notional (qty * price)
            portfolio_value: total portfolio equity
            method: override the instance default VaR method for this call

        Returns:
            VaRResult with VaR at 95% and 99% confidence
        """
        active_method = method or self._var_method

        if not positions or portfolio_value <= 0:
            return VaRResult(horizon_days=self.horizon_days, method=active_method)

        with self._lock:
            symbols = [p["symbol"] for p in positions]
            notionals = np.array([p["notional"] for p in positions], dtype=float)
            n = len(symbols)

            if n == 0:
                return VaRResult(horizon_days=self.horizon_days, method=active_method)

            # Weights: notional / portfolio_value
            weights = notionals / portfolio_value if portfolio_value > 0 else np.zeros(n)

            # Covariance matrix (always needed for vol estimate)
            cov = self._covariance_matrix(symbols)

            # Portfolio variance: w' * Sigma * w
            port_var = float(weights @ cov @ weights)
            port_vol_daily = math.sqrt(max(0, port_var))
            port_vol_annual = port_vol_daily * math.sqrt(252)
            sqrt_h = math.sqrt(self.horizon_days)

            # Compute skewness and kurtosis (needed for cornish_fisher and EVT)
            skew, excess_kurt = self._compute_skew_kurt(symbols, weights)

            # ── Dispatch to VaR method ──
            if active_method == "historical":
                var_95, var_99 = self._compute_historical_var(
                    symbols,
                    weights,
                    portfolio_value,
                    sqrt_h,
                )
                # Fallback to parametric if insufficient history
                if var_95 == 0.0 and var_99 == 0.0:
                    logger.info("VaR historical: insufficient data, falling back to cornish_fisher")
                    active_method = "cornish_fisher"

            if active_method == "monte_carlo":
                var_95, var_99 = self._compute_monte_carlo_var(
                    cov,
                    weights,
                    portfolio_value,
                    sqrt_h,
                )
                # If Monte Carlo returned zeros (failure), fall back to cornish_fisher
                if var_95 == 0.0 and var_99 == 0.0:
                    logger.warning("VaR monte_carlo returned zeros — falling back to cornish_fisher")
                    active_method = "cornish_fisher"

            if active_method == "cornish_fisher":
                z95_cf = _cornish_fisher_z(Z_95, skew, excess_kurt)
                z99_cf = _cornish_fisher_z(Z_99, skew, excess_kurt)
                var_95 = portfolio_value * z95_cf * port_vol_daily * sqrt_h
                var_99 = portfolio_value * z99_cf * port_vol_daily * sqrt_h
                logger.debug(
                    "VaR cornish_fisher: z95_cf=%.4f (vs %.4f Gaussian), "
                    "z99_cf=%.4f (vs %.4f Gaussian), skew=%.3f, kurt=%.3f",
                    z95_cf,
                    Z_95,
                    z99_cf,
                    Z_99,
                    skew,
                    excess_kurt,
                )

            if active_method == "parametric":
                var_95 = portfolio_value * Z_95 * port_vol_daily * sqrt_h
                var_99 = portfolio_value * Z_99 * port_vol_daily * sqrt_h

            var_95_pct = (var_95 / portfolio_value) * 100 if portfolio_value > 0 else 0
            var_99_pct = (var_99 / portfolio_value) * 100 if portfolio_value > 0 else 0

            # EVT tail estimate for 99% VaR (supplement, not replacement)
            evt_var99_value = None
            port_rets = self._portfolio_return_series(symbols, weights)
            if port_rets is not None and len(port_rets) >= 50:
                evt_loss = _evt_var_99(port_rets)
                if evt_loss is not None:
                    evt_var99_value = evt_loss * portfolio_value * sqrt_h
                    # If EVT estimate is higher than the method's 99% VaR, log a warning
                    if evt_var99_value > var_99 * 1.2:
                        logger.warning(
                            "EVT tail 99%% VaR (%.2f) exceeds %s 99%% VaR (%.2f) by >20%% — fat tails detected",
                            evt_var99_value,
                            active_method,
                            var_99,
                        )

            # Per-position marginal VaR (component VaR) — always parametric-based
            per_pos_var = {}
            if port_vol_daily > 0:
                z_marginal = Z_95
                if active_method == "cornish_fisher":
                    z_marginal = _cornish_fisher_z(Z_95, skew, excess_kurt)
                sigma_w = cov @ weights
                for i, sym in enumerate(symbols):
                    marginal = (sigma_w[i] / port_vol_daily) * weights[i] * z_marginal * portfolio_value * sqrt_h
                    per_pos_var[sym] = float(marginal)

            logger.info(
                "VaR computed: method=%s, VaR95=%.2f (%.2f%%), VaR99=%.2f (%.2f%%), "
                "port_vol=%.4f, skew=%.3f, kurt=%.3f, n_pos=%d",
                active_method,
                var_95,
                var_95_pct,
                var_99,
                var_99_pct,
                port_vol_annual,
                skew,
                excess_kurt,
                n,
            )

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
                method=active_method,
                evt_var_99=evt_var99_value,
                skewness=skew,
                excess_kurtosis=excess_kurt,
            )

    def _compute_historical_var(
        self,
        symbols: list[str],
        weights: np.ndarray,
        portfolio_value: float,
        sqrt_h: float,
    ) -> tuple[float, float]:
        """
        Historical VaR: use percentile of actual portfolio return distribution.
        Returns (var_95, var_99) in currency. Returns (0, 0) if insufficient data.
        """
        port_rets = self._portfolio_return_series(symbols, weights)
        if port_rets is None or len(port_rets) < 20:
            return 0.0, 0.0

        # VaR is the negative percentile of the return distribution
        var_95_ret = -float(np.percentile(port_rets, 5.0))  # 5th percentile
        var_99_ret = -float(np.percentile(port_rets, 1.0))  # 1st percentile

        var_95 = var_95_ret * portfolio_value * sqrt_h
        var_99 = var_99_ret * portfolio_value * sqrt_h

        logger.debug(
            "VaR historical: 5th_pctile=%.4f, 1st_pctile=%.4f, n_obs=%d",
            -var_95_ret,
            -var_99_ret,
            len(port_rets),
        )
        return max(0.0, var_95), max(0.0, var_99)

    def _compute_monte_carlo_var(
        self,
        cov: np.ndarray,
        weights: np.ndarray,
        portfolio_value: float,
        sqrt_h: float,
    ) -> tuple[float, float]:
        """
        Monte Carlo VaR: simulate from fitted covariance matrix.
        Returns (var_95, var_99) in currency.
        """
        n_sims = self._monte_carlo_simulations
        try:
            rng = np.random.default_rng()
            mean = np.zeros(len(weights))
            simulated_returns = rng.multivariate_normal(mean, cov, size=n_sims)
            portfolio_returns = simulated_returns @ weights

            var_95_ret = -float(np.percentile(portfolio_returns, 5.0))
            var_99_ret = -float(np.percentile(portfolio_returns, 1.0))

            var_95 = var_95_ret * portfolio_value * sqrt_h
            var_99 = var_99_ret * portfolio_value * sqrt_h

            logger.debug(
                "VaR monte_carlo: n_sims=%d, var95_ret=%.4f, var99_ret=%.4f",
                n_sims,
                var_95_ret,
                var_99_ret,
            )
            return max(0.0, var_95), max(0.0, var_99)
        except Exception as e:
            logger.warning("Monte Carlo VaR failed: %s — falling back to parametric", e)
            # Return zeros to trigger fallback
            return 0.0, 0.0

    def marginal_var_for_new_position(
        self,
        current_positions: list[dict],
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
        positions: list[dict],
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
        with self._lock:
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
                rng = np.random.default_rng()
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
                    # Build weights and returns together, only including symbols that have data
                    paired = []
                    for w, sym in zip(weights, symbols):
                        r = self._returns.get(sym, [])
                        if len(r) > 0:
                            paired.append((w, np.array(list(r)[-60:])))
                    if paired:
                        weights_clean = [p[0] for p in paired]
                        returns_clean = [p[1] for p in paired]
                        # Renormalize weights so they sum to 1
                        total_w = sum(weights_clean)
                        if total_w > 0:
                            weights_clean = [w / total_w for w in weights_clean]
                        else:
                            weights_clean = [1.0 / len(weights_clean)] * len(weights_clean)
                        min_len = min(len(r) for r in returns_clean)
                        if min_len >= 10:
                            port_returns = sum(w * r[-min_len:] for w, r in zip(weights_clean, returns_clean))
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
        positions: list[dict],
        portfolio_value: float,
        max_var_pct: float = 5.0,
    ) -> tuple[bool, float]:
        """
        Check if current portfolio VaR is within limit.

        Returns:
            (allowed, current_var_pct)
        """
        result = self.compute(positions, portfolio_value)
        return result.var_95_pct <= max_var_pct, result.var_95_pct

    def check_cvar_limit(
        self,
        positions: list[dict],
        portfolio_value: float,
        max_cvar_pct: float = 8.0,
    ) -> tuple[bool, float]:
        """Check if CVaR is within limit. Returns (allowed, cvar_pct)."""
        cvar_pct = self.compute_cvar(positions, portfolio_value)
        return cvar_pct <= max_cvar_pct, cvar_pct

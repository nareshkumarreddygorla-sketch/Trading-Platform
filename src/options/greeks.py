"""Options Greeks calculator using Black-Scholes model."""

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


@dataclass
class GreeksResult:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per 1% vol change
    rho: float
    iv: float | None = None


def black_scholes(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> GreeksResult:
    """Compute option price and Greeks via Black-Scholes.

    Parameters
    ----------
    spot : current underlying price
    strike : option strike price
    time_to_expiry : years to expiry (e.g. 30/365)
    risk_free_rate : annualised risk-free rate
    volatility : annualised volatility (sigma)
    option_type : ``"call"`` or ``"put"``
    """
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return GreeksResult(price=0, delta=0, gamma=0, theta=0, vega=0, rho=0)

    t, s, k, r, v = time_to_expiry, spot, strike, risk_free_rate, volatility
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    is_call = option_type.lower() == "call"
    discount = math.exp(-r * t)

    if is_call:
        price = s * _norm_cdf(d1) - k * discount * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        rho = k * t * discount * _norm_cdf(d2) / 100
        theta_sign = -1
    else:
        price = k * discount * _norm_cdf(-d2) - s * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
        rho = -k * t * discount * _norm_cdf(-d2) / 100
        theta_sign = -1

    gamma = _norm_pdf(d1) / (s * v * sqrt_t)
    theta = -(s * _norm_pdf(d1) * v / (2 * sqrt_t)) - r * k * discount * _norm_cdf(d2 if is_call else -d2) * theta_sign
    theta /= 365  # per calendar day
    vega = s * _norm_pdf(d1) * sqrt_t / 100  # per 1% vol change

    return GreeksResult(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Implied volatility via Newton-Raphson."""
    vol = 0.3
    for _ in range(max_iter):
        result = black_scholes(spot, strike, time_to_expiry, risk_free_rate, vol, option_type)
        diff = result.price - market_price
        if abs(diff) < tol:
            return vol
        vega_raw = result.vega * 100  # undo /100
        if abs(vega_raw) < 1e-10:
            break
        vol -= diff / vega_raw
        vol = max(0.01, min(5.0, vol))
    return vol

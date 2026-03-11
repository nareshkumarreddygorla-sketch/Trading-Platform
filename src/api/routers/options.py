"""API router for options trading (chain, Greeks, IV surface)."""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

_chain_mgr = None


def _get_mgr():
    global _chain_mgr
    if _chain_mgr is None:
        from src.options.chain import OptionsChainManager

        _chain_mgr = OptionsChainManager()
    return _chain_mgr


# Default spot prices for quick demo
_SPOT_DEFAULTS = {
    "NIFTY": 22500,
    "BANKNIFTY": 48000,
    "RELIANCE": 2500,
    "TCS": 3800,
    "INFY": 1500,
    "HDFCBANK": 1600,
    "SBIN": 750,
    "TATAMOTORS": 900,
    "ITC": 450,
}
_LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 550,
    "SBIN": 1500,
    "TATAMOTORS": 1400,
    "ITC": 1600,
}


class GreeksRequest(BaseModel):
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    days_to_expiry: int = Field(..., ge=1, le=365)
    volatility: float = Field(default=0.25, gt=0, le=5.0)
    risk_free_rate: float = Field(default=0.065, ge=0, le=0.5)
    option_type: str = Field(default="call")


@router.get("/options/chain/{underlying}")
async def get_options_chain(
    underlying: str,
    expiry: str | None = None,
    spot_price: float = Query(default=0, ge=0),
    num_strikes: int = Query(default=11, ge=5, le=31),
    volatility: float = Query(default=0.25, gt=0, le=5.0),
):
    mgr = _get_mgr()
    key = underlying.upper()

    if not expiry:
        exp_date = datetime.now(UTC) + timedelta(days=30)
        expiry = exp_date.strftime("%Y-%m-%d")

    if spot_price <= 0:
        spot_price = _SPOT_DEFAULTS.get(key, 1000)

    lot_size = _LOT_SIZES.get(key, 1)
    strikes = mgr.generate_strikes(spot_price, num_strikes)
    days = max(1, (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days)

    chain = mgr.build_chain(key, spot_price, expiry, strikes, lot_size, volatility, days)
    pcr = mgr.pcr(chain)
    max_pain = mgr.max_pain(chain)

    def _contract(c):
        g = c.greeks
        return {
            "symbol": c.symbol,
            "strike": c.strike,
            "last_price": c.last_price,
            "bid": c.bid,
            "ask": c.ask,
            "iv": c.iv,
            "volume": c.volume,
            "oi": c.open_interest,
            "delta": round(g.delta, 4) if g else 0,
            "gamma": round(g.gamma, 6) if g else 0,
            "theta": round(g.theta, 4) if g else 0,
            "vega": round(g.vega, 4) if g else 0,
        }

    return {
        "underlying": chain.underlying,
        "spot_price": chain.spot_price,
        "expiry": chain.expiry,
        "days_to_expiry": days,
        "lot_size": lot_size,
        "max_pain": max_pain,
        "pcr": pcr,
        "calls": [_contract(c) for c in chain.calls],
        "puts": [_contract(p) for p in chain.puts],
    }


@router.post("/options/greeks")
async def calculate_greeks(req: GreeksRequest):
    from src.options.greeks import black_scholes

    t = req.days_to_expiry / 365.0
    r = black_scholes(req.spot, req.strike, t, req.risk_free_rate, req.volatility, req.option_type)
    return {
        "option_type": req.option_type,
        "spot": req.spot,
        "strike": req.strike,
        "days_to_expiry": req.days_to_expiry,
        "volatility": req.volatility,
        "price": round(r.price, 2),
        "delta": round(r.delta, 4),
        "gamma": round(r.gamma, 6),
        "theta": round(r.theta, 4),
        "vega": round(r.vega, 4),
        "rho": round(r.rho, 4),
    }


@router.get("/options/iv/{underlying}")
async def get_iv_surface(
    underlying: str,
    spot_price: float = Query(default=0, ge=0),
):
    """Implied volatility surface (term structure x moneyness)."""
    mgr = _get_mgr()
    key = underlying.upper()
    if spot_price <= 0:
        spot_price = _SPOT_DEFAULTS.get(key, 1000)

    from src.options.greeks import black_scholes as bs

    expiries_days = [7, 14, 30, 60, 90]
    surface = []
    for days in expiries_days:
        strikes = mgr.generate_strikes(spot_price, 7)
        t = days / 365.0
        for strike in strikes:
            moneyness = strike / spot_price
            # Smile: OTM options have higher IV, short-dated have higher IV
            vol = 0.20 + 0.15 * (1 - moneyness) ** 2 + 0.02 / max(t, 0.02)
            r = bs(spot_price, strike, t, 0.065, vol, "call")
            surface.append(
                {
                    "days_to_expiry": days,
                    "strike": strike,
                    "moneyness": round(moneyness, 3),
                    "iv": round(vol * 100, 1),
                    "call_price": round(r.price, 2),
                    "delta": round(r.delta, 4),
                }
            )

    return {"underlying": key, "spot_price": spot_price, "surface": surface}

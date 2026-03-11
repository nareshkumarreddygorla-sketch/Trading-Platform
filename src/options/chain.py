"""Options chain data management with Greeks."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .greeks import GreeksResult, black_scholes

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    symbol: str
    underlying: str
    strike: float
    expiry: str
    option_type: str  # "call" or "put"
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    iv: float = 0.0
    greeks: GreeksResult | None = None
    lot_size: int = 1
    exchange: str = "NSE"


@dataclass
class OptionsChain:
    underlying: str
    spot_price: float
    expiry: str
    calls: list[OptionContract] = field(default_factory=list)
    puts: list[OptionContract] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class OptionsChainManager:
    """Build and cache options chains with Greeks."""

    def __init__(self, risk_free_rate: float = 0.065):
        self._risk_free_rate = risk_free_rate
        self._chains: dict[str, OptionsChain] = {}

    def build_chain(
        self,
        underlying: str,
        spot_price: float,
        expiry: str,
        strikes: list[float],
        lot_size: int = 1,
        volatility: float = 0.25,
        days_to_expiry: int = 30,
    ) -> OptionsChain:
        t = max(days_to_expiry, 1) / 365.0
        calls: list[OptionContract] = []
        puts: list[OptionContract] = []

        for strike in sorted(strikes):
            # Volatility smile: OTM options have higher IV
            moneyness = strike / spot_price
            smile_adj = 0.10 * (1 - moneyness) ** 2
            vol = volatility + smile_adj

            cg = black_scholes(spot_price, strike, t, self._risk_free_rate, vol, "call")
            calls.append(
                OptionContract(
                    symbol=f"{underlying}{expiry}C{strike:.0f}",
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type="call",
                    last_price=round(cg.price, 2),
                    bid=round(max(0, cg.price * 0.98), 2),
                    ask=round(cg.price * 1.02, 2),
                    iv=round(vol * 100, 1),
                    greeks=cg,
                    lot_size=lot_size,
                )
            )
            pg = black_scholes(spot_price, strike, t, self._risk_free_rate, vol, "put")
            puts.append(
                OptionContract(
                    symbol=f"{underlying}{expiry}P{strike:.0f}",
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type="put",
                    last_price=round(pg.price, 2),
                    bid=round(max(0, pg.price * 0.98), 2),
                    ask=round(pg.price * 1.02, 2),
                    iv=round(vol * 100, 1),
                    greeks=pg,
                    lot_size=lot_size,
                )
            )

        chain = OptionsChain(underlying=underlying, spot_price=spot_price, expiry=expiry, calls=calls, puts=puts)
        self._chains[f"{underlying}:{expiry}"] = chain
        return chain

    def generate_strikes(self, spot_price: float, num_strikes: int = 11, interval_pct: float = 2.5) -> list[float]:
        """Generate evenly-spaced strikes centred on ATM."""
        if spot_price < 500:
            step = max(5, int(spot_price * interval_pct / 100 / 5) * 5)
            atm = round(spot_price / 10) * 10
        elif spot_price < 5000:
            step = max(50, int(spot_price * interval_pct / 100 / 50) * 50)
            atm = round(spot_price / 50) * 50
        else:
            step = max(100, int(spot_price * interval_pct / 100 / 100) * 100)
            atm = round(spot_price / 100) * 100
        half = num_strikes // 2
        return [atm + (i - half) * step for i in range(num_strikes)]

    def get_chain(self, underlying: str, expiry: str) -> OptionsChain | None:
        return self._chains.get(f"{underlying}:{expiry}")

    def max_pain(self, chain: OptionsChain) -> float:
        """Max-pain strike: where option writers lose least money."""
        if not chain.calls or not chain.puts:
            return chain.spot_price
        strikes = sorted({c.strike for c in chain.calls})
        best_strike, min_pain = strikes[len(strikes) // 2], float("inf")
        for st in strikes:
            pain = 0.0
            for c in chain.calls:
                if st > c.strike:
                    pain += (st - c.strike) * max(c.open_interest, 1) * c.lot_size
            for p in chain.puts:
                if st < p.strike:
                    pain += (p.strike - st) * max(p.open_interest, 1) * p.lot_size
            if pain < min_pain:
                min_pain, best_strike = pain, st
        return best_strike

    def pcr(self, chain: OptionsChain) -> dict[str, float]:
        """Put-Call ratio by volume and open interest."""
        cv = sum(c.volume for c in chain.calls) or 1
        pv = sum(p.volume for p in chain.puts) or 1
        co = sum(c.open_interest for c in chain.calls) or 1
        po = sum(p.open_interest for p in chain.puts) or 1
        pcr_oi = po / co
        return {
            "pcr_volume": round(pv / cv, 3),
            "pcr_oi": round(pcr_oi, 3),
            "sentiment": "bearish" if pcr_oi > 1.2 else ("bullish" if pcr_oi < 0.8 else "neutral"),
        }

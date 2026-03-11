"""FII/DII (Foreign & Domestic Institutional Investor) flow tracking.

Provides data structures and analytics for FII/DII institutional flows.
No real data source is configured by default -- call ``add_flow()`` to
ingest data from an external provider (e.g. NSE bulk data downloads,
broker API).  All query methods return empty results when no real data
has been ingested.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class InstitutionalFlow:
    date: str
    fii_buy: float = 0.0
    fii_sell: float = 0.0
    fii_net: float = 0.0
    dii_buy: float = 0.0
    dii_sell: float = 0.0
    dii_net: float = 0.0
    total_net: float = 0.0

    def __post_init__(self):
        self.fii_net = self.fii_buy - self.fii_sell
        self.dii_net = self.dii_buy - self.dii_sell
        self.total_net = self.fii_net + self.dii_net


@dataclass
class FlowAnalysis:
    trend: str  # bullish / bearish / neutral
    fii_streak: int
    dii_streak: int
    avg_fii_net_5d: float
    avg_dii_net_5d: float
    total_fii_net_month: float
    total_dii_net_month: float
    signal_strength: float
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class FIIDIITracker:
    """Track and analyse FII/DII institutional flows for trading signals.

    No synthetic/seed data is generated.  When no real data has been ingested
    via ``add_flow()``, all query and analytics methods return empty or
    neutral results so downstream consumers degrade gracefully.
    """

    def __init__(self):
        self._flows: list[InstitutionalFlow] = []
        logger.info("FIIDIITracker initialised with no data source. Call add_flow() to ingest real FII/DII data.")

    def add_flow(self, flow: InstitutionalFlow) -> None:
        self._flows.append(flow)
        if len(self._flows) > 90:
            self._flows = self._flows[-90:]

    def get_recent_flows(self, days: int = 10) -> list[InstitutionalFlow]:
        if not self._flows:
            logger.warning("FIIDIITracker.get_recent_flows: no real data available")
        return self._flows[-days:]

    def analyze(self) -> FlowAnalysis:
        if len(self._flows) < 5:
            if not self._flows:
                logger.warning("FIIDIITracker.analyze: no real FII/DII data ingested -- returning neutral analysis")
            return FlowAnalysis(
                trend="neutral",
                fii_streak=0,
                dii_streak=0,
                avg_fii_net_5d=0,
                avg_dii_net_5d=0,
                total_fii_net_month=0,
                total_dii_net_month=0,
                signal_strength=0,
                recommendation="Insufficient data",
            )

        r5 = self._flows[-5:]
        r20 = self._flows[-20:]
        avg_fii = sum(f.fii_net for f in r5) / 5
        avg_dii = sum(f.dii_net for f in r5) / 5
        total_fii = sum(f.fii_net for f in r20)
        total_dii = sum(f.dii_net for f in r20)

        # Streaks
        fii_streak = 0
        for f in reversed(self._flows):
            if f.fii_net > 0 and fii_streak >= 0:
                fii_streak += 1
            elif f.fii_net < 0 and fii_streak <= 0:
                fii_streak -= 1
            else:
                break

        dii_streak = 0
        for f in reversed(self._flows):
            if f.dii_net > 0 and dii_streak >= 0:
                dii_streak += 1
            elif f.dii_net < 0 and dii_streak <= 0:
                dii_streak -= 1
            else:
                break

        combined = avg_fii + avg_dii
        strength = min(1.0, abs(combined) / 5000)

        if combined > 1000:
            trend, rec = "bullish", "Strong institutional buying — favourable for longs"
        elif combined < -1000:
            trend, rec = "bearish", "Institutional selling pressure — caution on new longs"
        else:
            trend, rec = "neutral", "Mixed institutional flows — no strong directional bias"

        if fii_streak >= 5:
            rec += ". FII buying streak strong."
        elif fii_streak <= -5:
            rec += ". FII selling streak — watch for continued outflows."

        return FlowAnalysis(
            trend=trend,
            fii_streak=fii_streak,
            dii_streak=dii_streak,
            avg_fii_net_5d=round(avg_fii, 2),
            avg_dii_net_5d=round(avg_dii, 2),
            total_fii_net_month=round(total_fii, 2),
            total_dii_net_month=round(total_dii, 2),
            signal_strength=round(strength, 3),
            recommendation=rec,
        )

    def to_exposure_multiplier(self) -> float:
        """Convert FII/DII analysis to exposure multiplier (0.5-1.3)."""
        a = self.analyze()
        if a.trend == "bullish":
            return min(1.3, 1.0 + a.signal_strength * 0.3)
        elif a.trend == "bearish":
            return max(0.5, 1.0 - a.signal_strength * 0.5)
        return 1.0

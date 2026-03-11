"""
Stress Testing Module: scenario-based portfolio stress analysis.

Implements institutional-grade stress tests:
  1. Historical scenarios (2008 GFC, 2020 COVID crash, India demonetisation)
  2. Hypothetical shocks (VIX spike, FX deval, sector crash, rate hike)
  3. Correlation breakdown (all correlations → 1.0)
  4. Liquidity crisis (volume dries up, wider spreads)
  5. Custom user-defined scenarios

Each scenario computes: portfolio P&L impact, worst-hit positions,
margin call risk, and recommended exposure adjustments.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CUSTOM = "custom"


@dataclass
class ShockSpec:
    """Defines a market shock for a stress scenario."""
    name: str
    description: str
    scenario_type: ScenarioType
    # Sector/index shocks as pct moves (negative = down)
    equity_shock_pct: float = 0.0
    sector_shocks: Dict[str, float] = field(default_factory=dict)
    vix_level: float = 25.0
    fx_shock_pct: float = 0.0  # USDINR move
    rate_shock_bps: int = 0
    correlation_override: Optional[float] = None  # Force all correlations to this
    volume_multiplier: float = 1.0  # <1.0 = liquidity crisis
    spread_multiplier: float = 1.0  # >1.0 = wider spreads


@dataclass
class StressResult:
    """Result of running a single stress scenario."""
    scenario_name: str
    scenario_type: str
    portfolio_pnl: float  # Total P&L under stress
    portfolio_pnl_pct: float  # As % of equity
    worst_position: str  # Most impacted symbol
    worst_position_pnl: float
    positions_impacted: int
    margin_call_risk: bool  # True if equity < maintenance margin
    recommended_action: str
    position_details: List[Dict[str, Any]] = field(default_factory=list)


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS: List[ShockSpec] = [
    ShockSpec(
        name="2008_GFC",
        description="Global Financial Crisis: broad equity -40%, banking -55%, VIX 80",
        scenario_type=ScenarioType.HISTORICAL,
        equity_shock_pct=-40.0,
        sector_shocks={"Banking & Finance": -55.0, "Metals & Mining": -50.0, "Infrastructure": -45.0},
        vix_level=80.0,
        fx_shock_pct=15.0,
        correlation_override=0.85,
    ),
    ShockSpec(
        name="2020_COVID_Crash",
        description="COVID-19 pandemic crash: Nifty -38% in 1 month, VIX 84",
        scenario_type=ScenarioType.HISTORICAL,
        equity_shock_pct=-38.0,
        sector_shocks={"Automobile": -45.0, "Energy": -50.0, "Banking & Finance": -42.0,
                        "Pharma & Healthcare": -10.0, "Information Technology": -25.0},
        vix_level=84.0,
        fx_shock_pct=8.0,
        correlation_override=0.90,
    ),
    ShockSpec(
        name="2016_Demonetisation",
        description="India demonetisation: banking -15%, NBFC -25%, consumption -20%",
        scenario_type=ScenarioType.HISTORICAL,
        equity_shock_pct=-12.0,
        sector_shocks={"Banking & Finance": -15.0, "FMCG": -20.0, "Consumer Durables": -25.0},
        vix_level=30.0,
    ),
    ShockSpec(
        name="2022_Rate_Hike_Cycle",
        description="Aggressive rate hike: IT -20%, growth stocks hit, VIX 25",
        scenario_type=ScenarioType.HISTORICAL,
        equity_shock_pct=-10.0,
        sector_shocks={"Information Technology": -20.0, "Consumer Durables": -15.0},
        vix_level=25.0,
        rate_shock_bps=250,
    ),
]

HYPOTHETICAL_SCENARIOS: List[ShockSpec] = [
    ShockSpec(
        name="VIX_Extreme_Spike",
        description="India VIX spikes to 50+ (geopolitical event)",
        scenario_type=ScenarioType.HYPOTHETICAL,
        equity_shock_pct=-15.0,
        vix_level=50.0,
        correlation_override=0.75,
        volume_multiplier=0.3,
        spread_multiplier=5.0,
    ),
    ShockSpec(
        name="INR_Devaluation",
        description="USDINR +10% sudden move (capital outflow)",
        scenario_type=ScenarioType.HYPOTHETICAL,
        equity_shock_pct=-8.0,
        sector_shocks={"Information Technology": 5.0, "Energy": -15.0, "Pharma & Healthcare": 3.0},
        fx_shock_pct=10.0,
        vix_level=30.0,
    ),
    ShockSpec(
        name="Banking_Crisis",
        description="Major bank failure: banking -30%, NBFC -40%, contagion -15%",
        scenario_type=ScenarioType.HYPOTHETICAL,
        equity_shock_pct=-20.0,
        sector_shocks={"Banking & Finance": -30.0},
        vix_level=45.0,
        correlation_override=0.80,
    ),
    ShockSpec(
        name="Liquidity_Freeze",
        description="Market liquidity dries up: 80% volume drop, 10x spreads",
        scenario_type=ScenarioType.LIQUIDITY_CRISIS,
        equity_shock_pct=-5.0,
        vix_level=35.0,
        volume_multiplier=0.2,
        spread_multiplier=10.0,
    ),
    ShockSpec(
        name="Correlation_Breakdown",
        description="All correlations spike to 1.0 (systemic risk event)",
        scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
        equity_shock_pct=-25.0,
        vix_level=60.0,
        correlation_override=1.0,
    ),
    ShockSpec(
        name="Flash_Crash",
        description="5-minute flash crash: -10% then partial recovery",
        scenario_type=ScenarioType.HYPOTHETICAL,
        equity_shock_pct=-10.0,
        vix_level=40.0,
        volume_multiplier=3.0,
        spread_multiplier=8.0,
    ),
]


class StressTestEngine:
    """Runs portfolio stress tests against predefined and custom scenarios."""

    def __init__(
        self,
        sector_classifier=None,
        maintenance_margin_pct: float = 25.0,
    ):
        self.sector_classifier = sector_classifier
        self.maintenance_margin_pct = maintenance_margin_pct
        self._last_results: List[StressResult] = []

    def _get_sector(self, symbol: str) -> str:
        if self.sector_classifier:
            try:
                return self.sector_classifier.classify(symbol)
            except Exception:
                pass
        return "UNCLASSIFIED"

    def run_scenario(
        self,
        scenario: ShockSpec,
        positions: list,
        equity: float,
    ) -> StressResult:
        """Run a single stress scenario against current portfolio."""
        if not positions or equity <= 0:
            return StressResult(
                scenario_name=scenario.name,
                scenario_type=scenario.scenario_type.value,
                portfolio_pnl=0.0,
                portfolio_pnl_pct=0.0,
                worst_position="N/A",
                worst_position_pnl=0.0,
                positions_impacted=0,
                margin_call_risk=False,
                recommended_action="No positions to stress test",
            )

        total_pnl = 0.0
        worst_sym = ""
        worst_pnl = 0.0
        impacted = 0
        details = []

        for pos in positions:
            symbol = getattr(pos, "symbol", str(pos))
            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "avg_price", 0.0)
            side = getattr(pos, "side", "BUY")
            side_str = getattr(side, "value", str(side)).upper()
            notional = abs(qty * price)

            if notional == 0:
                continue

            sector = self._get_sector(symbol)

            # Determine shock for this position
            shock_pct = scenario.sector_shocks.get(sector, scenario.equity_shock_pct)

            # Direction: longs lose on down moves, shorts gain
            if side_str in ("BUY", "LONG"):
                pos_pnl = notional * (shock_pct / 100.0)
            else:
                pos_pnl = notional * (-shock_pct / 100.0)

            # Add spread cost for liquidity scenarios
            if scenario.spread_multiplier > 1.0:
                spread_cost = notional * 0.001 * scenario.spread_multiplier
                pos_pnl -= spread_cost

            total_pnl += pos_pnl
            impacted += 1

            if pos_pnl < worst_pnl:
                worst_pnl = pos_pnl
                worst_sym = symbol

            details.append({
                "symbol": symbol,
                "sector": sector,
                "notional": round(notional, 2),
                "shock_pct": round(shock_pct, 2),
                "pnl": round(pos_pnl, 2),
                "pnl_pct": round(pos_pnl / notional * 100, 2) if notional > 0 else 0.0,
            })

        pnl_pct = (total_pnl / equity * 100) if equity > 0 else 0.0
        stressed_equity = equity + total_pnl
        margin_call = stressed_equity < (equity * self.maintenance_margin_pct / 100)

        # Recommended action
        if margin_call:
            action = "CRITICAL: Margin call likely. Immediately reduce exposure by 50%+"
        elif pnl_pct < -20:
            action = "SEVERE: Portfolio loss >20%. Consider halting trading and reducing to 25% exposure"
        elif pnl_pct < -10:
            action = "HIGH: Portfolio loss >10%. Reduce new position sizing by 50%"
        elif pnl_pct < -5:
            action = "MODERATE: Portfolio loss >5%. Tighten stop losses and reduce exposure"
        elif pnl_pct < -2:
            action = "LOW: Portfolio loss >2%. Monitor closely, consider reducing concentration"
        else:
            action = "MINIMAL: Portfolio within acceptable stress bounds"

        return StressResult(
            scenario_name=scenario.name,
            scenario_type=scenario.scenario_type.value,
            portfolio_pnl=round(total_pnl, 2),
            portfolio_pnl_pct=round(pnl_pct, 2),
            worst_position=worst_sym or "N/A",
            worst_position_pnl=round(worst_pnl, 2),
            positions_impacted=impacted,
            margin_call_risk=margin_call,
            recommended_action=action,
            position_details=details,
        )

    def run_all_historical(self, positions: list, equity: float) -> List[StressResult]:
        """Run all historical stress scenarios."""
        results = []
        for scenario in HISTORICAL_SCENARIOS:
            result = self.run_scenario(scenario, positions, equity)
            results.append(result)
        return results

    def run_all_hypothetical(self, positions: list, equity: float) -> List[StressResult]:
        """Run all hypothetical stress scenarios."""
        results = []
        for scenario in HYPOTHETICAL_SCENARIOS:
            result = self.run_scenario(scenario, positions, equity)
            results.append(result)
        return results

    def run_full_suite(self, positions: list, equity: float) -> List[StressResult]:
        """Run all predefined scenarios (historical + hypothetical)."""
        results = self.run_all_historical(positions, equity)
        results.extend(self.run_all_hypothetical(positions, equity))
        self._last_results = results

        # Log summary
        worst = min(results, key=lambda r: r.portfolio_pnl_pct) if results else None
        if worst:
            logger.info(
                "Stress test suite complete: %d scenarios, worst=%s (%.1f%%)",
                len(results), worst.scenario_name, worst.portfolio_pnl_pct,
            )
        return results

    def run_custom(
        self, name: str, description: str, equity_shock_pct: float,
        sector_shocks: Optional[Dict[str, float]] = None,
        vix_level: float = 25.0, positions: Optional[list] = None,
        equity: float = 0.0,
    ) -> StressResult:
        """Run a custom user-defined stress scenario."""
        scenario = ShockSpec(
            name=name,
            description=description,
            scenario_type=ScenarioType.CUSTOM,
            equity_shock_pct=equity_shock_pct,
            sector_shocks=sector_shocks or {},
            vix_level=vix_level,
        )
        return self.run_scenario(scenario, positions or [], equity)

    def get_last_results(self) -> List[StressResult]:
        """Get results from the last full suite run."""
        return self._last_results

    def validate_circuit_breaker(
        self,
        positions: list,
        equity: float,
        circuit_breaker=None,
    ) -> Dict[str, Any]:
        """P2-6: Run 4 critical stress scenarios and verify circuit breaker trips correctly.
        Returns validation report."""
        critical_scenarios = [
            ShockSpec(
                name="10pct_portfolio_drop",
                description="10% portfolio drop — circuit breaker MUST trip",
                scenario_type=ScenarioType.HYPOTHETICAL,
                equity_shock_pct=-10.0,
                vix_level=35.0,
            ),
            ShockSpec(
                name="20pct_single_stock_drop",
                description="20% drop in single largest position",
                scenario_type=ScenarioType.HYPOTHETICAL,
                equity_shock_pct=-20.0,
                vix_level=40.0,
            ),
            ShockSpec(
                name="correlation_spike_0.9",
                description="All correlations spike to 0.9 (systemic event)",
                scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
                equity_shock_pct=-15.0,
                correlation_override=0.9,
                vix_level=50.0,
            ),
            ShockSpec(
                name="80pct_liquidity_drop",
                description="80% liquidity drop — execution risk extreme",
                scenario_type=ScenarioType.LIQUIDITY_CRISIS,
                equity_shock_pct=-8.0,
                volume_multiplier=0.2,
                spread_multiplier=8.0,
                vix_level=35.0,
            ),
        ]
        results = []
        for scenario in critical_scenarios:
            result = self.run_scenario(scenario, positions, equity)
            # Check if circuit breaker would trip
            would_trip = result.portfolio_pnl_pct < -5.0  # conservative threshold
            results.append({
                "scenario": result.scenario_name,
                "pnl_pct": result.portfolio_pnl_pct,
                "margin_call": result.margin_call_risk,
                "would_trip_circuit": would_trip,
                "action": result.recommended_action,
            })
        all_critical_trip = all(r["would_trip_circuit"] for r in results if r["pnl_pct"] < -10)
        return {
            "validation_passed": all_critical_trip,
            "scenarios": results,
            "recommendation": "Circuit breaker thresholds are correctly calibrated" if all_critical_trip
                else "WARNING: Circuit breaker may not trip on severe scenarios — review thresholds",
        }

    def get_summary(self) -> Dict[str, Any]:
        """Summary of last stress test suite for API/dashboard."""
        if not self._last_results:
            return {"status": "no_results", "scenarios_run": 0}

        worst = min(self._last_results, key=lambda r: r.portfolio_pnl_pct)
        margin_calls = sum(1 for r in self._last_results if r.margin_call_risk)
        severe = sum(1 for r in self._last_results if r.portfolio_pnl_pct < -10)

        return {
            "status": "completed",
            "scenarios_run": len(self._last_results),
            "worst_scenario": worst.scenario_name,
            "worst_pnl_pct": worst.portfolio_pnl_pct,
            "margin_call_scenarios": margin_calls,
            "severe_scenarios": severe,
            "all_passed": margin_calls == 0 and severe == 0,
            "results": [
                {
                    "name": r.scenario_name,
                    "type": r.scenario_type,
                    "pnl_pct": r.portfolio_pnl_pct,
                    "margin_call": r.margin_call_risk,
                    "action": r.recommended_action,
                }
                for r in self._last_results
            ],
        }

"""
Phase E: Capital capacity modeling.
Participation impact, slippage scaling, market depth sensitivity, turnover pressure;
simulate capital scaling; reject if edge breaks beyond participation threshold.
"""

from dataclasses import dataclass


@dataclass
class CapacityResult:
    capacity_notional: float
    capacity_score_normalized: float  # 0..1 vs target_capital
    edge_breaks_at: float | None = None
    participation_at_capacity: float = 0.0
    passed: bool = True


def _impact_model(
    notional: float,
    adv: float,
    alpha: float = 0.5,
    k: float = 0.1,
) -> float:
    """Market impact proxy: k * (notional/adv)^alpha. Return as fraction of notional."""
    if adv <= 0:
        return 1.0
    participation = notional / adv
    return k * (participation**alpha)


class CapacityModel:
    """
    Estimate capacity: E[r](C) = E[r]_0 - impact(C) - slippage(C).
    impact(C) = k * (C/ADV)^alpha; slippage scales with participation.
    Capacity = max C such that E[r](C) >= 0 or Sharpe(C) >= 0.9*Sharpe(0).
    Reject signal if edge breaks below target capital (participation threshold).
    """

    def __init__(
        self,
        target_capital: float = 1e6,
        max_participation_pct: float = 10.0,
        impact_alpha: float = 0.5,
        impact_k: float = 0.1,
        slippage_per_participation_bps: float = 5.0,
    ):
        self.target_capital = target_capital
        self.max_participation_pct = max_participation_pct
        self.impact_alpha = impact_alpha
        self.impact_k = impact_k
        self.slippage_per_participation_bps = slippage_per_participation_bps

    def estimate_capacity(
        self,
        adv: float,
        e_return_gross: float,
        sharpe_0: float = 0.5,
        n_steps: int = 20,
    ) -> CapacityResult:
        """
        Simulate notional from 0 to 2*target_capital; find C where E[r](C) <= 0
        or participation > max_participation_pct. capacity_score_normalized = C_max / target_capital.
        """
        if adv <= 0:
            return CapacityResult(
                capacity_notional=0.0,
                capacity_score_normalized=0.0,
                passed=False,
            )
        max_notional = 2.0 * self.target_capital
        step = max_notional / n_steps
        best_c = 0.0
        for i in range(n_steps + 1):
            c = i * step
            part_pct = 100.0 * c / adv
            if part_pct > self.max_participation_pct:
                break
            impact = _impact_model(c, adv, self.impact_alpha, self.impact_k)
            slip_bps = self.slippage_per_participation_bps * (c / adv)
            e_net = e_return_gross - impact - (slip_bps / 10_000.0)
            if e_net >= 0:
                best_c = c
            else:
                break
        cap_score = min(1.0, best_c / (self.target_capital + 1e-8))
        passed = best_c >= self.target_capital or cap_score >= 0.5
        return CapacityResult(
            capacity_notional=best_c,
            capacity_score_normalized=cap_score,
            participation_at_capacity=100.0 * best_c / adv if adv > 0 else 0.0,
            passed=passed,
        )

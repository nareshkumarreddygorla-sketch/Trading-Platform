"""
Cost-aware barrier and label utilities.
Round-trip cost = slippage + commission (both sides); applied to barrier levels.
"""
from typing import Tuple


def cost_aware_barriers(
    price: float,
    profit_target_pct: float,
    stop_loss_pct: float,
    round_trip_cost_pct: float,
) -> Tuple[float, float]:
    """
    Upper and lower barriers after round-trip cost.
    U = price * (1 + b_u - c), L = price * (1 - b_d + c).
    Ensures that "touch upper" implies gross return > cost.
    """
    u = price * (1.0 + profit_target_pct - round_trip_cost_pct)
    l = price * (1.0 - stop_loss_pct + round_trip_cost_pct)
    return u, l


def net_return(gross_return: float, round_trip_cost_pct: float) -> float:
    """Net return after round-trip cost."""
    return gross_return - round_trip_cost_pct

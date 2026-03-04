"""
Performance feedback loop: track per-strategy win rate, rolling Sharpe, drawdown,
consecutive losses. After each fill update tracker; auto-disable degrading strategy;
adjust exposure multiplier. Integrates via FillHandler callback. Does not modify fill safety logic.
Cost-adjusted: deducts India transaction costs from PnL before recording.
Sharpe is annualized (sqrt(252)) for daily trading.
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    wins: int = 0
    losses: int = 0
    pnls: list = field(default_factory=list)
    max_pnl_window: int = 100
    consecutive_losses: int = 0
    disabled: bool = False
    exposure_multiplier: float = 1.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.0

    @property
    def rolling_sharpe(self) -> float:
        """Annualized Sharpe ratio (sqrt(252) for daily trades)."""
        if len(self.pnls) < 2:
            return 0.0
        import numpy as np
        arr = np.array(self.pnls[-self.max_pnl_window :])
        mean_pnl = np.mean(arr)
        std_pnl = np.std(arr)
        if std_pnl < 1e-12:
            return 0.0
        daily_sharpe = float(mean_pnl / (std_pnl + 1e-12))
        return daily_sharpe * math.sqrt(252)  # annualize

    @property
    def rolling_drawdown_pct(self) -> float:
        if len(self.pnls) < 2:
            return 0.0
        import numpy as np
        cum = np.cumsum(self.pnls[-self.max_pnl_window :])
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum)
        return float(100.0 * np.max(dd) / (peak[-1] + 1e-12)) if peak[-1] > 0 else 0.0


class PerformanceTracker:
    """
    Track per-strategy stats. On fill: update tracker; optionally disable strategy
    or adjust exposure. Callback from FillHandler (on_fill_callback) without changing fill safety.
    """

    def __init__(
        self,
        max_consecutive_losses_disable: int = 5,
        min_win_rate_disable: float = 0.35,
        max_drawdown_pct_disable: float = 15.0,
        on_strategy_disabled: Optional[Callable[[str, str], None]] = None,
        on_exposure_multiplier_changed: Optional[Callable[[str, float], None]] = None,
        cost_calculator=None,
    ):
        self.max_consecutive_losses_disable = max_consecutive_losses_disable
        self.min_win_rate_disable = min_win_rate_disable
        self.max_drawdown_pct_disable = max_drawdown_pct_disable
        self.on_strategy_disabled = on_strategy_disabled
        self.on_exposure_multiplier_changed = on_exposure_multiplier_changed
        self._stats: Dict[str, StrategyStats] = defaultdict(StrategyStats)

        # India transaction cost calculator for cost-adjusted Sharpe
        if cost_calculator is None:
            try:
                from src.costs.india_costs import IndiaCostCalculator
                self._cost_calc = IndiaCostCalculator()
            except ImportError:
                self._cost_calc = None
        else:
            self._cost_calc = cost_calculator

    def get_stats(self, strategy_id: str) -> StrategyStats:
        return self._stats[strategy_id]

    def get_all_stats(self) -> Dict[str, StrategyStats]:
        """Return stats for all tracked strategies."""
        return dict(self._stats)

    def summary(self) -> dict:
        """Aggregate summary across all strategies."""
        total_wins = sum(s.wins for s in self._stats.values())
        total_losses = sum(s.losses for s in self._stats.values())
        total_trades = total_wins + total_losses
        all_pnls = []
        for s in self._stats.values():
            all_pnls.extend(s.pnls)
        total_pnl = sum(all_pnls)
        win_rate = total_wins / total_trades if total_trades else 0.0
        avg_pnl = total_pnl / total_trades if total_trades else 0.0

        import numpy as np
        sharpe = 0.0
        if len(all_pnls) >= 2:
            arr = np.array(all_pnls)
            std = np.std(arr)
            daily_sharpe = float(np.mean(arr) / (std + 1e-12))
            sharpe = daily_sharpe * math.sqrt(252)  # annualize

        max_dd = max((s.rolling_drawdown_pct for s in self._stats.values()), default=0.0)

        return {
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_trade_pnl": avg_pnl,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "strategies": {
                sid: {
                    "wins": s.wins, "losses": s.losses, "win_rate": s.win_rate,
                    "pnl": sum(s.pnls), "sharpe": s.rolling_sharpe,
                    "drawdown_pct": s.rolling_drawdown_pct, "disabled": s.disabled,
                }
                for sid, s in self._stats.items()
            },
        }

    def is_disabled(self, strategy_id: str) -> bool:
        return self._stats[strategy_id].disabled

    def record_fill(self, strategy_id: str, realized_pnl: float, notional: float = 0.0, side: str = "BUY") -> None:
        """
        Call after fill is applied (e.g. from FillHandler callback).
        If notional provided, deducts round-trip India transaction costs from PnL.
        """
        # Deduct round-trip transaction costs if cost calculator available
        cost_adjusted_pnl = realized_pnl
        if self._cost_calc and notional > 0:
            try:
                costs = self._cost_calc.round_trip_cost(
                    buy_notional=notional,
                    sell_notional=notional,
                    product_type="INTRADAY",
                    exchange="NSE",
                )
                cost_adjusted_pnl = realized_pnl - costs.total
            except Exception as e:
                logger.debug("Cost calculation failed: %s", e)
        s = self._stats[strategy_id]
        s.pnls.append(cost_adjusted_pnl)
        if len(s.pnls) > s.max_pnl_window:
            s.pnls.pop(0)
        if realized_pnl > 0:
            s.wins += 1
            s.consecutive_losses = 0
        else:
            s.losses += 1
            s.consecutive_losses += 1

        if not s.disabled:
            if s.consecutive_losses >= self.max_consecutive_losses_disable:
                s.disabled = True
                logger.warning("Strategy %s auto-disabled: consecutive_losses=%s", strategy_id, s.consecutive_losses)
                if self.on_strategy_disabled:
                    try:
                        self.on_strategy_disabled(strategy_id, "consecutive_losses")
                    except Exception as e:
                        logger.exception("on_strategy_disabled failed: %s", e)
            elif (s.wins + s.losses) >= 10 and s.win_rate < self.min_win_rate_disable:
                s.disabled = True
                logger.warning("Strategy %s auto-disabled: win_rate=%.2f", strategy_id, s.win_rate)
                if self.on_strategy_disabled:
                    try:
                        self.on_strategy_disabled(strategy_id, "low_win_rate")
                    except Exception as e:
                        logger.exception("on_strategy_disabled failed: %s", e)
            elif s.rolling_drawdown_pct >= self.max_drawdown_pct_disable:
                s.disabled = True
                logger.warning("Strategy %s auto-disabled: drawdown_pct=%.1f", strategy_id, s.rolling_drawdown_pct)
                if self.on_strategy_disabled:
                    try:
                        self.on_strategy_disabled(strategy_id, "max_drawdown")
                    except Exception as e:
                        logger.exception("on_strategy_disabled failed: %s", e)

        # Re-enable logic: if disabled but performance recovered, re-enable
        if s.disabled and (s.wins + s.losses) >= 20:
            recent_pnls = s.pnls[-20:]
            recent_wins = sum(1 for p in recent_pnls if p > 0)
            if recent_wins / len(recent_pnls) > 0.50 and s.consecutive_losses < 3:
                s.disabled = False
                logger.info("Strategy %s auto-RE-ENABLED: recent_win_rate=%.0f%%", strategy_id, 100 * recent_wins / len(recent_pnls))
                if self.on_strategy_disabled:
                    try:
                        self.on_strategy_disabled(strategy_id, "re_enabled")
                    except Exception:
                        pass

        # Exposure multiplier — ALWAYS clamped to [0.3, 1.0] (never INCREASE exposure when losing)
        new_mult = 1.0
        if s.rolling_sharpe < -0.5 and (s.wins + s.losses) >= 5:
            new_mult = 0.3  # aggressive scale-down for very negative Sharpe
        elif s.rolling_sharpe < 0 and (s.wins + s.losses) >= 5:
            new_mult = max(0.3, min(1.0, 1.0 + 0.1 * s.rolling_sharpe))
        if abs(new_mult - s.exposure_multiplier) > 0.05:
            s.exposure_multiplier = new_mult
            if self.on_exposure_multiplier_changed:
                try:
                    self.on_exposure_multiplier_changed(strategy_id, new_mult)
                except Exception as e:
                    logger.exception("on_exposure_multiplier_changed failed: %s", e)

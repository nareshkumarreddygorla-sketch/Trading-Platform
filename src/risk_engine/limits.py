"""Risk limits: position, trade, portfolio exposure, sector, VaR, per-symbol, consecutive loss."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.events import Position, Signal


@dataclass
class LimitCheckResult:
    """Result of a limit check."""
    allowed: bool
    reason: str = ""


@dataclass
class RiskLimits:
    """Configurable risk limits. All checks enforced in can_place_order."""
    max_position_pct: float = 5.0  # per position as % of equity
    max_single_trade_loss_pct: float = 1.0
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 10
    max_sector_concentration_pct: float = 25.0
    max_correlation_exposure: float = 0.7  # max sum of correlated position weights
    var_limit_pct: Optional[float] = 5.0  # VaR as % of portfolio (real parametric VaR)
    circuit_breaker_drawdown_pct: float = 5.0
    max_per_symbol_pct: float = 10.0  # max exposure to single symbol as % of equity
    max_consecutive_losses: int = 5  # auto-disable new orders after N consecutive losing trades

    # Sprint 2 additions
    cvar_limit_pct: float = 8.0  # CVaR (Expected Shortfall) limit as % of portfolio
    max_intraday_loss_pct: float = 1.0  # max drawdown in 60-min rolling window as % of equity
    max_per_position_loss_pct: float = 3.0  # per-position hard stop loss %

    def check_position_size(self, equity: float, position_value: float) -> LimitCheckResult:
        if equity <= 0:
            return LimitCheckResult(False, "zero equity")
        pct = 100.0 * abs(position_value) / equity
        if pct > self.max_position_pct:
            return LimitCheckResult(False, f"position {pct:.2f}% > max {self.max_position_pct}%")
        return LimitCheckResult(True)

    def check_open_positions(self, current_count: int) -> LimitCheckResult:
        if current_count >= self.max_open_positions:
            return LimitCheckResult(False, f"open positions {current_count} >= max {self.max_open_positions}")
        return LimitCheckResult(True)

    def check_daily_loss(self, equity: float, daily_pnl: float) -> LimitCheckResult:
        if equity <= 0:
            return LimitCheckResult(False, "zero_or_negative_equity")
        loss_pct = -100.0 * daily_pnl / equity if daily_pnl < 0 else 0
        if loss_pct > self.max_daily_loss_pct:
            return LimitCheckResult(False, f"daily loss {loss_pct:.2f}% > max {self.max_daily_loss_pct}%")
        return LimitCheckResult(True)

    def check_single_trade_loss(self, equity: float, trade_loss: float) -> LimitCheckResult:
        if equity <= 0 or trade_loss >= 0:
            return LimitCheckResult(True)
        pct = -100.0 * trade_loss / equity
        if pct > self.max_single_trade_loss_pct:
            return LimitCheckResult(False, f"single trade loss {pct:.2f}% > max {self.max_single_trade_loss_pct}%")
        return LimitCheckResult(True)

    def check_per_symbol_exposure(self, equity: float, symbol_notional: float) -> LimitCheckResult:
        if equity <= 0:
            return LimitCheckResult(False, "equity <= 0, cannot check exposure")
        pct = 100.0 * symbol_notional / equity
        if pct > self.max_per_symbol_pct:
            return LimitCheckResult(False, f"per_symbol {pct:.2f}% > max {self.max_per_symbol_pct}%")
        return LimitCheckResult(True)

    def check_sector_concentration(self, equity: float, sector_notional: float) -> LimitCheckResult:
        if equity <= 0:
            return LimitCheckResult(False, "equity <= 0, cannot check sector concentration")
        pct = 100.0 * sector_notional / equity
        if pct > self.max_sector_concentration_pct:
            return LimitCheckResult(False, f"sector {pct:.2f}% > max {self.max_sector_concentration_pct}%")
        return LimitCheckResult(True)

    def check_var(self, equity: float, total_position_notional: float) -> LimitCheckResult:
        """Simple VaR: total notional as % of equity. Real VaR would use volatility."""
        if self.var_limit_pct is None:
            return LimitCheckResult(True)
        if equity <= 0:
            return LimitCheckResult(False, "equity <= 0, cannot check VaR")
        pct = 100.0 * total_position_notional / equity
        if pct > self.var_limit_pct:
            return LimitCheckResult(False, f"var exposure {pct:.2f}% > limit {self.var_limit_pct}%")
        return LimitCheckResult(True)

    def check_consecutive_losses(self, consecutive_loss_count: int) -> LimitCheckResult:
        if consecutive_loss_count >= self.max_consecutive_losses:
            return LimitCheckResult(False, f"consecutive_losses {consecutive_loss_count} >= max {self.max_consecutive_losses}")
        return LimitCheckResult(True)

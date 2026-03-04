"""Risk manager: apply limits, block trades, maintain state (positions, daily PnL)."""
import asyncio
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from src.core.events import Position, Signal

from .limits import LimitCheckResult, RiskLimits
from .metrics import RiskMetrics, compute_risk_metrics
import numpy as np

logger = logging.getLogger(__name__)


def _default_sector(_: str) -> str:
    return "GENERIC"


class RiskManager:
    """
    Central risk gate: all orders must pass limit checks.
    Tracks positions, daily PnL, consecutive losses; triggers circuit breaker on breach.
    Optionally integrates PortfolioVaR, CorrelationGuard, and TailRiskProtector for
    institutional-grade risk checks beyond simple notional limits.
    """

    def __init__(
        self,
        equity: float,
        limits: Optional[RiskLimits] = None,
        symbol_sector_map: Optional[Dict[str, str]] = None,
        portfolio_var=None,
        correlation_guard=None,
        tail_risk_protector=None,
    ):
        self.equity = equity
        self.limits = limits or RiskLimits()
        self.positions: List[Position] = []
        self.daily_pnl: float = 0.0
        self._circuit_open: bool = False
        self._exposure_multiplier: float = 1.0  # 0.5--1.5 from vol or LLM
        self._consecutive_losses: int = 0
        self._symbol_sector_map = symbol_sector_map or {}
        # Institutional risk modules (optional — gracefully skip if None)
        self._portfolio_var = portfolio_var
        self._correlation_guard = correlation_guard
        self._tail_risk_protector = tail_risk_protector

        # Thread-safety: lock for position mutations (asyncio.Lock for coroutine callers)
        self._positions_lock = asyncio.Lock()

        # Intraday rolling loss tracking (60-minute window of P&L snapshots)
        self._intraday_pnl_window: deque = deque(maxlen=60)  # (timestamp, pnl_snapshot)

        # Per-position hard stop loss tracking
        self._forced_close_symbols: List[str] = []

    def get_forced_close_symbols(self) -> List[str]:
        """Return symbols flagged for forced close by per-position loss check, then clear the list."""
        symbols = list(self._forced_close_symbols)
        self._forced_close_symbols.clear()
        return symbols

    def _sector(self, symbol: str) -> str:
        return self._symbol_sector_map.get(symbol, "GENERIC")

    def update_equity(self, equity: float) -> None:
        self.equity = max(0.0, float(equity))

    def register_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L at start of new trading day."""
        self.daily_pnl = 0.0
        self._consecutive_losses = 0

    def add_position(self, pos: Position) -> None:
        self.positions.append(pos)

    def add_or_merge_position(self, pos: Position) -> None:
        """
        Add fill to position: merge with existing (symbol, exchange, side) by summing quantity and VWAP.
        Call under same lock as can_place_order to avoid race with order entry.
        """
        for i, p in enumerate(self.positions):
            if (
                p.symbol == pos.symbol
                and getattr(p.exchange, "value", p.exchange) == getattr(pos.exchange, "value", pos.exchange)
                and getattr(p.side, "value", p.side) == getattr(pos.side, "value", pos.side)
            ):
                total_qty = p.quantity + pos.quantity
                if total_qty <= 0:
                    self.positions.pop(i)
                    return
                # VWAP: use absolute quantities to avoid sign issues with short positions
                abs_existing = abs(p.quantity)
                abs_new = abs(pos.quantity)
                total_abs = abs_existing + abs_new
                new_avg = (abs_existing * p.avg_price + abs_new * pos.avg_price) / total_abs if total_abs > 0 else p.avg_price
                self.positions[i] = Position(
                    symbol=p.symbol,
                    exchange=p.exchange,
                    side=p.side,
                    quantity=total_qty,
                    avg_price=new_avg,
                    unrealized_pnl=getattr(p, "unrealized_pnl", 0.0),
                    strategy_id=pos.strategy_id or p.strategy_id,
                )
                return
        self.positions.append(pos)

    def load_positions_for_recovery(self, positions: List[Position]) -> None:
        """
        Cold start only: replace internal positions with persisted state.
        Must not be used during normal trading. Call only at startup before OrderEntryService is used.
        """
        self.positions = list(positions)
        logger.info("RiskManager warmed with %d positions from persistence", len(self.positions))

    def remove_position(self, symbol: str, exchange: str, side: Optional[str] = None) -> None:
        """Remove position(s) for symbol+exchange, optionally filtered by side (BUY/SELL)."""
        exch_str = exchange if isinstance(exchange, str) else getattr(exchange, "value", "")
        side_val = side if isinstance(side, str) else getattr(side, "value", None)
        if side_val is not None:
            self.positions = [
                p for p in self.positions
                if not (p.symbol == symbol and getattr(p.exchange, "value", "") == exch_str and getattr(p.side, "value", "") == side_val)
            ]
        else:
            self.positions = [p for p in self.positions if not (p.symbol == symbol and getattr(p.exchange, "value", "") == exch_str)]

    def open_circuit(self) -> None:
        self._circuit_open = True
        logger.warning("Risk circuit breaker OPEN: new orders blocked")

    def close_circuit(self) -> None:
        self._circuit_open = False

    def is_circuit_open(self) -> bool:
        return self._circuit_open

    def set_exposure_multiplier(self, mult: float) -> None:
        """Set exposure multiplier from LLM or volatility (e.g. 0.5--1.5). Caps to [0.5, 1.5]."""
        self._exposure_multiplier = max(0.5, min(1.5, float(mult)))

    def set_volatility_scaling(self, current_vol: float, reference_vol: float = 15.0) -> None:
        """Scale exposure by vol: higher vol -> lower multiplier. reference_vol typical (e.g. 15%)."""
        if current_vol <= 0:
            self._exposure_multiplier = 1.0
            return
        mult = reference_vol / max(current_vol, 1e-6)
        self._exposure_multiplier = max(0.5, min(1.5, float(mult)))

    def effective_equity(self) -> float:
        """Equity scaled by exposure multiplier for position sizing."""
        return self.equity * self._exposure_multiplier

    def check_drawdown(self, peak_equity: float, current_equity: float) -> bool:
        """Return True if drawdown exceeds limit (should open circuit)."""
        if peak_equity <= 0:
            return False
        dd = (peak_equity - current_equity) / peak_equity * 100
        return dd >= self.limits.circuit_breaker_drawdown_pct

    def _position_notional_by_symbol(self) -> Dict[str, float]:
        """Symbol -> total notional (quantity * avg_price) across positions."""
        out: Dict[str, float] = {}
        for p in self.positions:
            notional = p.quantity * (p.avg_price or 0)
            out[p.symbol] = out.get(p.symbol, 0) + notional
        return out

    def _position_notional_by_sector(self) -> Dict[str, float]:
        """Sector -> total notional."""
        out: Dict[str, float] = {}
        for p in self.positions:
            sec = self._sector(p.symbol)
            notional = p.quantity * (p.avg_price or 0)
            out[sec] = out.get(sec, 0) + notional
        return out

    def can_place_order(
        self,
        signal: Signal,
        quantity: int,
        price: float,
        is_reducing: bool = False,
        force_reduce: bool = False,
    ) -> LimitCheckResult:
        """
        Check if order is allowed under current limits.
        is_reducing: True when the order would reduce existing exposure (exit order).
        force_reduce: True for emergency close-outs (bypasses daily loss, circuit, rate limit).
        """
        if quantity <= 0 or price <= 0:
            return LimitCheckResult(False, "invalid_quantity_or_price")
        # Force-reduce bypasses circuit breaker for emergency close-outs
        if self._circuit_open and not force_reduce:
            return LimitCheckResult(False, "circuit_breaker_open")
        if self.equity <= 0:
            return LimitCheckResult(False, "zero_or_negative_equity")

        # Skip daily loss and consecutive loss checks for reducing/exit orders
        if not is_reducing and not force_reduce:
            r = self.limits.check_consecutive_losses(self._consecutive_losses)
            if not r.allowed:
                return r
        eff_equity = self.effective_equity()
        if not is_reducing and not force_reduce:
            r = self.limits.check_daily_loss(self.equity, self.daily_pnl)
            if not r.allowed:
                return r

            # Intraday rolling loss check (Sprint 2.3)
            if self._intraday_pnl_window and self.limits.max_intraday_loss_pct > 0:
                now = time.time()
                window_start = now - 3600  # 60 minutes
                recent = [(ts, pnl) for ts, pnl in self._intraday_pnl_window if ts >= window_start]
                if recent:
                    max_pnl = max(pnl for _, pnl in recent)
                    current_pnl = recent[-1][1] if recent else 0.0
                    drawdown_in_window = max_pnl - current_pnl
                    if self.equity > 0 and (drawdown_in_window / self.equity) * 100 > self.limits.max_intraday_loss_pct:
                        return LimitCheckResult(False, f"intraday_rolling_loss: {drawdown_in_window:.0f} in 60min")
        r = self.limits.check_open_positions(len(self.positions))
        if not r.allowed:
            logger.warning("Order rejected: %s (symbol=%s qty=%s price=%s)", r.reason, signal.symbol, quantity, price)
            return r
        position_value = quantity * price
        r = self.limits.check_position_size(eff_equity, position_value)
        if not r.allowed:
            logger.warning("Order rejected: %s (symbol=%s qty=%s price=%s)", r.reason, signal.symbol, quantity, price)
            return r
        by_sym = self._position_notional_by_symbol()
        new_symbol_notional = by_sym.get(signal.symbol, 0) + position_value
        r = self.limits.check_per_symbol_exposure(self.equity, new_symbol_notional)
        if not r.allowed:
            return r
        by_sec = self._position_notional_by_sector()
        order_sector = self._sector(signal.symbol)
        new_sector_notional = by_sec.get(order_sector, 0) + position_value
        r = self.limits.check_sector_concentration(self.equity, new_sector_notional)
        if not r.allowed:
            return r
        total_notional = sum(by_sym.values()) + position_value
        r = self.limits.check_var(self.equity, total_notional)
        if not r.allowed:
            return r

        # ── Institutional risk checks (optional — skip if not wired) ──

        # Tail risk: block if VIX extreme, rapid drawdown, or circuit tripped this session
        if self._tail_risk_protector is not None:
            try:
                blocked, reason = self._tail_risk_protector.should_block_new_positions()
                if blocked:
                    return LimitCheckResult(False, f"tail_risk: {reason}")
            except Exception as e:
                logger.error("Tail risk check failed (BLOCKING order — fail-safe): %s", e)
                return LimitCheckResult(False, f"tail_risk_check_error: {e}")

        # Correlation guard: reject if new position too correlated with existing
        if self._correlation_guard is not None:
            try:
                existing_syms = [p.symbol for p in self.positions]
                existing_notionals = [p.quantity * (p.avg_price or 0) for p in self.positions]
                corr_result = self._correlation_guard.check_new_position(
                    new_symbol=signal.symbol,
                    existing_symbols=existing_syms,
                    existing_notionals=existing_notionals,
                    new_notional=position_value,
                    portfolio_value=self.equity,
                )
                if not corr_result.allowed:
                    return LimitCheckResult(False, f"correlation: {corr_result.reason}")
            except Exception as e:
                logger.error("Correlation check failed (BLOCKING order — fail-safe): %s", e)
                return LimitCheckResult(False, f"correlation_check_error: {e}")

        # Portfolio VaR: reject if adding position would breach VaR limit
        if self._portfolio_var is not None and self.limits.var_limit_pct:
            try:
                pos_dicts = [{"symbol": p.symbol, "notional": p.quantity * (p.avg_price or 0)} for p in self.positions]
                allowed, var_pct = self._portfolio_var.check_var_limit(
                    pos_dicts, self.equity, self.limits.var_limit_pct,
                )
                if not allowed:
                    return LimitCheckResult(False, f"portfolio_var: VaR {var_pct:.2f}% > limit {self.limits.var_limit_pct}%")
            except Exception as e:
                logger.error("Portfolio VaR check failed (BLOCKING order — fail-safe): %s", e)
                return LimitCheckResult(False, f"var_check_error: {e}")

        # CVaR (Expected Shortfall) check — fail-CLOSED
        if self._portfolio_var is not None and self.limits.cvar_limit_pct:
            try:
                cvar_allowed, cvar_pct = self._portfolio_var.check_cvar_limit(
                    pos_dicts, self.equity, self.limits.cvar_limit_pct
                )
                if not cvar_allowed:
                    return LimitCheckResult(False, f"portfolio_cvar: CVaR {cvar_pct:.2f}% > limit {self.limits.cvar_limit_pct}%")
            except Exception as e:
                logger.error("CVaR check failed (BLOCKING order — fail-safe): %s", e)
                return LimitCheckResult(False, f"cvar_check_error: {e}")

        return LimitCheckResult(True)

    def max_quantity_for_signal(self, price: float) -> int:
        """Max quantity allowed by position size limit (uses effective equity from exposure multiplier)."""
        if price <= 0:
            return 0
        max_val = self.effective_equity() * (self.limits.max_position_pct / 100.0)
        return int(max_val / price)

    def record_intraday_snapshot(self) -> None:
        """Record current P&L snapshot for intraday rolling loss tracking. Call every 60s."""
        self._intraday_pnl_window.append((time.time(), self.daily_pnl))

    def check_position_loss(self, current_prices: Dict[str, float]) -> List[Tuple[str, float, float]]:
        """
        Check per-position unrealized loss against hard stop.
        Returns list of (symbol, loss_pct, position_notional) for positions exceeding limit.
        """
        flagged: List[Tuple[str, float, float]] = []
        max_loss_pct = self.limits.max_per_position_loss_pct
        if max_loss_pct <= 0:
            return flagged
        for p in self.positions:
            current = current_prices.get(p.symbol, 0.0)
            if current <= 0 or (p.avg_price or 0) <= 0:
                continue
            notional = p.quantity * p.avg_price
            side_val = getattr(p.side, "value", str(p.side))
            if side_val == "BUY":
                pnl = (current - p.avg_price) / p.avg_price * 100
            else:
                pnl = (p.avg_price - current) / p.avg_price * 100
            if pnl < -max_loss_pct:
                flagged.append((p.symbol, pnl, notional))
                logger.warning("Position hard stop: %s loss=%.2f%% > limit=%.1f%%", p.symbol, pnl, max_loss_pct)
        self._forced_close_symbols = [s for s, _, _ in flagged]
        return flagged

    @property
    def positions_lock(self) -> asyncio.Lock:
        """Expose lock for external callers (FillHandler, OrderEntryService)."""
        return self._positions_lock

    def portfolio_risk_metrics(self, equity_curve: List[float]) -> RiskMetrics:
        """Compute risk metrics from equity curve (e.g. for reporting)."""
        if len(equity_curve) < 2:
            return RiskMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
        arr = np.array(equity_curve)
        returns = np.diff(arr) / (arr[:-1] + 1e-12)
        return compute_risk_metrics(returns)

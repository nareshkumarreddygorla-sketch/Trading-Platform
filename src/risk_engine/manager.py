"""Risk manager: apply limits, block trades, maintain state (positions, daily PnL)."""
import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.events import Position, Signal

from .limits import LimitCheckResult, RiskLimits
from .metrics import RiskMetrics, compute_risk_metrics
import numpy as np

logger = logging.getLogger(__name__)

_CIRCUIT_STATE_DIR = Path(__file__).resolve().parents[2] / "data"
_CIRCUIT_STATE_PATH = _CIRCUIT_STATE_DIR / "circuit_state.json"

# Default sector cap (no more than 30% of portfolio in any single sector)
_DEFAULT_SECTOR_CAP_PCT = 30.0


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
        load_persisted_state: bool = True,
        sector_classifier=None,
        sector_cap_pct: float = _DEFAULT_SECTOR_CAP_PCT,
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

        # Sector concentration monitoring via SectorClassifier
        self._sector_classifier = sector_classifier
        self._sector_cap_pct = sector_cap_pct

        # Thread-safety: lock for position mutations
        # Uses threading.Lock for synchronous contexts (FillHandler, etc.)
        # Async callers should use async_positions_lock for proper coordination
        self._positions_lock = threading.Lock()
        self._async_positions_lock: Optional[object] = None  # Lazily initialized asyncio.Lock

        # Restore persisted circuit breaker state on startup
        if load_persisted_state:
            self._load_circuit_state()

        # Intraday rolling loss tracking (60-minute window of P&L snapshots)
        self._intraday_pnl_window: deque = deque(maxlen=60)  # (timestamp, pnl_snapshot)

        # Per-position hard stop loss tracking
        self._forced_close_symbols: List[str] = []

    def get_forced_close_symbols(self) -> List[str]:
        """Return symbols flagged for forced close by per-position loss check, then clear the list."""
        with self._positions_lock:
            symbols = list(self._forced_close_symbols)
            self._forced_close_symbols.clear()
            return symbols

    def _sector(self, symbol: str) -> str:
        """Get sector using SectorClassifier (priority) or legacy symbol_sector_map."""
        if self._sector_classifier is not None:
            try:
                sec = self._sector_classifier.get_sector(symbol)
                if sec and sec != "UNCLASSIFIED":
                    return sec
            except Exception:
                pass
        return self._symbol_sector_map.get(symbol, "GENERIC")

    def update_equity(self, equity: float) -> None:
        with self._positions_lock:
            equity_val = float(equity)
            if equity_val < 0:
                logger.critical(
                    "NEGATIVE EQUITY DETECTED: %.2f — opening circuit breaker immediately. "
                    "Manual review required before resuming trading.",
                    equity_val,
                )
                self.equity = equity_val
                self._circuit_open = True
                self._save_circuit_state("negative_equity")
            else:
                self.equity = equity_val

    def register_pnl(self, pnl: float) -> None:
        with self._positions_lock:
            self.daily_pnl += pnl
            if pnl < 0:
                self._consecutive_losses += 1
            elif pnl > 0:
                self._consecutive_losses = 0
        # Persist daily PnL to survive crashes
        self._save_circuit_state("pnl_update")

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L at start of new trading day."""
        with self._positions_lock:
            self.daily_pnl = 0.0
            self._consecutive_losses = 0

    def add_position(self, pos: Position) -> None:
        with self._positions_lock:
            self.positions.append(pos)

    def add_or_merge_position(self, pos: Position) -> None:
        """
        Add fill to position: merge with existing (symbol, exchange, side) by summing quantity and VWAP.
        Acquires _positions_lock to avoid race with order entry.
        """
        with self._positions_lock:
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
        with self._positions_lock:
            self.positions = list(positions)
        logger.info("RiskManager warmed with %d positions from persistence", len(self.positions))

    def remove_position(self, symbol: str, exchange: str, side: Optional[str] = None) -> None:
        """Remove position(s) for symbol+exchange, optionally filtered by side (BUY/SELL)."""
        with self._positions_lock:
            exch_str = exchange if isinstance(exchange, str) else getattr(exchange, "value", "")
            side_val = side if isinstance(side, str) else getattr(side, "value", None)
            if side_val is not None:
                self.positions = [
                    p for p in self.positions
                    if not (p.symbol == symbol and getattr(p.exchange, "value", "") == exch_str and getattr(p.side, "value", "") == side_val)
                ]
            else:
                self.positions = [p for p in self.positions if not (p.symbol == symbol and getattr(p.exchange, "value", "") == exch_str)]

    def open_circuit(self, reason: str = "manual") -> None:
        self._circuit_open = True
        logger.warning("Risk circuit breaker OPEN: new orders blocked (reason=%s)", reason)
        self._save_circuit_state(reason)

    def close_circuit(self) -> None:
        self._circuit_open = False
        logger.info("Risk circuit breaker CLOSED: trading resumed")
        self._save_circuit_state("closed")

    def is_circuit_open(self) -> bool:
        return self._circuit_open

    # ── Circuit breaker state persistence ──

    def _save_circuit_state(self, reason: str = "") -> None:
        """Persist circuit breaker state and daily PnL to JSON using atomic write."""
        state = {
            "circuit_open": self._circuit_open,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "equity": self.equity,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "exposure_multiplier": round(self._exposure_multiplier, 4),
        }
        try:
            _CIRCUIT_STATE_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(_CIRCUIT_STATE_DIR), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, str(_CIRCUIT_STATE_PATH))
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.info("Circuit breaker state persisted: open=%s reason=%s", self._circuit_open, reason)
        except Exception:
            logger.exception("Failed to persist circuit breaker state")

    def _load_circuit_state(self) -> None:
        """Load persisted circuit breaker state on startup."""
        if not _CIRCUIT_STATE_PATH.exists():
            return
        try:
            with open(_CIRCUIT_STATE_PATH) as f:
                state = json.load(f)
            if state.get("circuit_open", False):
                self._circuit_open = True
                logger.critical(
                    "Circuit breaker was OPEN at last shutdown — keeping OPEN. "
                    "Reason: %s | Last updated: %s | Equity: %s | Daily PnL: %s. "
                    "Manually call close_circuit() after review.",
                    state.get("reason", "unknown"),
                    state.get("updated_at", "unknown"),
                    state.get("equity", "unknown"),
                    state.get("daily_pnl", "unknown"),
                )
            else:
                logger.info("Circuit breaker state loaded: closed (normal)")
            # Restore exposure multiplier (survives crashes)
            saved_mult = state.get("exposure_multiplier")
            if saved_mult is not None:
                self._exposure_multiplier = max(0.5, min(1.5, float(saved_mult)))
                logger.info("Restored exposure_multiplier=%.3f from persisted state", self._exposure_multiplier)
            # Restore daily PnL and consecutive losses (survive crashes)
            saved_pnl = state.get("daily_pnl")
            if saved_pnl is not None:
                self.daily_pnl = float(saved_pnl)
                logger.info("Restored daily_pnl=%.2f from persisted state", self.daily_pnl)
            saved_losses = state.get("consecutive_losses")
            if saved_losses is not None:
                self._consecutive_losses = int(saved_losses)
                logger.info("Restored consecutive_losses=%d from persisted state", self._consecutive_losses)
        except Exception:
            logger.exception(
                "Failed to load circuit breaker state — defaulting to OPEN for safety"
            )
            self._circuit_open = True

    def auto_close_circuit_if_stale(self, max_open_hours: float = 2.0) -> bool:
        """Auto-close circuit breaker if it's been open longer than max_open_hours.
        Returns True if circuit was auto-closed. Prevents indefinite trading halt
        when ops doesn't notice a manually armed circuit breaker."""
        if not self._circuit_open:
            return False
        try:
            if _CIRCUIT_STATE_PATH.exists():
                with open(_CIRCUIT_STATE_PATH) as f:
                    state = json.load(f)
                updated = state.get("updated_at")
                reason = state.get("reason", "")
                if updated and reason != "manual":
                    from datetime import datetime as _dt
                    open_time = _dt.fromisoformat(updated)
                    now = _dt.now(timezone.utc)
                    if open_time.tzinfo is None:
                        open_time = open_time.replace(tzinfo=timezone.utc)
                    hours_open = (now - open_time).total_seconds() / 3600
                    if hours_open >= max_open_hours:
                        self.close_circuit()
                        logger.warning(
                            "Circuit breaker AUTO-CLOSED after %.1f hours (reason was: %s). "
                            "Review trading logs before resuming.",
                            hours_open, reason,
                        )
                        return True
        except Exception as e:
            logger.error("auto_close_circuit_if_stale failed: %s", e)
        return False

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
        with self._positions_lock:
            for p in list(self.positions):
                notional = p.quantity * (p.avg_price or 0)
                out[p.symbol] = out.get(p.symbol, 0) + notional
        return out

    def _position_notional_by_sector(self) -> Dict[str, float]:
        """Sector -> total notional."""
        out: Dict[str, float] = {}
        with self._positions_lock:
            for p in list(self.positions):
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
        # Validate force_reduce actually reduces exposure
        if force_reduce and not is_reducing:
            logger.error(
                "force_reduce=True but is_reducing=False — rejecting order for %s. "
                "force_reduce may only be used for position-reducing orders.",
                signal.symbol,
            )
            return LimitCheckResult(False, "force_reduce_requires_is_reducing")
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
        with self._positions_lock:
            positions_snapshot = list(self.positions)
        r = self.limits.check_open_positions(len(positions_snapshot))
        if not r.allowed:
            logger.warning("Order rejected: %s (symbol=%s qty=%s price=%s)", r.reason, signal.symbol, quantity, price)
            return r
        position_value = quantity * price
        r = self.limits.check_position_size(eff_equity, position_value)
        if not r.allowed:
            logger.warning("Order rejected: %s (symbol=%s qty=%s price=%s)", r.reason, signal.symbol, quantity, price)
            return r

        # Single trade loss check: estimate max potential loss for this trade
        # and reject if it would exceed the per-trade loss limit
        if not is_reducing and not force_reduce:
            max_potential_loss = -(position_value * self.limits.max_single_trade_loss_pct / 100.0)
            r = self.limits.check_single_trade_loss(self.equity, max_potential_loss)
            if not r.allowed:
                logger.warning(
                    "Order rejected: %s (symbol=%s qty=%s price=%s potential_loss=%.2f)",
                    r.reason, signal.symbol, quantity, price, max_potential_loss,
                )
                return r

        # Compute notional maps from the snapshot to avoid re-acquiring _positions_lock
        by_sym: Dict[str, float] = {}
        by_sec: Dict[str, float] = {}
        for p in positions_snapshot:
            notional = p.quantity * (p.avg_price or 0)
            by_sym[p.symbol] = by_sym.get(p.symbol, 0) + notional
            sec = self._sector(p.symbol)
            by_sec[sec] = by_sec.get(sec, 0) + notional
        new_symbol_notional = by_sym.get(signal.symbol, 0) + position_value
        r = self.limits.check_per_symbol_exposure(self.equity, new_symbol_notional)
        if not r.allowed:
            return r
        order_sector = self._sector(signal.symbol)
        new_sector_notional = by_sec.get(order_sector, 0) + position_value
        r = self.limits.check_sector_concentration(self.equity, new_sector_notional)
        if not r.allowed:
            return r

        # ── Configurable sector cap (via SectorClassifier) ──
        if self._sector_classifier is not None:
            r = self.check_sector_cap_for_order(signal.symbol, position_value)
            if not r.allowed:
                logger.warning(
                    "Order rejected: %s (symbol=%s qty=%s price=%s)",
                    r.reason, signal.symbol, quantity, price,
                )
                return r

        total_notional = sum(by_sym.values()) + position_value
        r = self.limits.check_leverage(self.equity, total_notional)
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
                existing_syms = [p.symbol for p in positions_snapshot]
                existing_notionals = [p.quantity * (p.avg_price or 0) for p in positions_snapshot]
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

        # Build pos_dicts once for both VaR and CVaR checks (BUG 19 fix)
        pos_dicts = [{"symbol": p.symbol, "notional": p.quantity * (p.avg_price or 0)} for p in positions_snapshot]

        # Portfolio VaR: reject if adding position would breach VaR limit
        if self._portfolio_var is not None and self.limits.var_limit_pct:
            try:
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

    def check_overnight_risk(self, max_overnight_pct: float = 3.0) -> List[Tuple[str, float]]:
        """P2-3: Flag positions exceeding max_overnight_pct of equity for reduction before market close.
        Call at 15:00 IST. Returns list of (symbol, pct_of_equity) that should be reduced."""
        flagged: List[Tuple[str, float]] = []
        if self.equity <= 0:
            return flagged
        with self._positions_lock:
            for p in list(self.positions):
                notional = abs(p.quantity * (p.avg_price or 0))
                pct = (notional / self.equity) * 100.0
                if pct > max_overnight_pct:
                    flagged.append((p.symbol, round(pct, 2)))
        if flagged:
            logger.warning(
                "Overnight risk: %d positions exceed %.1f%% of equity: %s",
                len(flagged), max_overnight_pct, flagged[:5],
            )
        return flagged

    def record_intraday_snapshot(self) -> None:
        """Record current P&L snapshot for intraday rolling loss tracking. Call every 60s."""
        with self._positions_lock:
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
        with self._positions_lock:
            positions_copy = list(self.positions)
        for p in positions_copy:
            current = current_prices.get(p.symbol, 0.0)
            if current <= 0 or (p.avg_price or 0) <= 0:
                continue
            notional = float(p.quantity * p.avg_price)
            side_val = getattr(p.side, "value", str(p.side))
            if side_val == "BUY":
                pnl = float((current - p.avg_price) / p.avg_price * 100)
            else:
                pnl = float((p.avg_price - current) / p.avg_price * 100)
            if pnl < -max_loss_pct:
                flagged.append((p.symbol, pnl, notional))
                logger.warning("Position hard stop: %s loss=%.2f%% > limit=%.1f%%", p.symbol, pnl, max_loss_pct)
        with self._positions_lock:
            self._forced_close_symbols = [s for s, _, _ in flagged]
        return flagged

    @property
    def positions_lock(self) -> threading.Lock:
        """Expose lock for external callers (FillHandler, OrderEntryService)."""
        return self._positions_lock

    async def acquire_async_lock(self):
        """Acquire async-safe lock for position mutations in async contexts.
        This ensures proper coordination between async OrderEntryService and
        sync FillHandler by using a dual-lock protocol."""
        import asyncio as _asyncio
        if self._async_positions_lock is None:
            self._async_positions_lock = _asyncio.Lock()
        await self._async_positions_lock.acquire()
        # Use run_in_executor to avoid blocking the event loop
        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, self._positions_lock.acquire)

    def release_async_lock(self):
        """Release both async and threading locks."""
        try:
            self._positions_lock.release()
        except RuntimeError:
            pass
        if self._async_positions_lock is not None:
            try:
                self._async_positions_lock.release()
            except RuntimeError:
                pass

    def get_sector_exposures(self) -> Dict[str, dict]:
        """
        Compute real-time sector exposure breakdown from current positions.

        Returns:
            {sector: {notional, count, symbols, pct_of_equity, pct_of_portfolio}}
        """
        with self._positions_lock:
            positions_copy = list(self.positions)
        sectors: Dict[str, dict] = {}
        total_notional = 0.0
        for p in positions_copy:
            sector = self._sector(p.symbol)
            notional = abs(p.quantity * (p.avg_price or 0))
            total_notional += notional
            if sector not in sectors:
                sectors[sector] = {"notional": 0.0, "count": 0, "symbols": [], "pct_of_equity": 0.0, "pct_of_portfolio": 0.0}
            sectors[sector]["notional"] += notional
            sectors[sector]["count"] += 1
            sectors[sector]["symbols"].append(p.symbol)

        for sec_data in sectors.values():
            if self.equity > 0:
                sec_data["pct_of_equity"] = round((sec_data["notional"] / self.equity) * 100, 2)
            if total_notional > 0:
                sec_data["pct_of_portfolio"] = round((sec_data["notional"] / total_notional) * 100, 2)
            sec_data["notional"] = round(sec_data["notional"], 2)

        return sectors

    def check_sector_cap_for_order(
        self,
        symbol: str,
        order_notional: float,
    ) -> LimitCheckResult:
        """
        Check if an order would breach the configurable sector cap.
        Uses SectorClassifier for accurate classification.

        Returns LimitCheckResult.
        """
        if self._sector_classifier is None or self.equity <= 0:
            return LimitCheckResult(True)

        sector = self._sector(symbol)
        with self._positions_lock:
            positions_copy = list(self.positions)

        sector_notional = order_notional
        for p in positions_copy:
            if self._sector(p.symbol) == sector:
                sector_notional += abs(p.quantity * (p.avg_price or 0))

        sector_pct = (sector_notional / self.equity) * 100.0
        if sector_pct > self._sector_cap_pct:
            return LimitCheckResult(
                False,
                f"sector_cap: {sector} at {sector_pct:.1f}% > cap {self._sector_cap_pct:.1f}%",
            )
        return LimitCheckResult(True)

    def risk_snapshot(self) -> dict:
        """
        Generate a comprehensive risk snapshot for logging/monitoring.
        Includes sector exposures, VaR, circuit state, drawdown metrics.
        Thread-safe: acquires positions lock for all position reads.
        """
        with self._positions_lock:
            n_positions = len(self.positions)
            positions_copy = list(self.positions)
            daily_pnl = self.daily_pnl
            consecutive_losses = self._consecutive_losses

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": round(self.equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "circuit_open": self._circuit_open,
            "exposure_multiplier": round(self._exposure_multiplier, 3),
            "consecutive_losses": consecutive_losses,
            "n_positions": n_positions,
        }

        # Sector exposures
        snapshot["sector_exposures"] = self.get_sector_exposures()

        # Log sector exposures
        for sector, data in snapshot["sector_exposures"].items():
            if data["pct_of_equity"] > self._sector_cap_pct * 0.8:
                logger.warning(
                    "Sector exposure warning: %s at %.1f%% of equity (cap=%.1f%%)",
                    sector, data["pct_of_equity"], self._sector_cap_pct,
                )

        # VaR metrics if available
        if self._portfolio_var is not None:
            try:
                pos_dicts = [
                    {"symbol": p.symbol, "notional": p.quantity * (p.avg_price or 0)}
                    for p in positions_copy
                ]
                var_result = self._portfolio_var.compute(pos_dicts, self.equity)
                snapshot["var"] = var_result.as_dict()
            except Exception as e:
                snapshot["var_error"] = str(e)

        # Tail risk state if available
        if self._tail_risk_protector is not None:
            try:
                snapshot["tail_risk"] = self._tail_risk_protector.state.as_dict()
            except Exception:
                pass

        return snapshot

    def portfolio_risk_metrics(self, equity_curve: List[float]) -> RiskMetrics:
        """Compute risk metrics from equity curve (e.g. for reporting)."""
        if len(equity_curve) < 2:
            return RiskMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
        arr = np.array(equity_curve, dtype=float)
        if not np.all(np.isfinite(arr)):
            logger.warning("Non-finite values in equity curve, filtering %d values", np.sum(~np.isfinite(arr)))
            arr = arr[np.isfinite(arr)]
            if len(arr) < 2:
                return RiskMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
        returns = np.diff(arr) / (arr[:-1] + 1e-12)
        return compute_risk_metrics(returns)

"""
Autonomous execution loop: bar-based cycle, stable idempotency key, drift/regime gating.
Closed loop: pull bars → strategy runner → allocator → risk → OrderEntryService only.
No direct gateway calls. Idempotency key: {bar_ts}-{strategy_id}-{symbol}-{side}.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from src.core.events import Bar, Exchange, Position, Signal, SignalSide
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.request import OrderEntryRequest, OrderEntryResult
from src.core.events import OrderType

logger = logging.getLogger(__name__)

# NSE market hours: 9:15 AM - 3:30 PM IST (UTC+5:30)
_IST = timezone(timedelta(hours=5, minutes=30))
_NSE_OPEN_HOUR, _NSE_OPEN_MIN = 9, 15
_NSE_CLOSE_HOUR, _NSE_CLOSE_MIN = 15, 30

# NSE exchange holidays for 2025-2026 (source: NSE circulars)
# Format: (month, day) — checked against current year
_NSE_HOLIDAYS = {
    # 2025
    (2025, 2, 26),   # Mahashivratri
    (2025, 3, 14),   # Holi
    (2025, 3, 31),   # Id-Ul-Fitr (Ramadan)
    (2025, 4, 10),   # Mahavir Jayanti
    (2025, 4, 14),   # Dr. Ambedkar Jayanti
    (2025, 4, 18),   # Good Friday
    (2025, 5, 1),    # Maharashtra Day
    (2025, 6, 7),    # Bakri Id
    (2025, 7, 6),    # Muharram
    (2025, 8, 15),   # Independence Day
    (2025, 8, 16),   # Janmashtami
    (2025, 10, 2),   # Mahatma Gandhi Jayanti
    (2025, 10, 20),  # Diwali (Laxmi Pujan)
    (2025, 10, 21),  # Diwali Balipratipada
    (2025, 11, 5),   # Guru Nanak Jayanti
    (2025, 12, 25),  # Christmas
    # 2026 (provisional — update from NSE circular when published)
    (2026, 1, 26),   # Republic Day
    (2026, 2, 17),   # Mahashivratri
    (2026, 3, 3),    # Holi
    (2026, 3, 20),   # Id-Ul-Fitr
    (2026, 3, 30),   # Ram Navami
    (2026, 4, 3),    # Good Friday
    (2026, 4, 14),   # Dr. Ambedkar Jayanti
    (2026, 5, 1),    # Maharashtra Day
    (2026, 5, 28),   # Bakri Id
    (2026, 8, 15),   # Independence Day
    (2026, 10, 2),   # Mahatma Gandhi Jayanti
    (2026, 10, 8),   # Dussehra
    (2026, 10, 29),  # Diwali (Laxmi Pujan)
    (2026, 11, 25),  # Guru Nanak Jayanti
    (2026, 12, 25),  # Christmas
}


def _is_nse_market_hours() -> bool:
    """Check if current time is within NSE trading hours (Mon-Fri, 9:15-15:30 IST, excluding holidays)."""
    now_ist = datetime.now(_IST)
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Check NSE exchange holidays
    if (now_ist.year, now_ist.month, now_ist.day) in _NSE_HOLIDAYS:
        return False
    market_open = now_ist.replace(hour=_NSE_OPEN_HOUR, minute=_NSE_OPEN_MIN, second=0, microsecond=0)
    market_close = now_ist.replace(hour=_NSE_CLOSE_HOUR, minute=_NSE_CLOSE_MIN, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


def stable_idempotency_key(bar_ts_iso: str, strategy_id: str, symbol: str, side: str) -> str:
    """Stable key for autonomous orders. Same bar → same key → idempotency prevents duplicate."""
    return IdempotencyStore.derive_key_bar_stable(bar_ts_iso, strategy_id, symbol, side)


class AutonomousLoop:
    """
    Bar-based execution cycle. On each bar:
    - Check safe_mode; if set, skip.
    - Pull latest bars from bar provider.
    - Run strategy runner → signals.
    - Run portfolio allocator → (signal, quantity).
    - Apply risk caps (via OrderEntryService).
    - Submit via OrderEntryService only with stable idempotency key.
    """

    def __init__(
        self,
        submit_order_fn: Callable[..., Awaitable[OrderEntryResult]],
        *,
        get_safe_mode: Callable[[], bool],
        get_bar_ts: Optional[Callable[[], str]] = None,
        get_bars: Optional[Callable[[str, Exchange, str, int], List[Bar]]] = None,
        get_symbols: Optional[Callable[[], List[Tuple[str, Exchange]]]] = None,
        strategy_runner=None,
        allocator=None,
        get_risk_state: Optional[Callable[[], dict]] = None,
        get_positions: Optional[Callable[[], List[Position]]] = None,
        drift_gate: Optional[Callable[[], bool]] = None,
        regime_gate: Optional[Callable[[], bool]] = None,
        get_market_feed_healthy: Optional[Callable[[], bool]] = None,
        feature_engine=None,
        regime_classifier=None,
        market_scanner=None,
        performance_tracker=None,
        ws_broadcast: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_daily_reset: Optional[Callable[[], None]] = None,
        poll_interval_seconds: float = 60.0,
        paper_mode: bool = False,
    ):
        self.submit_order_fn = submit_order_fn
        self.get_safe_mode = get_safe_mode
        self.get_bar_ts = get_bar_ts or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.get_bars = get_bars
        self.get_symbols = get_symbols
        self.strategy_runner = strategy_runner
        self.allocator = allocator
        self.get_risk_state = get_risk_state
        self.get_positions = get_positions
        self.drift_gate = drift_gate
        self.regime_gate = regime_gate
        self.get_market_feed_healthy = get_market_feed_healthy
        self.feature_engine = feature_engine
        self.regime_classifier = regime_classifier
        self.market_scanner = market_scanner
        self.performance_tracker = performance_tracker
        self.ws_broadcast = ws_broadcast
        self.on_daily_reset = on_daily_reset
        self.poll_interval = poll_interval_seconds
        self.paper_mode = paper_mode
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_bar_ts: Optional[str] = None

        # ── Smart trade management ──
        # Track open positions for stop-loss/take-profit management
        self._open_trades: dict = {}  # trade_key -> {side, entry_price, qty, strategy_id, stop_loss, take_profit, trailing_stop}
        self._daily_pnl: float = 0.0
        self._daily_loss_limit: float = -0.02  # -2% daily loss → shutdown
        self._tick_count: int = 0
        self._current_trading_date: Optional[str] = None  # track date for daily P&L reset

        # ── Persistence (write-ahead for open trades) ──
        self._open_trade_repo = None  # set via set_open_trade_repo()

        # ── Kill switch auto-close tracking ──
        self._kill_switch_fn: Optional[Callable[[], Awaitable[bool]]] = None  # async callable returns True if armed
        self._last_kill_switch_armed: bool = False
        self._close_attempts: Dict[str, int] = {}  # trade_key -> retry count

        # ── Sentiment integration (Phase 1: news-aware trading) ──
        self._sentiment_service = None  # set via set_sentiment_service()
        self._last_sentiment_score: float = 0.5  # neutral default
        self._last_sentiment_ts: Optional[float] = None
        self._sentiment_cache_ttl: float = 300.0  # refresh sentiment every 5 min

    def set_sentiment_service(self, service) -> None:
        """Wire news sentiment service for exposure adjustment."""
        self._sentiment_service = service
        logger.info("Sentiment service wired to autonomous loop")

    async def _fetch_sentiment(self) -> float:
        """Fetch latest market sentiment; returns multiplier 0.5-1.2."""
        import time as _time
        if self._sentiment_service is None:
            return 1.0
        # Cache: only refresh every 5 minutes
        now = _time.time()
        if self._last_sentiment_ts and (now - self._last_sentiment_ts) < self._sentiment_cache_ttl:
            return self._sentiment_to_multiplier(self._last_sentiment_score)
        try:
            result = await self._sentiment_service.analyze(
                "Indian stock market overall outlook today", source="autonomous_loop"
            )
            if result:
                self._last_sentiment_score = result.score
                self._last_sentiment_ts = now
                mult = self._sentiment_to_multiplier(result.score)
                logger.info("Sentiment update: %s (score=%.2f, multiplier=%.2f, suggestion=%s)",
                           result.sentiment, result.score, mult, result.risk_reduction_suggestion)
                return mult
        except Exception as e:
            logger.debug("Sentiment fetch failed: %s", e)
        return 1.0

    @staticmethod
    def _sentiment_to_multiplier(score: float) -> float:
        """Convert sentiment score (0-1) to exposure multiplier (0.5-1.2)."""
        # score 0.0 (very negative) → 0.5x, score 0.5 (neutral) → 1.0x, score 1.0 (very positive) → 1.2x
        if score <= 0.3:
            return 0.5  # Very bearish: halve exposure
        elif score <= 0.45:
            return 0.7  # Bearish: reduce exposure
        elif score <= 0.55:
            return 1.0  # Neutral: normal
        elif score <= 0.7:
            return 1.1  # Bullish: slight increase
        else:
            return 1.2  # Very bullish: max increase

    def _bar_ts(self) -> str:
        return self.get_bar_ts()

    def _drift_ok(self) -> bool:
        if self.drift_gate is None:
            return True
        return self.drift_gate()

    def _regime_ok(self) -> bool:
        if self.regime_gate is None:
            return True
        return self.regime_gate()

    def _market_feed_ok(self) -> bool:
        if self.get_market_feed_healthy is None:
            return True
        return self.get_market_feed_healthy()

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message via WebSocket if callback is set."""
        if self.ws_broadcast:
            try:
                await self.ws_broadcast(message)
            except Exception as e:
                logger.debug("WebSocket broadcast failed: %s", e)

    def _reset_daily_pnl_if_new_day(self) -> None:
        """Reset daily P&L at the start of each new trading day (IST)."""
        today_ist = datetime.now(_IST).strftime("%Y-%m-%d")
        if self._current_trading_date != today_ist:
            if self._current_trading_date is not None:
                logger.info("New trading day %s — resetting daily P&L (was %.2f)", today_ist, self._daily_pnl)
            self._daily_pnl = 0.0
            self._current_trading_date = today_ist
            # Reset RiskManager's daily_pnl via callback
            if self.on_daily_reset:
                try:
                    self.on_daily_reset()
                    logger.info("RiskManager daily P&L reset for new day")
                except Exception as e:
                    logger.warning("on_daily_reset callback failed: %s", e)

    async def _tick(self) -> None:
        if self.get_safe_mode():
            return

        # ── Forced close: per-position hard stop loss (Sprint 7.9) ──
        _rm = getattr(self, '_risk_manager', None)
        if _rm is not None:
            try:
                forced_closes = _rm.get_forced_close_symbols()
                for sym in forced_closes:
                    logger.warning("Forced close triggered for %s (per-position loss limit)", sym)
                    await self._emergency_close_symbol(sym, reason="per_position_loss_limit")
            except Exception as e:
                logger.debug("Forced close check failed: %s", e)

        # Kill switch auto-close: detect transition from unarmed → armed
        if self._kill_switch_fn is not None:
            try:
                currently_armed = await self._kill_switch_fn()
                if currently_armed and not self._last_kill_switch_armed:
                    # Transition: kill switch just armed → auto-close all positions
                    await self._auto_close_all_positions()
                elif currently_armed and self._open_trades:
                    # Still armed with remaining positions → retry failed closes
                    remaining_to_retry = {k for k, v in self._close_attempts.items() if v > 0}
                    if remaining_to_retry:
                        await self._auto_close_all_positions()
                self._last_kill_switch_armed = currently_armed
            except Exception as e:
                logger.debug("Kill switch check in tick failed: %s", e)

        # Market hours gate: skip outside NSE 9:15-15:30 IST, Mon-Fri (paper mode trades anytime)
        if not self.paper_mode and not _is_nse_market_hours():
            return

        if not self._market_feed_ok():
            return
        if not self._drift_ok() or not self._regime_ok():
            return

        # Reset daily P&L at start of each new trading day
        self._reset_daily_pnl_if_new_day()

        # Daily loss limit check
        risk_state = self.get_risk_state() if self.get_risk_state else {}
        equity = risk_state.get("equity") or 100000.0
        if equity > 0 and self._daily_pnl / equity < self._daily_loss_limit:
            return

        bar_ts = self._bar_ts()
        if bar_ts == self._last_bar_ts:
            return
        self._tick_count += 1

        if not all([self.get_bars, self.get_symbols, self.strategy_runner, self.allocator, self.get_risk_state, self.get_positions]):
            # Even without BarCache, if we have a market_scanner, use it
            if self.market_scanner is None:
                self._last_bar_ts = bar_ts
                return

        from src.strategy_engine.base import MarketState

        symbols_exchanges = self.get_symbols() or []
        if not symbols_exchanges:
            logger.info("Tick %d: empty universe, skipping", self._tick_count)
            return
        all_signals: List[Signal] = []
        regime_scale_from_classifier: Optional[float] = None
        for symbol, exchange in symbols_exchanges:
            bars = self.get_bars(symbol, exchange, "1m", 100)
            if len(bars) < 20:
                continue
            latest = bars[-1]
            metadata = {}
            if self.feature_engine:
                try:
                    metadata["features"] = self.feature_engine.build_features(bars)
                except Exception as e:
                    logger.debug("Feature build failed %s: %s", symbol, e)
            _sym_regime_scale = None
            if self.regime_classifier and metadata.get("features"):
                try:
                    feats = metadata["features"]
                    import numpy as np
                    rets = np.array([bars[i].close / (bars[i - 1].close or 1e-12) - 1.0 for i in range(1, len(bars))])
                    vol = float(np.std(rets)) if len(rets) > 1 else 0.0
                    trend = feats.get("ema_spread", 0.0) or 0.0
                    res = self.regime_classifier.classify(rets, vol, float(trend), None)
                    metadata["regime"] = res.label.value if hasattr(res.label, "value") else str(res.label)
                    if res.label.value in ("high_volatility", "crisis"):
                        _sym_regime_scale = 0.5
                    elif res.label.value in ("low_volatility", "trending_up", "trending_down"):
                        _sym_regime_scale = 1.0
                    if _sym_regime_scale is not None:
                        regime_scale_from_classifier = min(regime_scale_from_classifier, _sym_regime_scale) if regime_scale_from_classifier is not None else _sym_regime_scale
                except Exception as e:
                    logger.debug("Regime classify failed: %s", e)
            state = MarketState(
                symbol=symbol,
                exchange=exchange,
                bars=bars,
                latest_price=latest.close,
                volume=latest.volume,
                metadata=metadata,
            )
            try:
                sigs = self.strategy_runner.run(state)
                all_signals.extend(sigs)
            except Exception as e:
                logger.exception("Strategy runner failed for %s: %s", symbol, e)

        logger.info("Tick %d: %d signals from %d symbols", self._tick_count, len(all_signals), len(symbols_exchanges))
        risk_state = self.get_risk_state()
        equity = risk_state.get("equity") or 0.0
        if equity <= 0:
            logger.warning("Tick %d: equity=%.2f, skipping allocation (no capital)", self._tick_count, equity)
            return
        exposure_mult = risk_state.get("exposure_multiplier") or 1.0

        # Apply news sentiment to exposure multiplier
        sentiment_mult = await self._fetch_sentiment()
        exposure_mult = round(max(0.5, min(1.5, exposure_mult * sentiment_mult)), 2)
        max_position_pct = risk_state.get("max_position_pct") or 5.0
        drawdown_scale = risk_state.get("drawdown_scale")
        regime_scale = risk_state.get("regime_scale") or regime_scale_from_classifier
        positions = self.get_positions()
        raw_allocated = self.allocator.allocate(
            all_signals,
            equity,
            positions,
            exposure_multiplier=exposure_mult,
            drawdown_scale=drawdown_scale,
            regime_scale=regime_scale,
            max_position_pct=max_position_pct,
        )
        allocated: List[Tuple[Signal, int]] = []
        for item in raw_allocated:
            if hasattr(item, "signal") and hasattr(item, "quantity"):
                allocated.append((item.signal, item.quantity))
            else:
                allocated.append((item[0], item[1]))
        logger.info("Allocated %d orders to submit", len(allocated))

        for signal, qty in allocated:
            if qty <= 0:
                continue
            idem_key = stable_idempotency_key(bar_ts, signal.strategy_id, signal.symbol, signal.side.value)
            # 8.3: Attach prediction metadata to orders for outcome tracking
            order_metadata = dict(metadata) if metadata else {}
            order_metadata.update({
                "predicted_direction": signal.side.value,
                "predicted_confidence": signal.score,
                "signal_source": signal.strategy_id,
                "entry_price": signal.price,
            })
            # Attach ensemble weights if available
            _ensemble = getattr(self, '_ensemble_engine', None)
            if _ensemble and hasattr(_ensemble, 'weights'):
                order_metadata["model_weights"] = {k: round(v, 4) for k, v in _ensemble.weights.items()}

            req = OrderEntryRequest(
                signal=signal,
                quantity=qty,
                order_type=OrderType.LIMIT,
                limit_price=signal.price,
                idempotency_key=idem_key,
                source="autonomous",
                metadata=order_metadata,
            )
            try:
                result = await self.submit_order_fn(req)
                if result.success:
                    logger.info("Autonomous order submitted order_id=%s strategy=%s symbol=%s side=%s qty=%s", result.order_id, signal.strategy_id, signal.symbol, signal.side.value, qty)
                    self._track_open_trade(signal, qty)
                else:
                    logger.warning("Autonomous order rejected reason=%s detail=%s", result.reject_reason, result.reject_detail)
            except Exception as e:
                logger.exception("Autonomous submit failed: %s", e)

        # Broadcast signal_generated events for each signal to WebSocket (dashboard SignalFeed)
        for signal in all_signals:
            await self._broadcast({
                "type": "signal_generated",
                "strategy_id": signal.strategy_id,
                "symbol": signal.symbol,
                "exchange": getattr(signal.exchange, "value", str(signal.exchange)),
                "side": signal.side.value,
                "score": signal.score,
                "price": signal.price,
                "reason": signal.reason,
                "timestamp": signal.ts.isoformat() if signal.ts else datetime.now(timezone.utc).isoformat(),
            })

        self._last_bar_ts = bar_ts

        # ── Check stop-loss / take-profit on open trades ──
        await self._check_stop_loss_take_profit(bar_ts)

        # Additionally, run full-market scanner if available
        if self.market_scanner is not None:
            try:
                import asyncio as _aio
                _loop = _aio.get_event_loop()
                scanner_signals = await _loop.run_in_executor(
                    None, lambda: self.market_scanner.scan_to_signals(bar_cache=None)
                )
                existing_symbols = {signal.symbol for signal, qty in allocated}
                submitted_symbols = set()
                for signal, qty in allocated:
                    submitted_symbols.add(signal.symbol)

                scanner_allocated = self.allocator.allocate(
                    scanner_signals,
                    equity,
                    positions,
                    exposure_multiplier=exposure_mult,
                    drawdown_scale=drawdown_scale,
                    regime_scale=regime_scale,
                    max_position_pct=max_position_pct,
                )
                for item in scanner_allocated:
                    if hasattr(item, "signal") and hasattr(item, "quantity"):
                        sig, qty = item.signal, item.quantity
                    else:
                        sig, qty = item[0], item[1]
                    if qty <= 0 or sig.symbol in submitted_symbols:
                        continue
                    submitted_symbols.add(sig.symbol)
                    idem_key = stable_idempotency_key(bar_ts, sig.strategy_id, sig.symbol, sig.side.value)
                    req = OrderEntryRequest(
                        signal=sig,
                        quantity=qty,
                        order_type=OrderType.LIMIT,
                        limit_price=sig.price,
                        idempotency_key=idem_key,
                        source="scanner",
                    )
                    try:
                        result = await self.submit_order_fn(req)
                        if result.success:
                            logger.info("Scanner order submitted order_id=%s symbol=%s side=%s qty=%s",
                                        result.order_id, sig.symbol, sig.side.value, qty)
                            self._track_open_trade(sig, qty)
                        else:
                            logger.debug("Scanner order rejected: %s", result.reject_reason)
                    except Exception as e:
                        logger.exception("Scanner submit failed: %s", e)
            except Exception as e:
                logger.warning("Market scanner tick failed: %s", e)

        # Mark-to-market: update unrealized P&L and broadcast
        await self._mark_to_market()

        logger.debug("Autonomous loop tick bar_ts=%s signals=%s allocated=%s open_trades=%s daily_pnl=%.2f",
                     bar_ts, len(all_signals), len(allocated), len(self._open_trades), self._daily_pnl)

    def set_kill_switch(self, kill_switch_fn: Callable[[], Awaitable[bool]]) -> None:
        """Set async callable that returns True if kill switch is armed."""
        self._kill_switch_fn = kill_switch_fn

    async def _auto_close_all_positions(self) -> None:
        """
        Emergency close-out: submit reduce-only LIMIT orders for all open positions.
        Rate limited: max 2 close orders per second.
        Uses force_reduce=True to bypass daily loss check, circuit breaker, and rate limiter.
        """
        if not self._open_trades:
            logger.info("Kill switch auto-close: no open trades to close")
            return

        logger.warning("Kill switch ARMED — initiating auto-close of %d open positions", len(self._open_trades))
        close_count = 0

        for trade_key, trade in list(self._open_trades.items()):
            symbol = trade.get("symbol") or trade_key.split(":")[0]

            # Rate limit: max 2 per second
            if close_count > 0 and close_count % 2 == 0:
                await asyncio.sleep(1.0)

            try:
                # Get current price for limit order with buffer
                exit_price = trade["entry_price"]  # fallback
                if self.get_bars:
                    bars = self.get_bars(symbol, Exchange.NSE, "1m", 2)
                    if bars:
                        exit_price = bars[-1].close

                # Apply 0.5% buffer for execution room
                close_side = SignalSide.SELL if trade["side"] == "BUY" else SignalSide.BUY
                if close_side == SignalSide.SELL:
                    limit_price = round(exit_price * 0.995, 2)  # sell slightly below market
                else:
                    limit_price = round(exit_price * 1.005, 2)  # buy slightly above market

                close_signal = Signal(
                    strategy_id=trade["strategy_id"],
                    symbol=symbol,
                    exchange=Exchange.NSE,
                    side=close_side,
                    score=1.0,
                    portfolio_weight=0.0,
                    risk_level="EMERGENCY",
                    reason=f"KILL_SWITCH_AUTO_CLOSE: {trade['side']} {symbol} entry={trade['entry_price']:.2f}",
                    price=limit_price,
                    ts=datetime.now(timezone.utc),
                )
                bar_ts = self._bar_ts()
                idem_key = stable_idempotency_key(
                    bar_ts, f"{trade['strategy_id']}_kill_close", symbol, close_side.value
                )
                req = OrderEntryRequest(
                    signal=close_signal,
                    quantity=trade["qty"],
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price,
                    idempotency_key=idem_key,
                    source="kill_switch_auto_close",
                    force_reduce=True,
                )
                result = await self.submit_order_fn(req)
                if result.success:
                    logger.info("Kill switch auto-close submitted: order_id=%s %s %s qty=%d limit=%.2f",
                                result.order_id, close_side.value, symbol, trade["qty"], limit_price)
                    if self._open_trade_repo:
                        self._open_trade_repo.delete_trade(trade_key)
                    del self._open_trades[trade_key]
                    self._close_attempts.pop(trade_key, None)
                    close_count += 1
                else:
                    retries = self._close_attempts.get(trade_key, 0)
                    self._close_attempts[trade_key] = retries + 1
                    logger.warning("Kill switch auto-close rejected for %s: %s (attempt %d)",
                                   symbol, result.reject_reason, retries + 1)
            except Exception as e:
                retries = self._close_attempts.get(trade_key, 0)
                self._close_attempts[trade_key] = retries + 1
                logger.exception("Kill switch auto-close failed for %s: %s (attempt %d)", symbol, e, retries + 1)

        logger.info("Kill switch auto-close completed: %d orders submitted, %d remaining",
                     close_count, len(self._open_trades))

    async def _emergency_close_symbol(self, symbol: str, reason: str = "forced_close") -> None:
        """Emergency close a single symbol's position from _open_trades. Used for per-position hard stops."""
        matching_keys = [k for k, v in self._open_trades.items() if (v.get("symbol") or k.split(":")[0]) == symbol]
        if not matching_keys:
            logger.info("Forced close: no open trade found for %s (may already be closed)", symbol)
            return

        for trade_key in matching_keys:
            trade = self._open_trades[trade_key]
            try:
                exit_price = trade["entry_price"]
                if self.get_bars:
                    bars = self.get_bars(symbol, Exchange.NSE, "1m", 2)
                    if bars:
                        exit_price = bars[-1].close

                close_side = SignalSide.SELL if trade["side"] == "BUY" else SignalSide.BUY
                if close_side == SignalSide.SELL:
                    limit_price = round(exit_price * 0.995, 2)
                else:
                    limit_price = round(exit_price * 1.005, 2)

                close_signal = Signal(
                    strategy_id=trade["strategy_id"],
                    symbol=symbol,
                    exchange=Exchange.NSE,
                    side=close_side,
                    score=1.0,
                    portfolio_weight=0.0,
                    risk_level="EMERGENCY",
                    reason=f"{reason}: {trade['side']} {symbol} entry={trade['entry_price']:.2f}",
                    price=limit_price,
                    ts=datetime.now(timezone.utc),
                )
                bar_ts = self._bar_ts()
                idem_key = stable_idempotency_key(bar_ts, f"{trade['strategy_id']}_{reason}", symbol, close_side.value)
                req = OrderEntryRequest(
                    signal=close_signal,
                    quantity=trade["qty"],
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price,
                    idempotency_key=idem_key,
                    source=reason,
                    force_reduce=True,
                )
                result = await self.submit_order_fn(req)
                if result.success:
                    logger.warning("Forced close submitted: order_id=%s %s %s qty=%d reason=%s",
                                   result.order_id, close_side.value, symbol, trade["qty"], reason)
                    if self._open_trade_repo:
                        self._open_trade_repo.delete_trade(trade_key)
                    del self._open_trades[trade_key]
                else:
                    logger.warning("Forced close rejected for %s: %s", symbol, result.reject_reason)
            except Exception as e:
                logger.exception("Forced close failed for %s: %s", symbol, e)

    def set_open_trade_repo(self, repo) -> None:
        """Attach a persistence repository for write-ahead open trade tracking."""
        self._open_trade_repo = repo

    def load_open_trades_from_db(self) -> int:
        """
        Cold-start recovery: load open trades from DB into in-memory dict.
        Returns number of trades loaded. Call AFTER set_open_trade_repo().
        """
        if self._open_trade_repo is None:
            return 0
        try:
            rows = self._open_trade_repo.load_all()
            loaded = 0
            for row in rows:
                trade_key = row["trade_key"]
                self._open_trades[trade_key] = {
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "entry_price": row["entry_price"],
                    "qty": row["qty"],
                    "strategy_id": row["strategy_id"],
                    "stop_loss": row["stop_loss"],
                    "take_profit": row["take_profit"],
                    "trailing_stop": row.get("trailing_stop"),
                }
                loaded += 1
            if loaded:
                logger.info("Cold-start: recovered %d open trades from DB", loaded)
            return loaded
        except Exception as e:
            logger.error("Cold-start: failed to load open trades: %s", e)
            return 0

    @staticmethod
    def _calc_atr(bars: list, period: int = 14) -> Optional[float]:
        """Calculate Average True Range from recent bars. Returns None if insufficient data."""
        if len(bars) < period + 1:
            return None
        import numpy as _np
        highs = _np.array([b.high for b in bars[-(period + 1):]], dtype=float)
        lows = _np.array([b.low for b in bars[-(period + 1):]], dtype=float)
        closes = _np.array([b.close for b in bars[-(period + 1):]], dtype=float)
        tr = _np.maximum(
            highs[1:] - lows[1:],
            _np.maximum(_np.abs(highs[1:] - closes[:-1]), _np.abs(lows[1:] - closes[:-1]))
        )
        return float(_np.mean(tr[-period:]))

    def _track_open_trade(self, signal: Signal, qty: int) -> None:
        """Track a new trade for stop-loss/take-profit management. Write-ahead: DB first."""
        entry_price = signal.price or 0.0
        if entry_price <= 0:
            return

        # Calculate real ATR from bars if available, fall back to 1.5% estimate
        atr_estimate = entry_price * 0.015
        if self.get_bars is not None:
            try:
                bars = self.get_bars(signal.symbol, signal.exchange, "1m", 100)
                atr = self._calc_atr(bars, 14)
                if atr is not None and atr > 0:
                    atr_estimate = atr
            except Exception:
                pass  # keep default estimate

        if hasattr(signal, 'stop_loss') and signal.stop_loss and signal.stop_loss > 0:
            stop_loss = signal.stop_loss
        else:
            stop_loss = entry_price - 2 * atr_estimate if signal.side.value == "BUY" else entry_price + 2 * atr_estimate

        if hasattr(signal, 'target') and signal.target and signal.target > 0:
            take_profit = signal.target
        else:
            risk = abs(entry_price - stop_loss)
            take_profit = entry_price + 1.5 * risk if signal.side.value == "BUY" else entry_price - 1.5 * risk

        trade_key = f"{signal.symbol}:{signal.strategy_id}"
        sl_rounded = round(stop_loss, 2)
        tp_rounded = round(take_profit, 2)

        # Write-ahead: persist to DB BEFORE in-memory update
        if self._open_trade_repo is not None:
            ok = self._open_trade_repo.upsert_trade(
                trade_key=trade_key,
                symbol=signal.symbol,
                exchange=getattr(signal.exchange, "value", "NSE"),
                side=signal.side.value,
                quantity=qty,
                entry_price=entry_price,
                stop_loss=sl_rounded,
                take_profit=tp_rounded,
                strategy_id=signal.strategy_id,
            )
            if not ok:
                logger.error("Write-ahead failed for trade %s — skipping in-memory track", trade_key)
                return

        self._open_trades[trade_key] = {
            "symbol": signal.symbol,
            "side": signal.side.value,
            "entry_price": entry_price,
            "qty": qty,
            "strategy_id": signal.strategy_id,
            "stop_loss": sl_rounded,
            "take_profit": tp_rounded,
            "trailing_stop": None,
        }
        logger.info("Trade tracked: %s %s entry=%.2f SL=%.2f TP=%.2f qty=%d",
                    signal.side.value, signal.symbol, entry_price, sl_rounded, tp_rounded, qty)

    async def _check_stop_loss_take_profit(self, bar_ts: str) -> None:
        """Check all open trades for stop-loss or take-profit hits."""
        if not self._open_trades or not self.get_bars:
            return

        to_close = []
        for trade_key, trade in list(self._open_trades.items()):
            symbol = trade.get("symbol") or trade_key.split(":")[0]
            try:
                bars = self.get_bars(symbol, Exchange.NSE, "1m", 5)
                if not bars:
                    continue
                current_price = bars[-1].close
                entry = trade["entry_price"]
                sl = trade["stop_loss"]
                tp = trade["take_profit"]
                side = trade["side"]

                # Trailing stop: if in profit by 1.5%, move stop to breakeven (BEFORE hit check)
                if side == "BUY" and current_price > entry * 1.015:
                    new_trailing = max(sl, entry)
                    if trade.get("trailing_stop") is None or new_trailing > trade["trailing_stop"]:
                        trade["trailing_stop"] = new_trailing
                        trade["stop_loss"] = new_trailing
                        sl = new_trailing  # update local for hit check
                        if self._open_trade_repo:
                            self._open_trade_repo.update_sl_tp(trade_key, stop_loss=new_trailing, trailing_stop=new_trailing)
                elif side == "SELL" and current_price < entry * 0.985:
                    new_trailing = min(sl, entry)
                    if trade.get("trailing_stop") is None or new_trailing < trade["trailing_stop"]:
                        trade["trailing_stop"] = new_trailing
                        trade["stop_loss"] = new_trailing
                        sl = new_trailing  # update local for hit check
                        if self._open_trade_repo:
                            self._open_trade_repo.update_sl_tp(trade_key, stop_loss=new_trailing, trailing_stop=new_trailing)

                hit_sl = (side == "BUY" and current_price <= sl) or (side == "SELL" and current_price >= sl)
                hit_tp = (side == "BUY" and current_price >= tp) or (side == "SELL" and current_price <= tp)

                if hit_sl or hit_tp:
                    reason = "STOP_LOSS" if hit_sl else "TAKE_PROFIT"
                    pnl = (current_price - entry) * trade["qty"] if side == "BUY" else (entry - current_price) * trade["qty"]
                    self._daily_pnl += pnl

                    if self.performance_tracker:
                        try:
                            self.performance_tracker.record_fill(trade["strategy_id"], pnl)
                        except Exception:
                            pass

                    logger.info("%s hit for %s %s: entry=%.2f exit=%.2f PnL=%.2f daily_total=%.2f",
                                reason, side, symbol, entry, current_price, pnl, self._daily_pnl)
                    to_close.append((trade_key, symbol, trade, current_price, reason, pnl))
            except Exception as e:
                logger.debug("SL/TP check failed for %s: %s", symbol, e)

        # Submit actual close orders to broker for each SL/TP hit
        for trade_key, symbol, trade, exit_price, reason, pnl in to_close:
            close_side = SignalSide.SELL if trade["side"] == "BUY" else SignalSide.BUY
            close_signal = Signal(
                strategy_id=trade["strategy_id"],
                symbol=symbol,
                exchange=Exchange.NSE,
                side=close_side,
                score=1.0,
                portfolio_weight=0.0,
                risk_level="NORMAL",
                reason=f"{reason}: entry={trade['entry_price']:.2f} exit={exit_price:.2f} pnl={pnl:.2f}",
                price=exit_price,
                ts=datetime.now(timezone.utc),
            )
            idem_key = stable_idempotency_key(
                bar_ts, f"{trade['strategy_id']}_{reason.lower()}", symbol, close_side.value
            )
            # Use LIMIT orders with buffer for SL/TP (avoid market order slippage)
            if reason == "STOP_LOSS":
                # SL: give 0.3% execution room
                if close_side == SignalSide.SELL:
                    sl_limit_price = round(exit_price * 0.997, 2)
                else:
                    sl_limit_price = round(exit_price * 1.003, 2)
            else:
                # TP: use exact exit price
                sl_limit_price = round(exit_price, 2)
            req = OrderEntryRequest(
                signal=close_signal,
                quantity=trade["qty"],
                order_type=OrderType.LIMIT,
                limit_price=sl_limit_price,
                idempotency_key=idem_key,
                source=f"autonomous_{reason.lower()}",
                force_reduce=True,
            )
            try:
                result = await self.submit_order_fn(req)
                if result.success:
                    logger.info("%s close order submitted: order_id=%s %s %s qty=%d exit=%.2f pnl=%.2f",
                                reason, result.order_id, close_side.value, symbol, trade["qty"], exit_price, pnl)
                else:
                    logger.warning("%s close order rejected for %s: %s — removing from tracking anyway",
                                   reason, symbol, result.reject_reason)
            except Exception as e:
                logger.exception("%s close order submit failed for %s: %s — removing from tracking", reason, symbol, e)

            # Broadcast trade closed event
            await self._broadcast({
                "type": "trade_closed",
                "symbol": symbol,
                "side": trade["side"],
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "quantity": trade["qty"],
                "pnl": pnl,
                "reason": reason,
                "strategy_id": trade["strategy_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Remove from DB (write-ahead: DB before in-memory)
            if self._open_trade_repo:
                self._open_trade_repo.delete_trade(trade_key)
            del self._open_trades[trade_key]

    async def _mark_to_market(self) -> None:
        """Update unrealized P&L for all open trades and broadcast to dashboard."""
        if not self._open_trades or not self.get_bars:
            return

        total_unrealized = 0.0
        open_positions = []
        for trade_key, trade in self._open_trades.items():
            symbol = trade.get("symbol") or trade_key.split(":")[0]
            try:
                bars = self.get_bars(symbol, Exchange.NSE, "1m", 2)
                if not bars:
                    continue
                current_price = bars[-1].close
                entry = trade["entry_price"]
                side = trade["side"]
                qty = trade["qty"]
                unrealized = (current_price - entry) * qty if side == "BUY" else (entry - current_price) * qty
                total_unrealized += unrealized
                open_positions.append({
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry,
                    "current_price": current_price,
                    "quantity": qty,
                    "unrealized_pnl": round(unrealized, 2),
                    "stop_loss": trade["stop_loss"],
                    "take_profit": trade["take_profit"],
                    "strategy_id": trade["strategy_id"],
                })
            except Exception:
                pass

        if open_positions:
            await self._broadcast({
                "type": "portfolio_mark_to_market",
                "open_positions": open_positions,
                "total_unrealized_pnl": round(total_unrealized, 2),
                "total_realized_pnl": round(self._daily_pnl, 2),
                "total_pnl": round(self._daily_pnl + total_unrealized, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Autonomous loop error: %s", e)
            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Autonomous loop started (poll_interval=%.1fs)", self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

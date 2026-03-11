"""
Autonomous execution loop: bar-based cycle, stable idempotency key, drift/regime gating.
Closed loop: pull bars → strategy runner → allocator → risk → OrderEntryService only.
No direct gateway calls. Idempotency key: {bar_ts}-{strategy_id}-{symbol}-{side}.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.events import Bar, Exchange, Position, Signal, SignalSide
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.request import OrderEntryRequest, OrderEntryResult
from src.core.events import OrderType
from src.execution.algorithms.twap import TWAPAlgorithm, TWAPConfig
from src.execution.algorithms.vwap import VWAPAlgorithm, VWAPConfig

logger = logging.getLogger(__name__)

# ADV thresholds for algorithm selection
_ADV_DIRECT_MAX_PCT = 1.0    # < 1% ADV → direct market order
_ADV_TWAP_MAX_PCT = 5.0      # 1-5% ADV → TWAP (30 min)
_ADV_VWAP_MAX_PCT = 10.0     # 5-10% ADV → VWAP (standard)
# > 10% ADV → VWAP with extended duration + alert

# NSE market hours: 9:15 AM - 3:30 PM IST (UTC+5:30)
_IST = timezone(timedelta(hours=5, minutes=30))
_NSE_OPEN_HOUR, _NSE_OPEN_MIN = 9, 15
_NSE_CLOSE_HOUR, _NSE_CLOSE_MIN = 15, 30

# NSE exchange holidays: loaded dynamically from deploy/nse_holidays.json
# Falls back to hardcoded set if JSON file is not available.
_NSE_HOLIDAYS: set = set()
_NSE_HOLIDAYS_LOADED = False


def _load_nse_holidays() -> set:
    """Load NSE holidays from deploy/nse_holidays.json config file.
    Returns set of (year, month, day) tuples.
    Falls back to empty set if file not found (paper mode still works)."""
    global _NSE_HOLIDAYS, _NSE_HOLIDAYS_LOADED
    if _NSE_HOLIDAYS_LOADED:
        return _NSE_HOLIDAYS

    import os
    holidays_set: set = set()

    # Try multiple paths for the holidays JSON file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate_paths = [
        os.path.join(base_dir, "deploy", "nse_holidays.json"),
        os.path.join(base_dir, "config", "nse_holidays.json"),
        os.environ.get("NSE_HOLIDAYS_PATH", ""),
    ]

    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            holiday_map = data.get("holidays", {})
            for year_str, dates in holiday_map.items():
                for date_str in dates:
                    parts = date_str.split("-")
                    if len(parts) == 3:
                        holidays_set.add((int(parts[0]), int(parts[1]), int(parts[2])))
            logger.info(
                "Loaded %d NSE holidays from %s (last_updated=%s)",
                len(holidays_set), path, data.get("last_updated", "unknown"),
            )
            _NSE_HOLIDAYS = holidays_set
            _NSE_HOLIDAYS_LOADED = True
            return holidays_set
        except Exception as e:
            logger.warning("Failed to load NSE holidays from %s: %s", path, e)

    # Fallback: minimal hardcoded holidays (fixed national holidays only)
    current_year = datetime.now(_IST).year
    for year in range(current_year, current_year + 2):
        holidays_set.update({
            (year, 1, 26),   # Republic Day
            (year, 8, 15),   # Independence Day
            (year, 10, 2),   # Gandhi Jayanti
            (year, 12, 25),  # Christmas
        })
    logger.warning(
        "NSE holidays JSON not found — using minimal fallback (%d holidays). "
        "Create deploy/nse_holidays.json for full holiday calendar.",
        len(holidays_set),
    )
    _NSE_HOLIDAYS = holidays_set
    _NSE_HOLIDAYS_LOADED = True
    return holidays_set


def _is_nse_market_hours() -> bool:
    """Check if current time is within NSE trading hours (Mon-Fri, 9:15-15:30 IST, excluding holidays)."""
    now_ist = datetime.now(_IST)
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Check NSE exchange holidays (dynamically loaded from config)
    holidays = _load_nse_holidays()
    if (now_ist.year, now_ist.month, now_ist.day) in holidays:
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
        adv_cache=None,
        get_market_price: Optional[Callable] = None,
        daily_loss_limit: float = -0.02,
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
        self._open_trades_lock = asyncio.Lock()  # guards _open_trades mutations across async yields
        # DEPRECATED: _daily_pnl is maintained only as a fallback mirror.
        # The single source of truth for daily PnL is RiskManager.daily_pnl.
        # All PnL mutations now go through RiskManager.register_pnl().
        self._daily_pnl: float = 0.0
        self._daily_loss_limit: float = daily_loss_limit
        self._tick_count: int = 0
        self._current_trading_date: Optional[str] = None  # track date for daily P&L reset

        # ── Persistence (write-ahead for open trades) ──
        self._open_trade_repo = None  # set via set_open_trade_repo()

        # ── Kill switch auto-close tracking ──
        self._kill_switch_fn: Optional[Callable[[], Awaitable[bool]]] = None  # async callable returns True if armed
        self._last_kill_switch_armed: bool = False
        self._close_attempts: Dict[str, int] = {}  # trade_key -> retry count

        # ── Feature normalization (P0-3: z-score before model inference) ──
        self._feature_normalizer = None  # set via set_feature_normalizer()

        # ── Sentiment integration (Phase 1: news-aware trading) ──
        self._sentiment_service = None  # set via set_sentiment_service()
        self._sentiment_predictor = None  # FinBERT fallback (no API key needed)
        self._last_sentiment_score: float = 0.5  # neutral default
        self._last_sentiment_ts: Optional[float] = None

        # ── Trade outcome recording (self-learning feedback loop) ──
        self._trade_outcome_repo = None  # set via set_trade_outcome_repo()
        self._current_regime: str = "unknown"
        self._sentiment_cache_ttl: float = 300.0  # refresh sentiment every 5 min
        self._last_sentiment_detail: Optional[Dict[str, float]] = None  # {positive, negative, neutral}

        # ── Dynamic universe fallback (when BarCache is empty) ──
        self._dynamic_universe_cache: List[Tuple[str, Exchange]] = []
        self._dynamic_universe_ts: Optional[float] = None
        self._dynamic_universe_ttl: float = 1800.0  # refresh every 30 min

        # ── Startup reconciliation ──
        self._broker_get_positions: Optional[Callable] = None
        self._startup_reconciled: bool = False

        # ── Stuck order detection ──
        self._pending_order_tracker: Dict[str, float] = {}  # idem_key -> submit_timestamp
        self._STUCK_ORDER_TIMEOUT_SECONDS: float = 300.0  # 5 minutes
        self._cancel_order_fn: Optional[Callable] = None

        # ── Broker submission timeout ──
        self._BROKER_SUBMIT_TIMEOUT_SECONDS: float = 30.0
        self._BROKER_SUBMIT_MAX_RETRIES: int = 3

        # ── Idempotency key persistence (DB-backed, survives restarts) ──
        self._position_recovery_manager = None  # set via set_position_recovery_manager()

        # ── Circuit breaker: pause on repeated tick failures ──
        self._consecutive_tick_failures: int = 0
        self._MAX_CONSECUTIVE_FAILURES: int = 5
        self._failure_backoff_seconds: float = 30.0
        self._loop_circuit_open: bool = False
        self._CIRCUIT_OPEN_POLL_SECONDS: float = 60.0

        # ── Bar freshness: detect stale data during market hours ──
        self._consecutive_stale_ticks: int = 0
        self._MAX_STALE_TICKS: int = 3
        self._BAR_FRESHNESS_SECONDS: float = 300.0  # 5 minutes
        self._signal_generation_paused: bool = False

        # ── Signal cache: last tick's signals for API exposure ──
        self._last_signals: List[Signal] = []
        self._last_signals_ts: Optional[str] = None

        # ── Market feed health: consecutive unhealthy tick tracking ──
        self._consecutive_unhealthy_ticks: int = 0
        self._UNHEALTHY_PAUSE_THRESHOLD: int = 3

        # ── Algo execution (TWAP/VWAP) ──
        self._adv_cache = adv_cache
        self._get_market_price = get_market_price
        self._algo_tasks: Dict[str, asyncio.Task] = {}  # idem_key -> background task

    def set_feature_normalizer(self, normalizer) -> None:
        """Wire FeatureNormalizer for z-score normalization of features before model inference."""
        self._feature_normalizer = normalizer
        logger.info("FeatureNormalizer wired to autonomous loop")

    def set_trade_outcome_repo(self, repo) -> None:
        """Wire trade outcome repository for self-learning feedback loop."""
        self._trade_outcome_repo = repo
        logger.info("Trade outcome repo wired to autonomous loop")

    def set_sentiment_service(self, service) -> None:
        """Wire news sentiment service for exposure adjustment."""
        self._sentiment_service = service
        logger.info("Sentiment service wired to autonomous loop")

    def set_sentiment_predictor(self, predictor) -> None:
        """Wire FinBERT sentiment predictor as fallback (no API key needed)."""
        self._sentiment_predictor = predictor
        logger.info("FinBERT sentiment predictor wired to autonomous loop (fallback)")

    async def _fetch_sentiment(self) -> float:
        """Fetch latest market sentiment; returns multiplier 0.5-1.2.

        Two-tier approach:
        1. LLM-based NewsSentimentService (if API key configured)
        2. FinBERT SentimentPredictor fallback (no API key needed)
        """
        import time as _time
        now = _time.time()

        # Cache: only refresh every 5 minutes
        if self._last_sentiment_ts and (now - self._last_sentiment_ts) < self._sentiment_cache_ttl:
            return self._sentiment_to_multiplier(self._last_sentiment_score)

        # Tier 1: LLM-based sentiment (OpenAI / Anthropic)
        if self._sentiment_service is not None:
            try:
                result = await self._sentiment_service.analyze(
                    "Indian stock market overall outlook today", source="autonomous_loop"
                )
                if result:
                    self._last_sentiment_score = result.score
                    self._last_sentiment_ts = now
                    self._last_sentiment_detail = {
                        "positive": result.score,
                        "negative": 1.0 - result.score,
                        "neutral": 0.0,
                        "source": "llm",
                    }
                    mult = self._sentiment_to_multiplier(result.score)
                    logger.info("Sentiment update (LLM): %s (score=%.2f, multiplier=%.2f, suggestion=%s)",
                               result.sentiment, result.score, mult, result.risk_reduction_suggestion)
                    return mult
            except Exception as e:
                logger.debug("LLM sentiment fetch failed: %s — falling back to FinBERT", e)

        # Tier 2: FinBERT-based sentiment (local, no API key)
        if self._sentiment_predictor is not None:
            try:
                _loop = asyncio.get_running_loop()
                prediction = await _loop.run_in_executor(
                    None, lambda: self._sentiment_predictor.predict({}, {"symbol": None})
                )
                if prediction and prediction.confidence > 0.05:
                    # prob_up maps to sentiment score: >0.5 = positive, <0.5 = negative
                    self._last_sentiment_score = prediction.prob_up
                    self._last_sentiment_ts = now
                    meta = prediction.metadata or {}
                    sentiment_breakdown = meta.get("sentiment", {})
                    self._last_sentiment_detail = {
                        "positive": sentiment_breakdown.get("positive", 0.33),
                        "negative": sentiment_breakdown.get("negative", 0.33),
                        "neutral": sentiment_breakdown.get("neutral", 0.34),
                        "source": "finbert",
                        "headlines_count": meta.get("headlines_count", 0),
                    }
                    mult = self._sentiment_to_multiplier(prediction.prob_up)
                    logger.info("Sentiment update (FinBERT): score=%.2f, multiplier=%.2f, headlines=%d, breakdown=%s",
                               prediction.prob_up, mult,
                               meta.get("headlines_count", 0), sentiment_breakdown)
                    return mult
            except Exception as e:
                logger.debug("FinBERT sentiment fetch failed: %s", e)

        return 1.0

    def _sentiment_blocks_buy(self) -> bool:
        """Check if current sentiment is strongly negative enough to block BUY signals."""
        if self._last_sentiment_detail is None:
            return False
        neg = self._last_sentiment_detail.get("negative", 0.33)
        # Block BUY signals when negative sentiment > 60%
        return neg > 0.60

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

    def _get_dynamic_universe_fallback(self) -> List[Tuple[str, Exchange]]:
        """Fallback: build trading universe from DynamicUniverse or YFinance feeder symbols
        when BarCache has no symbols yet."""
        import time as _time
        now = _time.time()

        # Return cached universe if fresh
        if (self._dynamic_universe_cache
                and self._dynamic_universe_ts
                and (now - self._dynamic_universe_ts) < self._dynamic_universe_ttl):
            return self._dynamic_universe_cache

        universe: List[Tuple[str, Exchange]] = []

        # Strategy 1: DynamicUniverse (scans full NSE market)
        try:
            from src.scanner.dynamic_universe import get_dynamic_universe
            du = get_dynamic_universe()
            symbols = du.get_trading_stocks(count=50)
            if symbols:
                universe = [(s, Exchange.NSE) for s in symbols]
                logger.info("Dynamic universe fallback: %d symbols from full NSE scan", len(universe))
        except Exception as e:
            logger.debug("DynamicUniverse fallback failed: %s", e)

        # Strategy 2: YFinance feeder default symbols (lightweight fallback)
        if not universe:
            try:
                from src.market_data.yfinance_fallback_feeder import DEFAULT_NSE_SYMBOLS
                universe = [
                    (s.replace(".NS", "").replace(".BO", ""), Exchange.NSE)
                    for s in DEFAULT_NSE_SYMBOLS
                ]
                logger.info("YFinance feeder fallback: %d default symbols", len(universe))
            except Exception as e:
                logger.debug("YFinance feeder fallback failed: %s", e)

        if universe:
            self._dynamic_universe_cache = universe
            self._dynamic_universe_ts = now

        return universe

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

    async def _check_bar_freshness(self) -> bool:
        """Check if latest bar data is fresh (within 5 minutes) during market hours.

        Returns True if bars are fresh or if check is not applicable (outside
        market hours, no bar provider, etc.).  Returns False if data is stale.
        """
        # Only enforce freshness during live market hours
        if not _is_nse_market_hours():
            self._consecutive_stale_ticks = 0
            return True

        if self.get_bars is None or self.get_symbols is None:
            return True

        symbols = self.get_symbols() or []
        if not symbols:
            return True

        # Sample the first available symbol to check freshness
        for symbol, exchange in symbols[:3]:
            try:
                bars = self.get_bars(symbol, exchange, "1m", 2)
                if not bars:
                    continue
                latest_bar = bars[-1]
                bar_time = latest_bar.ts if hasattr(latest_bar, "ts") and latest_bar.ts else None
                if bar_time is None:
                    return True  # cannot determine freshness, allow tick

                now_utc = datetime.now(timezone.utc)
                # Normalise bar_time to offset-aware UTC
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)
                age_seconds = (now_utc - bar_time).total_seconds()

                if age_seconds <= self._BAR_FRESHNESS_SECONDS:
                    self._consecutive_stale_ticks = 0
                    return True
                else:
                    self._consecutive_stale_ticks += 1
                    logger.warning(
                        "Stale bar data: %s latest bar is %.0fs old (limit %.0fs), "
                        "consecutive stale ticks: %d/%d",
                        symbol, age_seconds, self._BAR_FRESHNESS_SECONDS,
                        self._consecutive_stale_ticks, self._MAX_STALE_TICKS,
                    )
                    if self._consecutive_stale_ticks >= self._MAX_STALE_TICKS:
                        logger.critical(
                            "Bar data stale for %d consecutive ticks — pausing signal "
                            "generation and triggering safe mode",
                            self._consecutive_stale_ticks,
                        )
                        self._signal_generation_paused = True
                        # Trigger safe mode via the callback so the platform pauses trading
                        safe_mode_setter = getattr(self, "_set_safe_mode", None)
                        if safe_mode_setter is not None:
                            safe_mode_setter(True)
                        await self._broadcast({
                            "type": "circuit_breaker",
                            "reason": "stale_bar_data",
                            "consecutive_stale_ticks": self._consecutive_stale_ticks,
                            "last_bar_age_seconds": round(age_seconds, 1),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    return False
            except Exception as e:
                logger.debug("Bar freshness check failed for %s: %s", symbol, e)
                continue

        # Could not check any symbol — allow tick
        return True

    def _reset_daily_pnl_if_new_day(self) -> None:
        """Reset daily P&L at the start of each new trading day (IST).
        Delegates to RiskManager as the single source of truth."""
        today_ist = datetime.now(_IST).strftime("%Y-%m-%d")
        if self._current_trading_date != today_ist:
            if self._current_trading_date is not None:
                daily_pnl = self._get_daily_pnl()
                logger.info("New trading day %s — resetting daily P&L (was %.2f)", today_ist, daily_pnl)
            self._daily_pnl = 0.0  # reset fallback mirror
            self._signal_generation_paused = False
            self._current_trading_date = today_ist
            # Reset RiskManager's daily_pnl via callback (single source of truth)
            if self.on_daily_reset:
                try:
                    self.on_daily_reset()
                    logger.info("RiskManager daily P&L reset for new day")
                except Exception as e:
                    logger.warning("on_daily_reset callback failed: %s", e)

    def _get_daily_pnl(self) -> float:
        """Get daily P&L from RiskManager (single source of truth).
        Falls back to local _daily_pnl mirror if RiskManager is unavailable."""
        _rm = getattr(self, '_risk_manager', None)
        if _rm is not None:
            return _rm.daily_pnl
        # Fallback: try get_risk_state
        if self.get_risk_state:
            try:
                rs = self.get_risk_state()
                return rs.get("daily_pnl", self._daily_pnl)
            except Exception:
                pass
        return self._daily_pnl

    def _register_pnl(self, pnl: float) -> None:
        """Register realised P&L through RiskManager (single source of truth).
        Also updates the local _daily_pnl mirror for backwards compatibility."""
        self._daily_pnl += pnl  # keep fallback mirror in sync
        _rm = getattr(self, '_risk_manager', None)
        if _rm is not None:
            _rm.register_pnl(pnl)
        else:
            logger.debug(
                "PnL registered locally only (no RiskManager ref): pnl=%.2f total=%.2f",
                pnl, self._daily_pnl,
            )

    def _select_algo(self, symbol: str, qty: int, exchange: str = "NSE") -> str:
        """
        Select execution algorithm based on order size relative to ADV.

        Returns one of: "direct", "twap", "vwap", "vwap_extended".
        Falls back to "direct" if ADV data is unavailable.
        """
        if self._adv_cache is None:
            return "direct"
        try:
            adv = self._adv_cache.get_adv(symbol, exchange)
            if adv is None or adv <= 0:
                return "direct"
            pct_adv = (qty / adv) * 100.0
            if pct_adv < _ADV_DIRECT_MAX_PCT:
                return "direct"
            elif pct_adv < _ADV_TWAP_MAX_PCT:
                logger.info("Algo select: %s qty=%d is %.1f%% ADV → TWAP", symbol, qty, pct_adv)
                return "twap"
            elif pct_adv < _ADV_VWAP_MAX_PCT:
                logger.info("Algo select: %s qty=%d is %.1f%% ADV → VWAP", symbol, qty, pct_adv)
                return "vwap"
            else:
                logger.warning(
                    "Algo select: %s qty=%d is %.1f%% ADV (>10%%) → VWAP extended + ALERT",
                    symbol, qty, pct_adv,
                )
                return "vwap_extended"
        except Exception as e:
            logger.debug("ADV lookup failed for %s, falling back to direct: %s", symbol, e)
            return "direct"

    async def _submit_via_algo(
        self,
        signal: Signal,
        qty: int,
        idem_key: str,
        algo_type: str,
        source: str = "autonomous",
        order_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Launch a TWAP or VWAP algo as a background task.

        The parent order's idempotency key is checked *before* launching so that
        duplicate bar ticks never spawn a second algo for the same signal.
        Child slice orders go through submit_order_fn (OrderEntryService) as normal.
        """
        # Deduplicate: if an algo task for this idem_key is already running, skip
        existing = self._algo_tasks.get(idem_key)
        if existing is not None and not existing.done():
            logger.debug("Algo already running for idem_key=%s, skipping", idem_key)
            return

        exchange_str = getattr(signal.exchange, "value", str(signal.exchange))
        _algo_child_seq = 0  # BUG 30: deterministic child sequence counter

        async def _algo_submit_child(
            symbol: str, side: str, quantity: int,
            order_type: str = "LIMIT", limit_price: float = None,
            exchange: str = "NSE",
        ) -> Optional[str]:
            """Adapter: convert algo child order into OrderEntryRequest through submit_order_fn."""
            nonlocal _algo_child_seq
            child_signal = Signal(
                strategy_id=signal.strategy_id,
                symbol=symbol,
                exchange=signal.exchange,
                side=SignalSide.BUY if side == "BUY" else SignalSide.SELL,
                score=signal.score,
                portfolio_weight=signal.portfolio_weight,
                risk_level=signal.risk_level,
                reason=f"algo_child:{algo_type}:{idem_key[:24]}",
                price=limit_price or signal.price,
                ts=datetime.now(timezone.utc),
            )
            ot = OrderType.LIMIT if order_type == "LIMIT" and limit_price else OrderType.MARKET
            # BUG 30 FIX: Replace non-deterministic id() with deterministic values
            # (slice index from the algo + quantity) so idempotency key is stable
            # across retries and process restarts.
            child_idem = f"{idem_key}::{algo_type}_child_{_algo_child_seq}_{quantity}"
            _algo_child_seq += 1
            child_req = OrderEntryRequest(
                signal=child_signal,
                quantity=quantity,
                order_type=ot,
                limit_price=limit_price,
                idempotency_key=child_idem,
                source=f"{source}_{algo_type}",
                metadata=order_metadata,
            )
            try:
                result = await self.submit_order_fn(child_req)
                if result.success:
                    return result.order_id
                else:
                    logger.warning("Algo child order rejected: %s %s", result.reject_reason, result.reject_detail)
                    return None
            except Exception as e:
                logger.exception("Algo child submit failed: %s", e)
                return None

        async def _run_algo():
            try:
                if algo_type == "twap":
                    algo = TWAPAlgorithm(submit_order_fn=_algo_submit_child)
                    config = TWAPConfig(
                        total_quantity=qty,
                        symbol=signal.symbol,
                        side=signal.side.value,
                        exchange=exchange_str,
                        duration_minutes=30,
                        num_slices=10,
                    )
                    execution = algo.create_schedule(config)
                    await algo.execute(execution, get_market_price=self._get_market_price)
                elif algo_type in ("vwap", "vwap_extended"):
                    algo = VWAPAlgorithm(submit_order_fn=_algo_submit_child)
                    # Extended: use full trading day; standard: 2-hour window
                    if algo_type == "vwap_extended":
                        now_ist = datetime.now(_IST)
                        start_hhmm = f"{now_ist.hour:02d}:{now_ist.minute:02d}"
                        end_hhmm = "15:15"
                    else:
                        now_ist = datetime.now(_IST)
                        start_hhmm = f"{now_ist.hour:02d}:{now_ist.minute:02d}"
                        # 2 hours from now, capped at market close
                        end_hour = min(now_ist.hour + 2, 15)
                        end_min = now_ist.minute if end_hour < 15 else 15
                        end_hhmm = f"{end_hour:02d}:{end_min:02d}"
                    if end_hhmm <= start_hhmm:
                        algo_type = "direct"
                    config = VWAPConfig(
                        total_quantity=qty,
                        symbol=signal.symbol,
                        side=signal.side.value,
                        exchange=exchange_str,
                        start_time=start_hhmm,
                        end_time=end_hhmm,
                    )
                    execution = algo.create_schedule(config)
                    await algo.execute(execution, get_market_price=self._get_market_price)

                # Broadcast algo completion
                await self._broadcast({
                    "type": "algo_execution_complete",
                    "algo": algo_type,
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "total_qty": qty,
                    "idem_key": idem_key,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.exception("Algo %s execution failed for %s: %s", algo_type, signal.symbol, e)
            finally:
                self._algo_tasks.pop(idem_key, None)

        task = asyncio.create_task(_run_algo())
        self._algo_tasks[idem_key] = task
        logger.info(
            "Algo %s launched: %s %s qty=%d idem_key=%s",
            algo_type, signal.side.value, signal.symbol, qty, idem_key,
        )

    async def _tick(self) -> None:
        if self.get_safe_mode():
            return

        # ── Stuck order detection: cancel orders pending > 5 minutes ──
        try:
            await self._detect_stuck_orders()
        except Exception as e:
            logger.debug("Stuck order detection error: %s", e)

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

        if not self._market_feed_ok() and not self.paper_mode:
            self._consecutive_unhealthy_ticks += 1
            if self._consecutive_unhealthy_ticks >= self._UNHEALTHY_PAUSE_THRESHOLD:
                if not self._signal_generation_paused:
                    logger.critical(
                        "Market feed unhealthy for %d consecutive ticks — "
                        "pausing signal generation",
                        self._consecutive_unhealthy_ticks,
                    )
                    self._signal_generation_paused = True
            return
        else:
            if self._consecutive_unhealthy_ticks > 0:
                logger.info(
                    "Market feed recovered after %d consecutive unhealthy ticks",
                    self._consecutive_unhealthy_ticks,
                )
            self._consecutive_unhealthy_ticks = 0
            if self._signal_generation_paused:
                logger.info("Signal generation resumed — market feed healthy again")
                self._signal_generation_paused = False

        if not self._drift_ok() or not self._regime_ok():
            return

        # Check bar data freshness before using bars for signal generation
        if not self.paper_mode and not await self._check_bar_freshness():
            logger.warning("Skipping tick: bar data is stale")
            return

        # Reset daily P&L at start of each new trading day
        self._reset_daily_pnl_if_new_day()

        # Daily loss limit check — enforce the -2% circuit breaker
        # RiskManager is single source of truth for daily PnL
        risk_state = self.get_risk_state() if self.get_risk_state else {}
        equity = risk_state.get("equity") or 100000.0
        risk_daily_pnl = self._get_daily_pnl()
        if equity > 0 and risk_daily_pnl / equity < self._daily_loss_limit:
            loss_pct = (risk_daily_pnl / equity) * 100.0
            logger.critical(
                "DAILY LOSS CIRCUIT BREAKER: daily P&L %.2f (%.2f%% of equity %.0f) "
                "breaches limit %.1f%% — pausing signal generation for remainder of day",
                risk_daily_pnl, loss_pct, equity, self._daily_loss_limit * 100,
            )
            self._signal_generation_paused = True
            await self._broadcast({
                "type": "circuit_breaker",
                "reason": "daily_loss_limit",
                "daily_pnl": round(risk_daily_pnl, 2),
                "loss_pct": round(loss_pct, 2),
                "limit_pct": self._daily_loss_limit * 100,
                "equity": round(equity, 2),
                "message": f"Daily loss {loss_pct:.2f}% exceeds {self._daily_loss_limit * 100:.1f}% limit — trading halted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
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

        # ── Core tick body wrapped for circuit breaker ──
        try:
            await self._tick_core(bar_ts)
            # Success: reset circuit breaker counters
            self._consecutive_tick_failures = 0
            if self._loop_circuit_open:
                logger.info(
                    "Circuit breaker CLOSED — tick succeeded after %d prior failures",
                    self._MAX_CONSECUTIVE_FAILURES,
                )
                await self._broadcast({
                    "type": "circuit_breaker",
                    "reason": "circuit_closed",
                    "message": "Autonomous loop recovered — circuit breaker closed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            self._loop_circuit_open = False
        except Exception as e:
            self._consecutive_tick_failures += 1
            logger.exception(
                "Tick failed (consecutive failures: %d/%d): %s",
                self._consecutive_tick_failures, self._MAX_CONSECUTIVE_FAILURES, e,
            )
            if self._consecutive_tick_failures >= self._MAX_CONSECUTIVE_FAILURES and not self._loop_circuit_open:
                self._loop_circuit_open = True
                logger.critical(
                    "CIRCUIT BREAKER OPEN — %d consecutive tick failures. "
                    "Loop will poll every %.0fs until a tick succeeds.",
                    self._consecutive_tick_failures,
                    self._CIRCUIT_OPEN_POLL_SECONDS,
                )
                await self._broadcast({
                    "type": "circuit_breaker",
                    "reason": "consecutive_failures",
                    "consecutive_failures": self._consecutive_tick_failures,
                    "max_failures": self._MAX_CONSECUTIVE_FAILURES,
                    "message": "Autonomous loop circuit breaker OPEN — repeated tick failures",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            # Backoff sleep on failure (in addition to normal poll interval)
            await asyncio.sleep(self._failure_backoff_seconds)

    async def _tick_core(self, bar_ts: str) -> None:
        """Core tick logic — separated so _tick() can wrap it with circuit breaker.

        Any exception raised here will be caught by _tick() and counted as a
        circuit breaker failure.  On success _tick() resets the failure counter.
        """
        from src.strategy_engine.base import MarketState

        symbols_exchanges = self.get_symbols() or []
        used_fallback_universe = False
        if not symbols_exchanges:
            # Fallback: use dynamic universe or YFinance feeder symbols
            symbols_exchanges = self._get_dynamic_universe_fallback()
            if symbols_exchanges:
                used_fallback_universe = True
                logger.info("Tick %d: BarCache empty, using fallback universe (%d symbols)",
                           self._tick_count, len(symbols_exchanges))
            else:
                # Last resort: jump straight to market scanner if available
                if self.market_scanner is not None:
                    logger.info("Tick %d: no universe available, running market scanner only", self._tick_count)
                    symbols_exchanges = []  # will skip bar-based loop, scanner runs below
                else:
                    logger.info("Tick %d: empty universe and no fallback available, skipping", self._tick_count)
                    return
        all_signals: List[Signal] = []
        regime_scale_from_classifier: Optional[float] = None
        symbol_metadata: Dict[str, dict] = {}
        if self.get_bars is None:
            symbols_exchanges = []
        for symbol, exchange in symbols_exchanges:
            bars = self.get_bars(symbol, exchange, "1m", 100)
            if len(bars) < 20:
                continue
            latest = bars[-1]
            metadata = {}
            symbol_metadata[symbol] = metadata
            if self.feature_engine:
                try:
                    raw_features = self.feature_engine.build_features(bars)
                    # P0-3: Apply z-score normalization before model inference
                    if self._feature_normalizer is not None:
                        raw_features = self._feature_normalizer.normalize(raw_features)
                    metadata["features"] = raw_features
                except Exception as e:
                    logger.debug("Feature build failed %s: %s", symbol, e)
            _sym_regime_scale = None
            if self.regime_classifier and metadata.get("features"):
                try:
                    feats = metadata["features"]
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
            # Strategy runner: exceptions propagate to circuit breaker
            if hasattr(self.strategy_runner, 'run_with_multi_timeframe') and self.get_bars:
                sigs = self.strategy_runner.run_with_multi_timeframe(state, self.get_bars)
            else:
                sigs = self.strategy_runner.run(state)
            all_signals.extend(sigs)

        logger.info("Tick %d: %d raw signals from %d symbols", self._tick_count, len(all_signals), len(symbols_exchanges))

        # ── Ensemble AI Enhancement ──────────────────────────────────────
        # If the ensemble engine is available, use its multi-model consensus
        # (XGBoost + LSTM + Transformer + Sentiment + RL) to re-score strategy
        # signals. Signals aligned with ensemble direction get boosted; signals
        # contradicting ensemble direction get suppressed. This prevents
        # classical strategies from trading against the AI consensus.
        _ensemble = getattr(self, '_ensemble_engine', None)
        if _ensemble is not None and all_signals:
            enhanced_signals: List[Signal] = []
            # Group signals by symbol so we call predict() once per symbol
            _signals_by_symbol: Dict[str, List[Signal]] = {}
            for sig in all_signals:
                _signals_by_symbol.setdefault(sig.symbol, []).append(sig)

            for sym, sym_signals in _signals_by_symbol.items():
                # Get cached features for this symbol (built during bar loop above)
                sym_meta = symbol_metadata.get(sym, {})
                features = sym_meta.get("features")
                if not features:
                    # No features available for this symbol -- pass signals through unmodified
                    enhanced_signals.extend(sym_signals)
                    continue

                # Build context for ensemble prediction
                ensemble_context = {"symbol": sym}

                try:
                    prediction = _ensemble.predict(features, ensemble_context)
                except Exception as e:
                    logger.debug("Ensemble predict failed for %s during signal enhancement: %s", sym, e)
                    enhanced_signals.extend(sym_signals)
                    continue

                if prediction is None:
                    # Ensemble returned None (halted / insufficient agreement) --
                    # suppress ALL signals for this symbol as a safety measure.
                    # When the ensemble cannot form a consensus, it is safer to
                    # not trade than to trade on potentially conflicting signals.
                    logger.info(
                        "Ensemble halt for %s — suppressing %d signals (no model consensus)",
                        sym, len(sym_signals),
                    )
                    continue

                # Ensemble produced a valid prediction. Determine its directional view.
                # prob_up > 0.55 = bullish, prob_up < 0.45 = bearish, else neutral.
                ensemble_bullish = prediction.prob_up > 0.55
                ensemble_bearish = prediction.prob_up < 0.45
                ensemble_confidence = prediction.confidence

                for sig in sym_signals:
                    # Signals from the ml_predictor strategy are already ensemble-derived;
                    # do not double-adjust them.
                    if sig.strategy_id == "ml_predictor":
                        enhanced_signals.append(sig)
                        continue

                    signal_is_buy = sig.side == SignalSide.BUY
                    aligned = (signal_is_buy and ensemble_bullish) or (not signal_is_buy and ensemble_bearish)
                    contradicts = (signal_is_buy and ensemble_bearish) or (not signal_is_buy and ensemble_bullish)

                    if aligned:
                        # Signal aligns with ensemble consensus -- boost score proportionally
                        # to ensemble confidence (max +20% boost, capped at 1.0)
                        boost = min(0.20, ensemble_confidence * 0.25)
                        new_score = min(1.0, sig.score + boost)
                        enhanced_sig = sig.model_copy(update={
                            "score": round(new_score, 4),
                            "metadata": {
                                **sig.metadata,
                                "ensemble_aligned": True,
                                "ensemble_prob_up": round(prediction.prob_up, 4),
                                "ensemble_confidence": round(ensemble_confidence, 4),
                                "original_score": sig.score,
                            },
                        })
                        enhanced_signals.append(enhanced_sig)
                    elif contradicts:
                        # Signal contradicts ensemble consensus -- suppress score proportionally
                        # to ensemble confidence (max -30% reduction, floored at 0.0)
                        penalty = min(0.30, ensemble_confidence * 0.35)
                        new_score = max(0.0, sig.score - penalty)
                        if new_score < 0.15:
                            # Score too low after penalty -- drop the signal entirely
                            logger.debug(
                                "Ensemble suppressed %s %s %s: score %.2f→%.2f (dropped)",
                                sig.strategy_id, sig.side.value, sym, sig.score, new_score,
                            )
                            continue
                        enhanced_sig = sig.model_copy(update={
                            "score": round(new_score, 4),
                            "metadata": {
                                **sig.metadata,
                                "ensemble_aligned": False,
                                "ensemble_prob_up": round(prediction.prob_up, 4),
                                "ensemble_confidence": round(ensemble_confidence, 4),
                                "original_score": sig.score,
                            },
                        })
                        enhanced_signals.append(enhanced_sig)
                    else:
                        # Ensemble is neutral (prob_up between 0.45-0.55) -- pass through
                        # with metadata annotation but no score adjustment
                        enhanced_sig = sig.model_copy(update={
                            "metadata": {
                                **sig.metadata,
                                "ensemble_aligned": None,
                                "ensemble_prob_up": round(prediction.prob_up, 4),
                                "ensemble_confidence": round(ensemble_confidence, 4),
                            },
                        })
                        enhanced_signals.append(enhanced_sig)

            pre_enhance = len(all_signals)
            all_signals = enhanced_signals
            # Re-sort by score descending after enhancement
            all_signals.sort(key=lambda s: s.score, reverse=True)
            logger.info(
                "Ensemble enhancement: %d→%d signals (suppressed %d)",
                pre_enhance, len(all_signals), pre_enhance - len(all_signals),
            )

        risk_state = self.get_risk_state()
        equity = risk_state.get("equity") or 0.0
        if equity <= 0:
            logger.warning("Tick %d: equity=%.2f, skipping allocation (no capital)", self._tick_count, equity)
            return
        exposure_mult = risk_state.get("exposure_multiplier") or 1.0

        # Apply news sentiment to exposure multiplier
        sentiment_mult = await self._fetch_sentiment()
        exposure_mult = round(max(0.5, min(1.5, exposure_mult * sentiment_mult)), 2)

        # ── Sentiment-based signal filtering ──
        # Strongly negative sentiment → skip BUY signals entirely
        if self._sentiment_blocks_buy() and all_signals:
            pre_filter = len(all_signals)
            all_signals = [s for s in all_signals if s.side != SignalSide.BUY]
            filtered_out = pre_filter - len(all_signals)
            if filtered_out > 0:
                logger.info("Sentiment filter: blocked %d BUY signals (negative sentiment %.0f%%)",
                           filtered_out,
                           (self._last_sentiment_detail or {}).get("negative", 0) * 100)

        # Cache signals for API exposure (strategies/signals endpoint)
        self._last_signals = list(all_signals)
        self._last_signals_ts = bar_ts

        max_position_pct = risk_state.get("max_position_pct") or 5.0
        drawdown_scale = risk_state.get("drawdown_scale")
        regime_scale = risk_state.get("regime_scale") or regime_scale_from_classifier
        positions = self.get_positions()
        # Allocator: exceptions propagate to circuit breaker
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
            # Hard safety cap: enforce max_position_pct at submission level
            if signal.price and signal.price > 0 and equity > 0:
                max_qty_cap = int(equity * (max_position_pct / 100.0) / signal.price)
                if qty > max_qty_cap:
                    logger.info("Position cap: %s qty %d→%d (%.1f%%→%.1f%%)", signal.symbol, qty, max_qty_cap, qty * signal.price / equity * 100, max_qty_cap * signal.price / equity * 100)
                    qty = max_qty_cap
                if qty <= 0:
                    continue
            idem_key = stable_idempotency_key(bar_ts, signal.strategy_id, signal.symbol, signal.side.value)
            # 8.3: Attach prediction metadata to orders for outcome tracking
            # BUG 10 FIX: Look up the correct per-symbol metadata instead of using
            # the loop variable which only holds the last iteration's value.
            _sig_metadata = symbol_metadata.get(signal.symbol, {})
            order_metadata = dict(_sig_metadata) if _sig_metadata else {}
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

            # Select execution algorithm based on order size vs ADV
            algo_type = self._select_algo(
                signal.symbol, qty,
                getattr(signal.exchange, "value", str(signal.exchange)),
            )
            order_metadata["algo_type"] = algo_type

            # Feed staleness gate: reject if bar data is stale during market hours
            if _is_nse_market_hours() and self._signal_generation_paused:
                logger.warning("Order blocked: signal generation paused due to stale data (symbol=%s)", signal.symbol)
                continue

            # SEBI-compliant decision logging: log WHY we're placing this order
            decision_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": signal.symbol,
                "side": signal.side.value,
                "quantity": qty,
                "signal_score": signal.score,
                "signal_reason": signal.reason,
                "sentiment_score": self._last_sentiment_score,
                "sentiment_detail": self._last_sentiment_detail,
                "regime": self._current_regime,
                "daily_pnl": self._get_daily_pnl(),
                "paper_mode": self.paper_mode,
            }
            logger.info("DECISION_LOG: %s", json.dumps(decision_log, default=str))

            if algo_type != "direct":
                # Launch algo as background task; idempotency is checked inside _submit_via_algo
                await self._submit_via_algo(
                    signal, qty, idem_key, algo_type,
                    source="autonomous", order_metadata=order_metadata,
                )
                await self._track_open_trade(signal, qty)
            else:
                # Direct order — exceptions propagate to circuit breaker
                req = OrderEntryRequest(
                    signal=signal,
                    quantity=qty,
                    order_type=OrderType.LIMIT,
                    limit_price=signal.price,
                    idempotency_key=idem_key,
                    source="autonomous",
                    metadata=order_metadata,
                )
                result = await self.submit_order_fn(req)
                if result.success:
                    logger.info("Autonomous order submitted order_id=%s strategy=%s symbol=%s side=%s qty=%s", result.order_id, signal.strategy_id, signal.symbol, signal.side.value, qty)
                    await self._track_open_trade(signal, qty)
                else:
                    logger.warning("Autonomous order rejected reason=%s detail=%s", result.reject_reason, result.reject_detail)

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
        # TEMP DISABLED: Scanner blocks tick loop for 300-symbol universe; strategy signals suffice
        if False and self.market_scanner is not None:
            try:
                _loop = asyncio.get_running_loop()

                # If fallback universe, scan those specific symbols; otherwise scan default universe
                _scanner_universe = None
                if used_fallback_universe and symbols_exchanges:
                    _scanner_universe = [f"{s}.NS" for s, _ex in symbols_exchanges]

                scanner_signals = await _loop.run_in_executor(
                    None, lambda: self.market_scanner.scan_to_signals(
                        universe=_scanner_universe, bar_cache=None
                    )
                )

                # Apply sentiment filter to scanner signals too
                if self._sentiment_blocks_buy() and scanner_signals:
                    _pre = len(scanner_signals)
                    scanner_signals = [s for s in scanner_signals if s.side != SignalSide.BUY]
                    if len(scanner_signals) < _pre:
                        logger.info("Sentiment filter: blocked %d scanner BUY signals", _pre - len(scanner_signals))

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
                    # Hard safety cap: enforce max_position_pct at submission level
                    if sig.price and sig.price > 0 and equity > 0:
                        max_qty_cap = int(equity * (max_position_pct / 100.0) / sig.price)
                        if qty > max_qty_cap:
                            logger.info("Scanner position cap: %s qty %d→%d", sig.symbol, qty, max_qty_cap)
                            qty = max_qty_cap
                        if qty <= 0:
                            continue
                    submitted_symbols.add(sig.symbol)
                    idem_key = stable_idempotency_key(bar_ts, sig.strategy_id, sig.symbol, sig.side.value)

                    # Feed staleness gate: reject if bar data is stale during market hours
                    if _is_nse_market_hours() and self._signal_generation_paused:
                        logger.warning("Scanner order blocked: signal generation paused due to stale data (symbol=%s)", sig.symbol)
                        continue

                    # SEBI-compliant decision logging for scanner orders
                    scanner_decision_log = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": sig.symbol,
                        "side": sig.side.value,
                        "quantity": qty,
                        "signal_score": sig.score,
                        "signal_reason": sig.reason,
                        "sentiment_score": self._last_sentiment_score,
                        "sentiment_detail": self._last_sentiment_detail,
                        "regime": self._current_regime,
                        "daily_pnl": self._get_daily_pnl(),
                        "paper_mode": self.paper_mode,
                    }
                    logger.info("DECISION_LOG: %s", json.dumps(scanner_decision_log, default=str))

                    # Select execution algorithm for scanner orders too
                    scanner_algo = self._select_algo(
                        sig.symbol, qty,
                        getattr(sig.exchange, "value", str(sig.exchange)),
                    )
                    if scanner_algo != "direct":
                        await self._submit_via_algo(
                            sig, qty, idem_key, scanner_algo, source="scanner",
                        )
                        await self._track_open_trade(sig, qty)
                    else:
                        req = OrderEntryRequest(
                            signal=sig,
                            quantity=qty,
                            order_type=OrderType.LIMIT,
                            limit_price=sig.price,
                            idempotency_key=idem_key,
                            source="scanner",
                        )
                        try:
                            result = await asyncio.wait_for(self.submit_order_fn(req), timeout=10.0)
                            if result.success:
                                logger.info("Scanner order submitted order_id=%s symbol=%s side=%s qty=%s",
                                            result.order_id, sig.symbol, sig.side.value, qty)
                                await self._track_open_trade(sig, qty)
                            else:
                                logger.debug("Scanner order rejected: %s %s", result.reject_reason, result.reject_detail)
                        except asyncio.TimeoutError:
                            logger.warning("Scanner order timeout for %s (10s) — skipping", sig.symbol)
                        except Exception as e:
                            logger.exception("Scanner submit failed: %s", e)
            except Exception as e:
                logger.warning("Market scanner tick failed: %s", e)

        # Mark-to-market: update unrealized P&L and broadcast
        await self._mark_to_market()

        logger.debug("Autonomous loop tick bar_ts=%s signals=%s allocated=%s open_trades=%s daily_pnl=%.2f",
                     bar_ts, len(all_signals), len(allocated), len(self._open_trades), self._get_daily_pnl())

    def set_kill_switch(self, kill_switch_fn: Callable[[], Awaitable[bool]]) -> None:
        """Set async callable that returns True if kill switch is armed."""
        self._kill_switch_fn = kill_switch_fn

    async def _auto_close_all_positions(self) -> None:
        """
        Emergency close-out: submit reduce-only LIMIT orders for all open positions.
        Rate limited: max 2 close orders per second.
        Uses force_reduce=True to bypass daily loss check, circuit breaker, and rate limiter.
        Uses _open_trades_lock to prevent race conditions across async yields.
        """
        async with self._open_trades_lock:
            if not self._open_trades:
                logger.info("Kill switch auto-close: no open trades to close")
                return

            logger.warning("Kill switch ARMED — initiating auto-close of %d open positions", len(self._open_trades))
            close_count = 0

            # Snapshot keys to iterate safely; check existence before each operation
            trade_keys = list(self._open_trades.keys())

        for trade_key in trade_keys:
            # Rate limit: max 2 per second
            if close_count > 0 and close_count % 2 == 0:
                await asyncio.sleep(1.0)

            async with self._open_trades_lock:
                trade = self._open_trades.get(trade_key)
                if trade is None:
                    continue  # already closed by another coroutine
                # Copy trade data so we can release the lock during I/O
                trade = dict(trade)

            symbol = trade.get("symbol") or trade_key.split(":")[0]

            try:
                # Resolve trade exchange; fall back to NSE if not stored
                trade_exchange = trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE)
                # Get current price for limit order with buffer
                exit_price = trade["entry_price"]  # fallback
                if self.get_bars:
                    bars = self.get_bars(symbol, trade_exchange, "1m", 2)
                    if bars:
                        exit_price = bars[-1].close

                # Apply 0.5% buffer for execution room
                close_side = SignalSide.SELL if trade["side"] == "BUY" else SignalSide.BUY
                if close_side == SignalSide.SELL:
                    limit_price = round(exit_price * 0.995, 2)  # sell slightly below market
                else:
                    limit_price = round(exit_price * 1.005, 2)  # buy slightly above market

                _strat_id = trade.get("strategy_id") or trade.get("strategy", "unknown")
                if not _strat_id:
                    _strat_id = "kill_switch_emergency"
                close_signal = Signal(
                    strategy_id=_strat_id,
                    symbol=symbol,
                    exchange=trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE),
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
                    bar_ts, f"{_strat_id}_kill_close", symbol, close_side.value
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
                    async with self._open_trades_lock:
                        if self._open_trade_repo:
                            self._open_trade_repo.delete_trade(trade_key)
                        self._open_trades.pop(trade_key, None)
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

        async with self._open_trades_lock:
            remaining = len(self._open_trades)
        logger.info("Kill switch auto-close completed: %d orders submitted, %d remaining",
                     close_count, remaining)

    async def _emergency_close_symbol(self, symbol: str, reason: str = "forced_close") -> None:
        """Emergency close a single symbol's position from _open_trades. Used for per-position hard stops."""
        async with self._open_trades_lock:
            matching_keys = [k for k, v in self._open_trades.items() if (v.get("symbol") or k.split(":")[0]) == symbol]
        if not matching_keys:
            logger.info("Forced close: no open trade found for %s (may already be closed)", symbol)
            return

        for trade_key in matching_keys:
            async with self._open_trades_lock:
                trade = self._open_trades.get(trade_key)
                if trade is None:
                    continue  # already closed by another coroutine
                trade = dict(trade)
            try:
                trade_exchange = trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE)
                exit_price = trade["entry_price"]
                if self.get_bars:
                    bars = self.get_bars(symbol, trade_exchange, "1m", 2)
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
                    exchange=trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE),
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
                    async with self._open_trades_lock:
                        if self._open_trade_repo:
                            self._open_trade_repo.delete_trade(trade_key)
                        self._open_trades.pop(trade_key, None)
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

    # ── Startup Reconciliation ──

    def set_broker_get_positions(self, fn: Callable) -> None:
        """Set the broker position query function for startup reconciliation."""
        self._broker_get_positions = fn
        logger.info("Broker position query function wired for startup reconciliation")

    def set_position_recovery_manager(self, manager) -> None:
        """Set the PositionRecoveryManager for DB-backed idempotency keys and position recovery."""
        self._position_recovery_manager = manager
        logger.info("PositionRecoveryManager wired to autonomous loop")

    def set_cancel_order_fn(self, fn: Callable) -> None:
        """Set the order cancellation function for stuck order detection."""
        self._cancel_order_fn = fn
        logger.info("Cancel order function wired for stuck order detection")

    async def startup_reconciliation(self) -> Dict[str, Any]:
        """
        On boot, query broker for open positions and sync with local state.
        Must be called after set_broker_get_positions() and load_open_trades_from_db().

        Returns reconciliation report.
        """
        report: Dict[str, Any] = {
            "success": True,
            "broker_positions": 0,
            "local_trades": len(self._open_trades),
            "mismatches": [],
            "actions": [],
        }

        if self._broker_get_positions is None:
            logger.info("Startup reconciliation skipped: no broker position function set (paper mode)")
            report["skipped"] = True
            return report

        try:
            # Query broker for current positions
            broker_positions = self._broker_get_positions()
            if asyncio.iscoroutine(broker_positions):
                broker_positions = await broker_positions
            if broker_positions is None:
                broker_positions = []

            report["broker_positions"] = len(broker_positions)

            # Build broker position map
            broker_map: Dict[str, Dict[str, Any]] = {}
            for bp in broker_positions:
                symbol = getattr(bp, "symbol", bp.get("symbol", "")) if isinstance(bp, dict) else bp.symbol
                side = getattr(bp, "side", bp.get("side", "BUY")) if isinstance(bp, dict) else getattr(bp.side, "value", str(bp.side))
                if isinstance(side, type) and hasattr(side, "value"):
                    side = side.value
                qty = getattr(bp, "quantity", bp.get("quantity", 0)) if isinstance(bp, dict) else bp.quantity
                key = f"{symbol}:{side}"
                broker_map[key] = {"symbol": symbol, "side": side, "quantity": qty}

            # Build local trade map
            local_map: Dict[str, Dict[str, Any]] = {}
            async with self._open_trades_lock:
                for trade_key, trade in self._open_trades.items():
                    symbol = trade.get("symbol", trade_key.split(":")[0])
                    side = trade.get("side", "BUY")
                    key = f"{symbol}:{side}"
                    local_map[key] = trade

            # Find positions on broker but not tracked locally
            for key, bp in broker_map.items():
                if key not in local_map and bp["quantity"] > 0:
                    report["mismatches"].append({
                        "type": "missing_locally",
                        "key": key,
                        "broker_qty": bp["quantity"],
                    })
                    logger.warning(
                        "Startup reconciliation: broker has position %s (qty=%.1f) "
                        "not tracked locally",
                        key, bp["quantity"],
                    )
                    report["actions"].append(f"WARNING: untracked broker position {key}")

            # Find local trades not on broker (phantom)
            for key, trade in local_map.items():
                if key not in broker_map:
                    report["mismatches"].append({
                        "type": "phantom_local",
                        "key": key,
                        "local_qty": trade.get("qty", 0),
                    })
                    logger.warning(
                        "Startup reconciliation: local trade %s (qty=%.1f) "
                        "NOT found on broker — may need cleanup",
                        key, trade.get("qty", 0),
                    )

            # Check quantity mismatches
            for key in set(local_map.keys()) & set(broker_map.keys()):
                local_qty = local_map[key].get("qty", 0)
                broker_qty = broker_map[key]["quantity"]
                if abs(local_qty - broker_qty) > 0.01:
                    report["mismatches"].append({
                        "type": "quantity_mismatch",
                        "key": key,
                        "local_qty": local_qty,
                        "broker_qty": broker_qty,
                    })
                    logger.warning(
                        "Startup reconciliation: quantity mismatch for %s "
                        "(local=%.1f, broker=%.1f)",
                        key, local_qty, broker_qty,
                    )

            self._startup_reconciled = True
            logger.info(
                "Startup reconciliation complete: broker=%d, local=%d, mismatches=%d",
                report["broker_positions"],
                report["local_trades"],
                len(report["mismatches"]),
            )
        except Exception as e:
            report["success"] = False
            report["error"] = str(e)
            logger.error("Startup reconciliation failed: %s", e)

        return report

    # ── Stuck Order Detection ──

    def track_pending_order(self, idem_key: str) -> None:
        """Track a pending order for stuck detection. Called after order submission."""
        import time as _time
        self._pending_order_tracker[idem_key] = _time.time()

    def mark_order_complete(self, idem_key: str) -> None:
        """Remove an order from stuck tracking (filled, cancelled, or rejected)."""
        self._pending_order_tracker.pop(idem_key, None)

    async def _detect_stuck_orders(self) -> List[str]:
        """
        Detect orders stuck in PENDING/SUBMITTED state for > 5 minutes.
        Cancels stuck orders and returns list of cancelled idem_keys.
        """
        import time as _time
        now = _time.time()
        stuck_keys: List[str] = []

        for idem_key, submit_ts in list(self._pending_order_tracker.items()):
            age = now - submit_ts
            if age > self._STUCK_ORDER_TIMEOUT_SECONDS:
                stuck_keys.append(idem_key)
                logger.warning(
                    "Stuck order detected: idem_key=%s age=%.0fs (>%.0fs threshold)",
                    idem_key[:40], age, self._STUCK_ORDER_TIMEOUT_SECONDS,
                )

        cancelled: List[str] = []
        for idem_key in stuck_keys:
            try:
                if self._cancel_order_fn is not None:
                    cancel_result = self._cancel_order_fn(idem_key)
                    if asyncio.iscoroutine(cancel_result):
                        cancel_result = await cancel_result
                    logger.info("Stuck order cancelled: idem_key=%s", idem_key[:40])
                self._pending_order_tracker.pop(idem_key, None)
                cancelled.append(idem_key)
            except Exception as e:
                logger.warning("Failed to cancel stuck order %s: %s", idem_key[:40], e)

        if cancelled:
            await self._broadcast({
                "type": "stuck_orders_cancelled",
                "count": len(cancelled),
                "idem_keys": [k[:40] for k in cancelled],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return cancelled

    # ── Broker Submission with Timeout and Retry ──

    async def _submit_with_timeout(self, req) -> Optional[Any]:
        """
        Submit an order with a 30-second timeout on the broker API call.
        Retries up to 3 times on timeout.
        Returns OrderEntryResult or None on total failure.
        """
        last_error = None
        for attempt in range(1, self._BROKER_SUBMIT_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self.submit_order_fn(req),
                    timeout=self._BROKER_SUBMIT_TIMEOUT_SECONDS,
                )
                # Track the pending order
                if result.success and hasattr(req, "idempotency_key") and req.idempotency_key:
                    self.track_pending_order(req.idempotency_key)
                return result
            except asyncio.TimeoutError:
                last_error = f"Broker API timeout (attempt {attempt}/{self._BROKER_SUBMIT_MAX_RETRIES})"
                logger.warning(
                    "Broker submit timeout for %s (attempt %d/%d, timeout=%.0fs)",
                    getattr(req, "idempotency_key", "unknown")[:40],
                    attempt, self._BROKER_SUBMIT_MAX_RETRIES,
                    self._BROKER_SUBMIT_TIMEOUT_SECONDS,
                )
                if attempt < self._BROKER_SUBMIT_MAX_RETRIES:
                    # Exponential backoff: 2s, 4s
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                logger.exception(
                    "Broker submit failed for %s (attempt %d): %s",
                    getattr(req, "idempotency_key", "unknown")[:40],
                    attempt, e,
                )
                break  # Non-timeout errors don't retry

        logger.error(
            "Broker submission failed after all retries: %s — last error: %s",
            getattr(req, "idempotency_key", "unknown")[:40],
            last_error,
        )
        await self._broadcast({
            "type": "broker_submit_failed",
            "idem_key": getattr(req, "idempotency_key", "unknown")[:40],
            "error": last_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return None

    # ── DB-backed Idempotency Key Persistence ──

    def persist_idempotency_key(self, key: str) -> bool:
        """Persist an idempotency key to DB so it survives restarts."""
        if self._position_recovery_manager is not None:
            return self._position_recovery_manager.persist_idempotency_key(key)
        return False

    def is_idempotency_key_used_db(self, key: str) -> bool:
        """Check if an idempotency key exists in DB (not just in-memory)."""
        if self._position_recovery_manager is not None:
            return self._position_recovery_manager.is_idempotency_key_used(key)
        return False

    @staticmethod
    def _calc_atr(bars: list, period: int = 14) -> Optional[float]:
        """Calculate Average True Range from recent bars. Returns None if insufficient data."""
        if len(bars) < period + 1:
            return None
        highs = np.array([b.high for b in bars[-(period + 1):]], dtype=float)
        lows = np.array([b.low for b in bars[-(period + 1):]], dtype=float)
        closes = np.array([b.close for b in bars[-(period + 1):]], dtype=float)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        )
        return float(np.mean(tr[-period:]))

    async def _track_open_trade(self, signal: Signal, qty: int) -> None:
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
            except Exception as e:
                logger.debug("ATR calculation failed for %s, using default estimate: %s", signal.symbol, e)

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

        async with self._open_trades_lock:
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
        # Phase 1: Under lock, iterate trades, update trailing stops, build to_close list
        async with self._open_trades_lock:
            for trade_key, trade in list(self._open_trades.items()):
                symbol = trade.get("symbol") or trade_key.split(":")[0]
                try:
                    trade_exchange = trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE)
                    bars = self.get_bars(symbol, trade_exchange, "1m", 5)
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
                        # Delegate PnL to RiskManager (single source of truth)
                        self._register_pnl(pnl)

                        if self.performance_tracker:
                            try:
                                self.performance_tracker.record_fill(trade.get("strategy_id") or "unknown", pnl)
                            except Exception as e:
                                logger.debug("Performance tracker record_fill failed: %s", e)

                        logger.info("%s hit for %s %s: entry=%.2f exit=%.2f PnL=%.2f daily_total=%.2f",
                                    reason, side, symbol, entry, current_price, pnl, self._get_daily_pnl())
                        to_close.append((trade_key, symbol, trade, current_price, reason, pnl))
                except Exception as e:
                    logger.debug("SL/TP check failed for %s: %s", symbol, e)

        # Phase 2: Submit close orders outside the lock (involves await)
        closed_trade_keys = []
        for trade_key, symbol, trade, exit_price, reason, pnl in to_close:
            close_side = SignalSide.SELL if trade["side"] == "BUY" else SignalSide.BUY
            _strat_id = trade.get("strategy_id") or trade.get("strategy", "unknown")
            if not _strat_id:
                _strat_id = "sl_tp_close"
            close_signal = Signal(
                strategy_id=_strat_id,
                symbol=symbol,
                exchange=trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE),
                side=close_side,
                score=1.0,
                portfolio_weight=0.0,
                risk_level="NORMAL",
                reason=f"{reason}: entry={trade['entry_price']:.2f} exit={exit_price:.2f} pnl={pnl:.2f}",
                price=exit_price,
                ts=datetime.now(timezone.utc),
            )
            idem_key = stable_idempotency_key(
                bar_ts, f"{_strat_id}_{reason.lower()}", symbol, close_side.value
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

            # Record trade outcome for self-learning feedback loop
            if self._trade_outcome_repo:
                try:
                    self._trade_outcome_repo.record_outcome(
                        trade_key=trade_key,
                        symbol=symbol,
                        side=trade["side"],
                        quantity=trade["qty"],
                        entry_price=trade["entry_price"],
                        exit_price=exit_price,
                        strategy_id=trade.get("strategy_id", ""),
                        model_id=trade.get("model_id", ""),
                        signal_confidence=trade.get("confidence", 0.0),
                        signal_score=trade.get("score", 0.0),
                        regime_at_entry=trade.get("regime", self._current_regime),
                        features_at_entry=trade.get("features"),
                        entry_time=trade.get("opened_at"),
                        exit_reason=reason.lower(),
                        holding_bars=trade.get("bars_held", 0),
                    )
                except Exception as e:
                    logger.warning("Trade outcome recording failed for %s: %s", trade_key, e)

            # Remove from DB (write-ahead: DB before in-memory)
            if self._open_trade_repo:
                self._open_trade_repo.delete_trade(trade_key)
            closed_trade_keys.append(trade_key)

        # Phase 3: Under lock, remove closed trades from in-memory dict
        if closed_trade_keys:
            async with self._open_trades_lock:
                for trade_key in closed_trade_keys:
                    self._open_trades.pop(trade_key, None)

    async def _mark_to_market(self) -> None:
        """Update unrealized P&L for all open trades and broadcast to dashboard."""
        if not self._open_trades or not self.get_bars:
            return

        total_unrealized = 0.0
        open_positions = []
        for trade_key, trade in list(self._open_trades.items()):
            symbol = trade.get("symbol") or trade_key.split(":")[0]
            try:
                trade_exchange = trade.get("exchange", Exchange.NSE) if isinstance(trade, dict) else getattr(trade, 'exchange', Exchange.NSE)
                bars = self.get_bars(symbol, trade_exchange, "1m", 2)
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
            except Exception as e:
                logger.debug("Mark-to-market price fetch failed for %s: %s", trade_key, e)

        if open_positions:
            await self._broadcast({
                "type": "portfolio_mark_to_market",
                "open_positions": open_positions,
                "total_unrealized_pnl": round(total_unrealized, 2),
                "total_realized_pnl": round(self._get_daily_pnl(), 2),
                "total_pnl": round(self._get_daily_pnl() + total_unrealized, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    async def _run_preflight_checks(self) -> bool:
        """Run pre-flight checks before the first tick.

        Returns True if all critical checks pass (or paper mode allows degraded start).
        In live mode (paper_mode=False), any critical failure aborts the loop.
        """
        results: Dict[str, bool] = {}
        is_live = not self.paper_mode

        # 0. NSE holiday calendar staleness check
        holidays = _load_nse_holidays()
        max_holiday_year = max((y for y, _m, _d in holidays), default=0)
        current_year = datetime.now(_IST).year
        if current_year > max_holiday_year:
            logger.warning(
                "NSE holiday calendar is stale (last year: %d, current: %d) "
                "— update from NSE circular",
                max_holiday_year,
                current_year,
            )

        # 1. Database connectivity
        try:
            from src.core.config import get_settings
            db_url = get_settings().database_url
            if db_url:
                from sqlalchemy import create_engine, text
                engine = create_engine(db_url, pool_pre_ping=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                engine.dispose()
            results["database"] = True
            logger.info("Pre-flight check: database ............. PASS")
        except Exception as e:
            results["database"] = False
            logger.error("Pre-flight check: database ............. FAIL (%s)", e)

        # 2. Market data health
        try:
            if self.get_market_feed_healthy is not None:
                healthy = self.get_market_feed_healthy()
                results["market_data"] = bool(healthy)
            else:
                results["market_data"] = True  # no feed configured, skip
            status = "PASS" if results["market_data"] else "FAIL"
            logger.info("Pre-flight check: market_data .......... %s", status)
        except Exception as e:
            results["market_data"] = False
            logger.error("Pre-flight check: market_data .......... FAIL (%s)", e)

        # 3. Model loaded
        try:
            has_models = (
                self.strategy_runner is not None
                and hasattr(self.strategy_runner, "models")
                and self.strategy_runner.models
            )
            # Fallback: check if strategy_runner itself exists (it is the model)
            if not has_models and self.strategy_runner is not None:
                has_models = True
            results["model_loaded"] = bool(has_models)
            status = "PASS" if results["model_loaded"] else "FAIL"
            logger.info("Pre-flight check: model_loaded ......... %s", status)
        except Exception as e:
            results["model_loaded"] = False
            logger.error("Pre-flight check: model_loaded ......... FAIL (%s)", e)

        # 4. Broker session (only meaningful in live mode)
        try:
            if is_live:
                # Check if gateway is available via submit_order_fn's underlying service
                broker_ok = self.submit_order_fn is not None
                if broker_ok and self.get_positions is not None:
                    # Actually probe the broker to verify connectivity (not just that the fn exists)
                    try:
                        positions = self.get_positions()
                        # get_positions may return a coroutine or a list
                        if asyncio.iscoroutine(positions):
                            positions = await positions
                        # If we got here without exception, broker is reachable
                        logger.info("Pre-flight broker probe: got %d positions", len(positions) if positions else 0)
                    except Exception as probe_err:
                        logger.error("Pre-flight broker probe failed: %s", probe_err)
                        broker_ok = False
                results["broker_session"] = bool(broker_ok)
            else:
                results["broker_session"] = True  # paper mode, always OK
            status = "PASS" if results["broker_session"] else "FAIL"
            logger.info("Pre-flight check: broker_session ....... %s", status)
        except Exception as e:
            results["broker_session"] = False
            logger.error("Pre-flight check: broker_session ....... FAIL (%s)", e)

        # Evaluate results
        critical_failures = [k for k, v in results.items() if not v]
        if critical_failures:
            if is_live:
                logger.critical(
                    "LIVE MODE PRE-FLIGHT FAILED — refusing to start. "
                    "Failed checks: %s",
                    ", ".join(critical_failures),
                )
                return False
            else:
                logger.warning(
                    "Paper mode pre-flight has warnings (continuing anyway). "
                    "Failed checks: %s",
                    ", ".join(critical_failures),
                )
                return True
        else:
            logger.info("All pre-flight checks passed — starting autonomous loop")
            return True

    async def _loop(self) -> None:
        # Run pre-flight checks before the first tick
        preflight_ok = await self._run_preflight_checks()
        if not preflight_ok:
            logger.critical("Autonomous loop aborted due to pre-flight check failure")
            self._running = False
            return

        # ── Startup reconciliation: sync local state with broker ──
        if not self._startup_reconciled:
            try:
                recon_report = await self.startup_reconciliation()
                if recon_report.get("mismatches"):
                    logger.warning(
                        "Startup reconciliation found %d mismatches — "
                        "review before trading",
                        len(recon_report["mismatches"]),
                    )
                    await self._broadcast({
                        "type": "startup_reconciliation",
                        "mismatches": len(recon_report["mismatches"]),
                        "broker_positions": recon_report.get("broker_positions", 0),
                        "local_trades": recon_report.get("local_trades", 0),
                    })
            except Exception as e:
                logger.warning("Startup reconciliation error: %s", e)

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Autonomous loop error: %s", e)
            # When circuit breaker is open, slow down to 1 tick per CIRCUIT_OPEN_POLL_SECONDS
            if self._loop_circuit_open:
                await asyncio.sleep(self._CIRCUIT_OPEN_POLL_SECONDS)
            else:
                await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Reset circuit breaker counters on restart
        self._consecutive_tick_failures = 0
        self._loop_circuit_open = False
        self._consecutive_stale_ticks = 0
        self._signal_generation_paused = False
        self._consecutive_unhealthy_ticks = 0
        self._task = asyncio.create_task(self._loop())
        logger.info("Autonomous loop started (poll_interval=%.1fs)", self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        # Cancel all running algo background tasks
        for idem_key, task in list(self._algo_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._algo_tasks.clear()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

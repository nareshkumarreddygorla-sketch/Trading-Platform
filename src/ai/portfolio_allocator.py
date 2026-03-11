"""
AI portfolio allocator: adaptive position sizing using calibrated Kelly criterion,
trade outcome feedback, volatility targeting, correlation-aware sizing,
drawdown scaling, liquidity-aware sizing, and sector caps.

Ranks signals, limits concurrent trades, applies volatility scaling, exposure multiplier,
per-strategy and sector caps. Respects RiskManager.can_place_order() for every candidate.
Returns SizedSignal (signal, quantity). Does NOT call broker.
All execution goes through OrderEntryService.
"""

import logging
import math
from dataclasses import dataclass, field

from src.ai.position_sizing.sizing import (
    SizingConfig,
    dynamic_position_fraction,
    kelly_binary,
)
from src.core.events import Position, Signal
from src.risk_engine import RiskManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults for calibrated Kelly
# ---------------------------------------------------------------------------
_DEFAULT_WIN_RATE = 0.45  # P1-4: conservative default (was 0.50)
_DEFAULT_WIN_LOSS_RATIO = 1.0  # P1-4: conservative default (was 1.5)
_MIN_TRADES_FOR_CALIBRATION = 50  # Minimum 50 trades for statistically significant win rate (95% CI ±14%)
_UNCALIBRATED_MAX_POSITION_PCT = 2.0  # P1-4: cap at 2% when uncalibrated (<50 trades)
_KELLY_SAFETY_FRACTION = 0.5  # half-Kelly
_VOL_TARGET_ANNUAL = 0.15
_DEFAULT_DAILY_VOL = 0.025  # 2.5% daily vol when unknown
_CORRELATION_PENALTY_THRESHOLD = 0.50
_MAX_DRAWDOWN_REDUCTION = 0.50  # at max drawdown, reduce sizing by 50%

# ---------------------------------------------------------------------------
# Liquidity-aware sizing constants
# ---------------------------------------------------------------------------
_MAX_ADV_FRACTION = 0.05  # 5% of Average Daily Volume
_MARKET_IMPACT_COEFF = 0.1  # impact = coeff * sqrt(order_size / ADV)
_DEFAULT_SECTOR_CAP_PCT = 30.0  # no more than 30% of portfolio in any sector


@dataclass
class StrategyStats:
    """Calibrated performance statistics for a single strategy."""

    win_rate: float = _DEFAULT_WIN_RATE
    win_loss_ratio: float = _DEFAULT_WIN_LOSS_RATIO
    trade_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    kelly_fraction: float = 0.0
    calibrated: bool = False

    def __post_init__(self):
        self.kelly_fraction = self._compute_kelly()

    def _compute_kelly(self) -> float:
        raw = kelly_binary(self.win_rate, self.win_loss_ratio)
        return raw * _KELLY_SAFETY_FRACTION


@dataclass
class SizedSignal:
    """Signal with allocated quantity. Safe to pass to order entry."""

    signal: Signal
    quantity: int


@dataclass
class AllocationDiagnostics:
    """Optional diagnostics attached to each allocation pass for observability."""

    total_signals: int = 0
    candidates_after_confidence: int = 0
    portfolio_vol_scale: float = 1.0
    drawdown_scale: float = 1.0
    strategy_kelly: dict[str, float] = field(default_factory=dict)
    correlation_penalties: dict[str, float] = field(default_factory=dict)
    sized_count: int = 0


class PortfolioAllocator:
    """
    Rank signals, apply max concurrent trades, volatility scaling, exposure multiplier,
    per-strategy cap, sector cap. Filter every (signal, qty) through RiskManager.can_place_order().
    Returns list of SizedSignal only. Does not call broker.

    Enhanced with:
      - Calibrated Kelly criterion from actual trade outcomes
      - Volatility targeting (~15% annualized portfolio vol)
      - Correlation-aware sizing (penalise correlated new positions)
      - Drawdown scaling (progressive reduction during drawdowns)
      - Full backwards compatibility when no trade history exists
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        max_concurrent_trades: int = 5,
        max_capital_pct_per_signal: float = 10.0,
        per_strategy_cap_pct: dict | None = None,
        min_confidence: float = 0.5,
        volatility_scale: float = 1.0,
        exposure_multiplier: float = 1.0,
        # --- new adaptive sizing knobs ---
        trade_outcome_repo=None,
        correlation_guard=None,
        calibration_lookback_days: int = 90,
        vol_target_annual: float = _VOL_TARGET_ANNUAL,
        max_drawdown_pct: float = 10.0,
        max_drawdown_reduction: float = _MAX_DRAWDOWN_REDUCTION,
        correlation_penalty_threshold: float = _CORRELATION_PENALTY_THRESHOLD,
        sizing_config: SizingConfig | None = None,
        # --- liquidity-aware sizing ---
        adv_cache=None,
        sector_classifier=None,
        max_adv_fraction: float = _MAX_ADV_FRACTION,
        market_impact_coeff: float = _MARKET_IMPACT_COEFF,
        sector_cap_pct: float = _DEFAULT_SECTOR_CAP_PCT,
    ):
        self.risk_manager = risk_manager
        self.max_concurrent_trades = max_concurrent_trades
        self.max_capital_pct_per_signal = max_capital_pct_per_signal
        self.per_strategy_cap_pct = per_strategy_cap_pct or {}
        self.min_confidence = min_confidence
        self.volatility_scale = max(0.0, min(1.0, volatility_scale))
        self.exposure_multiplier = max(0.0, min(1.0, exposure_multiplier))

        # Adaptive sizing components (all optional for backwards compatibility)
        self._trade_outcome_repo = trade_outcome_repo
        self._correlation_guard = correlation_guard
        self._calibration_lookback_days = calibration_lookback_days
        self._vol_target_annual = vol_target_annual
        self._max_drawdown_pct = max(max_drawdown_pct, 0.01)
        self._max_drawdown_reduction = max(0.0, min(1.0, max_drawdown_reduction))
        self._correlation_penalty_threshold = correlation_penalty_threshold
        self._sizing_config = sizing_config or SizingConfig(
            sigma_target_annual=vol_target_annual,
            max_dd_pct=max_drawdown_pct,
            drawdown_scale_alpha=max_drawdown_reduction,
        )

        # Liquidity-aware sizing
        self._adv_cache = adv_cache  # must expose .get_adv(symbol, exchange) -> float | None
        self._sector_classifier = sector_classifier  # SectorClassifier from risk_engine.sector_map
        self._max_adv_fraction = max_adv_fraction
        self._market_impact_coeff = market_impact_coeff
        self._sector_cap_pct = sector_cap_pct

        # Calibrated per-strategy statistics
        self._strategy_stats: dict[str, StrategyStats] = {}
        self._peak_equity: float = 0.0
        self._last_diagnostics: AllocationDiagnostics | None = None

        # Calibrate on init if repo is available
        if self._trade_outcome_repo is not None:
            self._calibrate_from_history()

    # ------------------------------------------------------------------
    # Calibration from trade outcomes
    # ------------------------------------------------------------------

    def _calibrate_from_history(self) -> None:
        """
        Load historical trade outcomes and compute per-strategy win rate
        and win/loss ratio. Only overwrites defaults when sufficient data
        exists (>= _MIN_TRADES_FOR_CALIBRATION).
        """
        if self._trade_outcome_repo is None:
            return

        try:
            outcomes = self._trade_outcome_repo.get_recent_outcomes(
                limit=5000,
                lookback_days=self._calibration_lookback_days,
            )
        except Exception:
            logger.warning(
                "Failed to load trade outcomes for Kelly calibration; falling back to defaults",
                exc_info=True,
            )
            return

        if not outcomes:
            logger.info("No trade outcomes found; using default Kelly parameters")
            return

        # Group by strategy_id
        by_strategy: dict[str, list[dict]] = {}
        for o in outcomes:
            sid = o.get("strategy_id", "") or "unknown"
            by_strategy.setdefault(sid, []).append(o)

        for strategy_id, trades in by_strategy.items():
            stats = self._compute_strategy_stats(strategy_id, trades)
            self._strategy_stats[strategy_id] = stats
            if stats.calibrated:
                logger.info(
                    "Calibrated %s: win_rate=%.2f, W/L=%.2f, kelly_f=%.4f (%d trades)",
                    strategy_id,
                    stats.win_rate,
                    stats.win_loss_ratio,
                    stats.kelly_fraction,
                    stats.trade_count,
                )
            else:
                logger.info(
                    "Insufficient history for %s (%d trades); using defaults",
                    strategy_id,
                    stats.trade_count,
                )

    @staticmethod
    def _compute_strategy_stats(
        strategy_id: str,
        trades: list[dict],
    ) -> StrategyStats:
        """
        Compute calibrated stats from a list of trade outcome dicts.
        Returns defaults when insufficient data.
        """
        if not trades:
            return StrategyStats()

        wins = []
        losses = []
        for t in trades:
            pnl = t.get("realized_pnl", 0.0)
            if pnl is None:
                continue
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))
            # pnl == 0 is ignored (scratch trades)

        total = len(wins) + len(losses)
        if total < _MIN_TRADES_FOR_CALIBRATION:
            return StrategyStats(trade_count=total, calibrated=False)

        win_rate = len(wins) / total if total > 0 else _DEFAULT_WIN_RATE
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 1.0  # avoid div-by-zero

        # Win/loss ratio: average profit per winning trade / average loss per losing trade
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else _DEFAULT_WIN_LOSS_RATIO

        # Bound to reasonable range to prevent extreme Kelly fractions
        win_loss_ratio = max(0.1, min(10.0, win_loss_ratio))
        win_rate = max(0.05, min(0.95, win_rate))

        return StrategyStats(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            trade_count=total,
            avg_win=avg_win,
            avg_loss=avg_loss,
            calibrated=True,
        )

    def recalibrate(self) -> None:
        """Public method to force recalibration from latest trade outcomes."""
        self._strategy_stats.clear()
        self._calibrate_from_history()

    # ------------------------------------------------------------------
    # Drawdown scaling
    # ------------------------------------------------------------------

    def _compute_drawdown_scale(
        self,
        equity: float,
        drawdown_scale_override: float | None,
    ) -> float:
        """
        Compute drawdown scaling factor.

        If the caller provides an explicit drawdown_scale, honour it.
        Otherwise compute from peak equity tracking:
          scale = max(1 - max_dd_reduction * (dd_pct / max_dd_pct), 1 - max_dd_reduction)
        Clamped to [1 - max_dd_reduction, 1.0].
        """
        if drawdown_scale_override is not None:
            return max(0.0, min(1.0, drawdown_scale_override))

        # Track peak equity (initialize to first observed equity to avoid reset-on-restart issues)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity <= 0:
            return 1.0

        dd_pct = ((self._peak_equity - equity) / self._peak_equity) * 100.0
        if dd_pct <= 0:
            return 1.0

        # Linear reduction: at max_drawdown_pct, reduce by max_drawdown_reduction
        reduction = self._max_drawdown_reduction * min(dd_pct / self._max_drawdown_pct, 1.0)
        return max(1.0 - self._max_drawdown_reduction, 1.0 - reduction)

    # ------------------------------------------------------------------
    # Volatility targeting
    # ------------------------------------------------------------------

    def _compute_vol_target_scale(
        self,
        positions: list[Position],
        equity: float,
    ) -> float:
        """
        Scale total exposure so that the portfolio's estimated annualized
        volatility stays near the vol target.

        If no CorrelationGuard is available, returns 1.0 (no vol targeting).
        """
        if self._correlation_guard is None or equity <= 0:
            return 1.0

        if not positions:
            return 1.0

        symbols = [p.symbol for p in positions]
        notionals = [abs(p.quantity * p.avg_price) for p in positions]
        total_notional = sum(notionals)

        if total_notional <= 0:
            return 1.0

        try:
            # Use public method pattern to avoid accessing private method of another class
            if hasattr(self._correlation_guard, "_estimate_portfolio_vol"):
                port_daily_vol = self._correlation_guard._estimate_portfolio_vol(
                    symbols,
                    notionals,
                    total_notional,
                )
            else:
                return 1.0
        except Exception:
            logger.debug("Vol targeting: failed to estimate portfolio vol", exc_info=True)
            return 1.0

        if port_daily_vol <= 0:
            return 1.0

        port_annual_vol = port_daily_vol * math.sqrt(252)
        if port_annual_vol <= 0:
            return 1.0

        # Scale to hit target: if current vol is 20% and target is 15%, scale = 0.75
        raw_scale = self._vol_target_annual / port_annual_vol
        # Clamp: don't lever up more than 1.5x, don't go below 0.2x
        return max(0.2, min(1.5, raw_scale))

    # ------------------------------------------------------------------
    # Correlation-aware sizing
    # ------------------------------------------------------------------

    def _correlation_penalty(
        self,
        symbol: str,
        existing_symbols: list[str],
    ) -> float:
        """
        Compute a sizing penalty in [0, 1] based on the maximum pairwise
        correlation between `symbol` and existing positions.

        Returns 1.0 (no penalty) if correlation is below threshold.
        Linearly reduces to 0.5 as correlation approaches 1.0.
        """
        if self._correlation_guard is None or not existing_symbols:
            return 1.0

        max_abs_corr = 0.0
        for existing_sym in existing_symbols:
            try:
                corr = self._correlation_guard.pairwise_correlation(symbol, existing_sym)
                max_abs_corr = max(max_abs_corr, abs(corr))
            except Exception:
                pass

        threshold = self._correlation_penalty_threshold
        if max_abs_corr <= threshold:
            return 1.0

        # Linear penalty from 1.0 at threshold to 0.5 at correlation=1.0
        penalty_range = 1.0 - threshold
        if penalty_range <= 0:
            return 1.0
        excess = max_abs_corr - threshold
        penalty = 1.0 - 0.5 * (excess / penalty_range)
        return max(0.5, min(1.0, penalty))

    # ------------------------------------------------------------------
    # Kelly-based position fraction
    # ------------------------------------------------------------------

    def _kelly_position_fraction(
        self,
        signal: Signal,
        current_drawdown_pct: float,
        regime_multiplier: float,
    ) -> float | None:
        """
        Compute Kelly-based position fraction using calibrated strategy stats.

        Returns None if we should fall back to the legacy capital-percentage method
        (e.g. no calibrated stats, Kelly yields 0).
        """
        stats = self._strategy_stats.get(signal.strategy_id)
        if stats is None or not stats.calibrated:
            return None

        fraction = dynamic_position_fraction(
            p_win=stats.win_rate,
            win_loss_ratio=stats.win_loss_ratio,
            confidence=signal.score,
            current_drawdown_pct=current_drawdown_pct,
            regime_multiplier=regime_multiplier,
            config=self._sizing_config,
        )
        if fraction <= 0:
            return None

        return fraction

    # ------------------------------------------------------------------
    # Liquidity-aware sizing
    # ------------------------------------------------------------------

    def _adv_cap_quantity(
        self,
        symbol: str,
        raw_qty: int,
        price: float,
        exchange: str = "NSE",
    ) -> int:
        """
        Cap quantity to min(raw_qty, max_adv_fraction * ADV).
        Returns adjusted quantity.  If no ADV data is available, returns raw_qty unchanged.
        """
        if self._adv_cache is None or price <= 0:
            return raw_qty
        try:
            adv = self._adv_cache.get_adv(symbol, exchange)
            if adv is None or adv <= 0:
                return raw_qty
            max_qty_by_adv = int(self._max_adv_fraction * adv)
            if max_qty_by_adv <= 0:
                logger.warning(
                    "ADV cap: %s ADV=%.0f too low, max_qty=0 — skipping",
                    symbol,
                    adv,
                )
                return 0
            if raw_qty > max_qty_by_adv:
                logger.info(
                    "ADV cap: %s qty %d -> %d (%.1f%% of ADV %.0f, limit %.0f%%)",
                    symbol,
                    raw_qty,
                    max_qty_by_adv,
                    (raw_qty / adv) * 100,
                    adv,
                    self._max_adv_fraction * 100,
                )
            return min(raw_qty, max_qty_by_adv)
        except Exception as e:
            logger.debug("ADV check failed for %s: %s — using raw qty", symbol, e)
            return raw_qty

    def _market_impact_cost(
        self,
        symbol: str,
        order_size: int,
        exchange: str = "NSE",
    ) -> float:
        """
        Estimate market impact cost as a fraction of price.
        impact = coeff * sqrt(order_size / ADV)
        Returns 0.0 if no ADV data available.
        """
        if self._adv_cache is None or order_size <= 0:
            return 0.0
        try:
            adv = self._adv_cache.get_adv(symbol, exchange)
            if adv is None or adv <= 0:
                return 0.0
            impact = self._market_impact_coeff * math.sqrt(order_size / adv)
            return min(impact, 0.05)  # cap at 5% to prevent absurd values
        except Exception:
            return 0.0

    def _check_sector_cap(
        self,
        symbol: str,
        notional: float,
        positions: list[Position],
        equity: float,
    ) -> bool:
        """
        Check if adding this position would breach the sector cap.
        Returns True if allowed, False if it would breach the cap.
        """
        if self._sector_classifier is None or equity <= 0:
            return True
        try:
            sector = self._sector_classifier.get_sector(symbol)
            # Compute current sector notional
            sector_notional = 0.0
            for p in positions:
                if self._sector_classifier.get_sector(p.symbol) == sector:
                    sector_notional += abs(p.quantity * p.avg_price)
            projected_pct = ((sector_notional + notional) / equity) * 100.0
            if projected_pct > self._sector_cap_pct:
                logger.info(
                    "Sector cap: %s sector=%s projected=%.1f%% > cap=%.1f%% — skipping",
                    symbol,
                    sector,
                    projected_pct,
                    self._sector_cap_pct,
                )
                return False
            return True
        except Exception as e:
            logger.debug("Sector cap check failed for %s: %s — allowing", symbol, e)
            return True

    # ------------------------------------------------------------------
    # Main allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        signals: list[Signal],
        equity: float,
        positions: list[Position],
        *,
        exposure_multiplier: float | None = None,
        drawdown_scale: float | None = None,
        regime_scale: float | None = None,
        max_position_pct: float | None = None,
        volatility_scale: float | None = None,
    ) -> list[SizedSignal]:
        """
        Rank signals (by score), cap count, allocate capital, apply scaling.
        Each (signal, qty) is checked with risk_manager.can_place_order(); only allowed ones returned.

        Adaptive sizing layers (applied when historical data is available):
          1. Calibrated Kelly fraction per strategy (replaces fixed pct when calibrated)
          2. Volatility targeting: scale total exposure to maintain vol target
          3. Correlation penalty: reduce sizing for correlated new positions
          4. Drawdown scaling: progressively reduce during drawdowns
        Falls back to legacy percentage-based sizing when data is insufficient.
        """
        diag = AllocationDiagnostics(total_signals=len(signals))

        if equity <= 0:
            self._last_diagnostics = diag
            return []

        # --- Global scale factors ---
        em = exposure_multiplier if exposure_multiplier is not None else self.exposure_multiplier
        vs = volatility_scale if volatility_scale is not None else self.volatility_scale
        scale = em * vs

        # Regime scale
        regime_mult = 1.0
        if regime_scale is not None:
            regime_mult = max(0.0, min(1.0, regime_scale))
            scale *= regime_mult

        # Drawdown scaling (adaptive)
        dd_scale = self._compute_drawdown_scale(equity, drawdown_scale)
        scale *= dd_scale
        diag.drawdown_scale = dd_scale

        # Volatility targeting
        vol_scale = self._compute_vol_target_scale(positions, equity)
        scale *= vol_scale
        diag.portfolio_vol_scale = vol_scale

        scale = max(0.0, min(1.5, scale))  # allow slight leverage from vol targeting

        # --- Filter and rank candidates ---
        candidates = [s for s in signals if s.score >= self.min_confidence]
        diag.candidates_after_confidence = len(candidates)
        if not candidates:
            self._last_diagnostics = diag
            return []

        sorted_sigs = sorted(candidates, key=lambda s: s.score, reverse=True)
        top = sorted_sigs[: self.max_concurrent_trades]

        max_pct = (
            max_position_pct
            if max_position_pct is not None
            else getattr(self.risk_manager.limits, "max_position_pct", 5.0)
        )

        # Track existing position symbols for correlation penalty
        existing_symbols = [p.symbol for p in positions]
        # Also track symbols we're allocating in this batch
        batch_symbols: list[str] = []

        # Current drawdown percentage for Kelly
        current_dd_pct = 0.0
        if self._peak_equity > 0 and equity < self._peak_equity:
            current_dd_pct = ((self._peak_equity - equity) / self._peak_equity) * 100.0

        out: list[SizedSignal] = []
        for signal in top:
            price = signal.price or 0.0
            if price <= 0:
                continue

            # --- Determine notional allocation ---
            # Try calibrated Kelly first
            kelly_f = self._kelly_position_fraction(
                signal,
                current_drawdown_pct=current_dd_pct,
                regime_multiplier=regime_mult,
            )

            # --- Deduct market impact from expected return before Kelly ---
            exchange_str = (
                getattr(signal.exchange, "value", str(signal.exchange)) if hasattr(signal, "exchange") else "NSE"
            )

            if kelly_f is not None:
                # Kelly-based allocation: fraction of equity
                # kelly_f from dynamic_position_fraction already includes drawdown
                # scaling, so divide out dd_scale to avoid applying it twice.
                notional = equity * kelly_f * (scale / dd_scale if dd_scale > 0 else scale)
                diag.strategy_kelly[signal.strategy_id] = kelly_f
            else:
                # Legacy: percentage-based allocation
                # P1-4: cap uncalibrated strategies at 2% equity (was 5%+)
                strategy_cap_pct = self.per_strategy_cap_pct.get(signal.strategy_id) or self.max_capital_pct_per_signal
                strategy_cap_pct = min(strategy_cap_pct, _UNCALIBRATED_MAX_POSITION_PCT)
                notional = equity * (strategy_cap_pct / 100.0) * scale

            # --- Correlation penalty ---
            all_existing = existing_symbols + batch_symbols
            corr_penalty = self._correlation_penalty(signal.symbol, all_existing)
            if corr_penalty < 1.0:
                diag.correlation_penalties[signal.symbol] = corr_penalty
            notional *= corr_penalty

            # --- Hard cap: max_position_pct ---
            max_notional = equity * (max_pct / 100.0)
            notional = min(notional, max_notional)

            if notional <= 0:
                continue

            raw_qty = max(1, int(notional / price))
            max_qty = int(max_notional / price) if price > 0 else 0
            qty = min(raw_qty, max_qty)
            if qty <= 0:
                continue

            # --- ADV (Average Daily Volume) cap ---
            qty = self._adv_cap_quantity(signal.symbol, qty, price, exchange_str)
            if qty <= 0:
                logger.debug("Allocator skip: %s ADV too low for any position", signal.symbol)
                continue

            # --- Market impact cost deduction ---
            impact = self._market_impact_cost(signal.symbol, qty, exchange_str)
            if impact > 0.02:  # if impact > 2%, warn and reduce
                # Reduce quantity to bring impact under 2%
                if self._adv_cache is not None:
                    try:
                        adv = self._adv_cache.get_adv(signal.symbol, exchange_str)
                        if adv and adv > 0:
                            # Solve: coeff * sqrt(q/adv) <= 0.02  =>  q <= adv * (0.02/coeff)^2
                            target_impact = 0.02
                            max_qty_impact = int(adv * (target_impact / self._market_impact_coeff) ** 2)
                            if max_qty_impact > 0 and max_qty_impact < qty:
                                logger.info(
                                    "Market impact: %s impact=%.3f (%.1f%%) — reducing qty %d -> %d",
                                    signal.symbol,
                                    impact,
                                    impact * 100,
                                    qty,
                                    max_qty_impact,
                                )
                                qty = max_qty_impact
                    except Exception:
                        pass
            if qty <= 0:
                continue

            # --- Sector position cap ---
            notional_final = qty * price
            if not self._check_sector_cap(signal.symbol, notional_final, positions, equity):
                logger.debug("Allocator skip: %s breaches sector cap", signal.symbol)
                continue

            # --- Risk manager gate ---
            check = self.risk_manager.can_place_order(signal, qty, price)
            if not check.allowed:
                logger.debug(
                    "Allocator skip: risk can_place_order rejected %s qty=%s: %s",
                    signal.symbol,
                    qty,
                    check.reason,
                )
                continue

            out.append(SizedSignal(signal=signal, quantity=qty))
            batch_symbols.append(signal.symbol)

        diag.sized_count = len(out)
        self._last_diagnostics = diag

        if out:
            logger.info(
                "Allocated %d/%d signals (dd_scale=%.2f, vol_scale=%.2f, scale=%.3f)",
                len(out),
                len(signals),
                dd_scale,
                vol_scale,
                scale,
            )

        return out

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def last_diagnostics(self) -> AllocationDiagnostics | None:
        """Diagnostics from the most recent allocate() call."""
        return self._last_diagnostics

    @property
    def strategy_stats(self) -> dict[str, StrategyStats]:
        """Calibrated per-strategy statistics (read-only view)."""
        return dict(self._strategy_stats)

    def get_strategy_kelly(self, strategy_id: str) -> float:
        """Return the calibrated half-Kelly fraction for a strategy, or 0 if uncalibrated."""
        stats = self._strategy_stats.get(strategy_id)
        if stats is None or not stats.calibrated:
            return 0.0
        return stats.kelly_fraction

"""
TWAP (Time-Weighted Average Price) Execution Algorithm.
Splits large orders into equal time-weighted child orders.

Use case: Orders that are 1-5% of ADV.

Drift circuit breaker:
  - Monitors execution price vs TWAP benchmark (arrival price).
  - WARNING threshold (default 0.5%): pauses TWAP and alerts.
  - CRITICAL threshold (default 1.0%): cancels remaining slices.
  - All drift events logged for post-trade analysis.
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """TWAP drift severity levels."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"     # Drift exceeds warning threshold -> pause
    CRITICAL = "CRITICAL"   # Drift exceeds critical threshold -> cancel


@dataclass
class TWAPDriftEvent:
    """Recorded drift event for post-trade analysis."""
    exec_id: str
    slice_sequence: int
    timestamp: datetime
    arrival_price: float
    execution_price: float
    drift_bps: float
    severity: DriftSeverity
    action_taken: str  # "none", "paused", "cancelled"
    cumulative_vwap: float = 0.0


@dataclass
class TWAPConfig:
    """Configuration for TWAP execution."""
    total_quantity: int
    symbol: str
    side: str  # BUY or SELL
    exchange: str = "NSE"
    duration_minutes: int = 30
    num_slices: int = 10
    limit_offset_bps: float = 5.0  # limit price offset from market in basis points
    max_participation_pct: float = 10.0  # max % of each interval's volume
    randomize_pct: float = 20.0  # randomize slice sizes by +/- this %
    # Drift circuit breaker thresholds (basis points)
    drift_warning_pct: float = 0.5    # 0.5% = 50 bps -> pause and alert
    drift_critical_pct: float = 1.0   # 1.0% = 100 bps -> cancel remaining slices
    drift_check_enabled: bool = True   # Enable/disable drift monitoring


@dataclass
class TWAPSlice:
    """A single child order in the TWAP schedule."""
    slice_id: str
    sequence: int
    quantity: int
    scheduled_time: datetime
    status: str = "PENDING"  # PENDING, SENT, FILLED, FAILED
    order_id: Optional[str] = None
    fill_price: float = 0.0
    fill_qty: int = 0


@dataclass
class TWAPExecution:
    """Tracks the overall TWAP execution."""
    exec_id: str
    config: TWAPConfig
    slices: List[TWAPSlice] = field(default_factory=list)
    status: str = "CREATED"  # CREATED, RUNNING, SUBMITTED, COMPLETED, CANCELLED, DRIFT_PAUSED, DRIFT_CANCELLED
    total_filled: int = 0
    total_submitted: int = 0  # Qty submitted (fills tracked via FillListener)
    vwap: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Drift circuit breaker state
    arrival_price: float = 0.0       # Price at TWAP start (benchmark)
    drift_events: List[TWAPDriftEvent] = field(default_factory=list)
    drift_paused: bool = False
    drift_cancelled: bool = False
    max_drift_bps: float = 0.0       # Maximum observed drift in bps


class TWAPAlgorithm:
    """
    Time-Weighted Average Price execution algorithm.

    Splits a parent order into N equal-sized child orders,
    spaced evenly across the specified duration.

    Includes drift circuit breaker: monitors execution price vs arrival price.
    - Warning threshold: pause TWAP and alert.
    - Critical threshold: cancel all remaining slices.
    """

    def __init__(
        self,
        submit_order_fn: Callable[..., Coroutine],
        on_drift_alert: Optional[Callable] = None,
    ):
        """
        Args:
            submit_order_fn: Async function to submit a single order.
                Signature: (symbol, side, qty, order_type, limit_price, exchange) -> order_id
            on_drift_alert: Optional callback(exec_id, drift_event) for drift notifications.
        """
        self._submit = submit_order_fn
        self._on_drift_alert = on_drift_alert
        self._active_executions: Dict[str, TWAPExecution] = {}

    def create_schedule(self, config: TWAPConfig) -> TWAPExecution:
        """
        Create TWAP execution schedule.

        Returns TWAPExecution with pre-computed slice schedule.
        """
        exec_id = f"twap_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Compute slice sizes
        base_qty = config.total_quantity // config.num_slices
        remainder = config.total_quantity - base_qty * config.num_slices
        interval_seconds = (config.duration_minutes * 60) / config.num_slices

        slices = []
        import random
        for i in range(config.num_slices):
            # Randomize slice size
            randomize = config.randomize_pct / 100
            factor = 1.0 + random.uniform(-randomize, randomize)
            qty = max(1, int(base_qty * factor))

            # BUG 11 FIX: Clamp last slice to remaining quantity to prevent negative qty
            # when randomized slices exceed total.
            if i == config.num_slices - 1:
                filled_so_far = sum(s.quantity for s in slices)
                remaining = config.total_quantity - filled_so_far
                if remaining <= 0:
                    continue  # all quantity already allocated, skip last slice
                qty = max(0, remaining)

            scheduled = now + timedelta(seconds=interval_seconds * i)

            slices.append(TWAPSlice(
                slice_id=f"{exec_id}_s{i:03d}",
                sequence=i,
                quantity=qty,
                scheduled_time=scheduled,
            ))

        execution = TWAPExecution(
            exec_id=exec_id,
            config=config,
            slices=slices,
        )
        self._active_executions[exec_id] = execution

        logger.info(
            "TWAP schedule created: %s %s %d shares in %d slices over %d min",
            config.side, config.symbol, config.total_quantity,
            config.num_slices, config.duration_minutes,
        )
        return execution

    def _check_drift(
        self,
        execution: TWAPExecution,
        current_price: float,
        slice_seq: int,
        cumulative_vwap: float,
    ) -> DriftSeverity:
        """
        Check execution price drift against arrival price benchmark.

        Returns severity level and records drift event.
        """
        config = execution.config
        if not config.drift_check_enabled or execution.arrival_price <= 0:
            return DriftSeverity.NORMAL

        # Drift = signed deviation from arrival price in basis points
        drift_pct = abs(current_price - execution.arrival_price) / execution.arrival_price * 100
        drift_bps = drift_pct * 100  # convert % to bps for logging

        # For BUY orders, price going UP is adverse; for SELL, price going DOWN is adverse
        if config.side == "BUY":
            adverse = current_price > execution.arrival_price
        else:
            adverse = current_price < execution.arrival_price

        # Track max drift
        if drift_bps > execution.max_drift_bps:
            execution.max_drift_bps = drift_bps

        # Only trigger on adverse drift
        if not adverse:
            return DriftSeverity.NORMAL

        severity = DriftSeverity.NORMAL
        action = "none"

        if drift_pct >= config.drift_critical_pct:
            severity = DriftSeverity.CRITICAL
            action = "cancelled"
            execution.drift_cancelled = True
            execution.status = "DRIFT_CANCELLED"
            logger.critical(
                "TWAP DRIFT CRITICAL: %s %s drift=%.1fbps (%.3f%%) > critical=%.1f%% — "
                "CANCELLING remaining slices. arrival=%.2f current=%.2f",
                config.side, config.symbol, drift_bps, drift_pct,
                config.drift_critical_pct, execution.arrival_price, current_price,
            )
        elif drift_pct >= config.drift_warning_pct:
            severity = DriftSeverity.WARNING
            action = "paused"
            execution.drift_paused = True
            execution.status = "DRIFT_PAUSED"
            logger.warning(
                "TWAP DRIFT WARNING: %s %s drift=%.1fbps (%.3f%%) > warning=%.1f%% — "
                "PAUSING TWAP. arrival=%.2f current=%.2f",
                config.side, config.symbol, drift_bps, drift_pct,
                config.drift_warning_pct, execution.arrival_price, current_price,
            )

        drift_event = TWAPDriftEvent(
            exec_id=execution.exec_id,
            slice_sequence=slice_seq,
            timestamp=datetime.now(timezone.utc),
            arrival_price=execution.arrival_price,
            execution_price=current_price,
            drift_bps=drift_bps,
            severity=severity,
            action_taken=action,
            cumulative_vwap=cumulative_vwap,
        )
        execution.drift_events.append(drift_event)

        # Notify external listener if configured
        if self._on_drift_alert and severity != DriftSeverity.NORMAL:
            try:
                self._on_drift_alert(execution.exec_id, drift_event)
            except Exception as e:
                logger.debug("Drift alert callback failed: %s", e)

        return severity

    async def resume(self, exec_id: str) -> bool:
        """Resume a drift-paused TWAP execution."""
        execution = self._active_executions.get(exec_id)
        if execution and execution.drift_paused and execution.status == "DRIFT_PAUSED":
            execution.drift_paused = False
            execution.status = "RUNNING"
            logger.info("TWAP resumed after drift pause: %s", exec_id)
            return True
        return False

    async def execute(
        self,
        execution: TWAPExecution,
        get_market_price: Optional[Callable] = None,
    ) -> TWAPExecution:
        """
        Execute the TWAP schedule with drift circuit breaker.

        Args:
            execution: Pre-created TWAPExecution
            get_market_price: Optional async fn(symbol) -> current price
        """
        config = execution.config
        execution.status = "RUNNING"
        execution.start_time = datetime.now(timezone.utc)
        interval = (config.duration_minutes * 60) / config.num_slices

        total_cost = 0.0
        total_filled = 0
        total_submitted = 0

        # Capture arrival price for drift benchmark
        if get_market_price and config.drift_check_enabled:
            try:
                arrival = await get_market_price(config.symbol)
                if arrival and arrival > 0:
                    execution.arrival_price = arrival
                    logger.info(
                        "TWAP drift benchmark: %s arrival_price=%.2f "
                        "(warning=%.1f%% critical=%.1f%%)",
                        config.symbol, arrival,
                        config.drift_warning_pct, config.drift_critical_pct,
                    )
            except Exception:
                pass

        for slice_order in execution.slices:
            if execution.status == "CANCELLED" or execution.drift_cancelled:
                slice_order.status = "CANCELLED"
                logger.info(
                    "TWAP slice %d skipped (execution %s): %s",
                    slice_order.sequence, execution.status, execution.exec_id,
                )
                continue

            # If drift-paused, wait up to 2 intervals for resume, then cancel
            if execution.drift_paused:
                pause_wait = interval * 2
                waited = 0.0
                check_interval = min(5.0, interval / 2)
                while execution.drift_paused and waited < pause_wait:
                    await asyncio.sleep(check_interval)
                    waited += check_interval
                if execution.drift_paused:
                    # Timed out waiting for resume -> cancel remaining
                    execution.drift_cancelled = True
                    execution.status = "DRIFT_CANCELLED"
                    slice_order.status = "CANCELLED"
                    logger.warning(
                        "TWAP drift pause timeout after %.0fs; cancelling remaining slices: %s",
                        waited, execution.exec_id,
                    )
                    continue

            try:
                # Get current market price for limit calculation
                limit_price = None
                current_market_price = None
                if get_market_price:
                    try:
                        price = await get_market_price(config.symbol)
                        if price and price > 0:
                            current_market_price = price
                            offset = config.limit_offset_bps / 10000
                            if config.side == "BUY":
                                limit_price = round(price * (1 + offset), 2)
                            else:
                                limit_price = round(price * (1 - offset), 2)
                    except Exception:
                        pass

                # Drift circuit breaker check (before submitting slice)
                if current_market_price and config.drift_check_enabled:
                    cumulative_vwap = total_cost / total_filled if total_filled > 0 else 0.0
                    severity = self._check_drift(
                        execution, current_market_price,
                        slice_order.sequence, cumulative_vwap,
                    )
                    if severity == DriftSeverity.CRITICAL:
                        slice_order.status = "CANCELLED"
                        continue
                    elif severity == DriftSeverity.WARNING:
                        # Pause: skip this slice (will be retried after resume or timeout)
                        slice_order.status = "DRIFT_PAUSED"
                        continue

                # Submit child order
                order_type = "LIMIT" if limit_price else "MARKET"
                order_id = await self._submit(
                    symbol=config.symbol,
                    side=config.side,
                    quantity=slice_order.quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    exchange=config.exchange,
                )

                slice_order.order_id = order_id
                # BUG 12 FIX: LIMIT orders do not fill immediately. Mark as
                # SUBMITTED/PENDING. Real fill confirmation should come from
                # the fill handler (FillListener / PaperFillSimulator).
                slice_order.status = "SUBMITTED"
                total_submitted += slice_order.quantity

                # NOTE: total_filled stays 0 here — actual fills are tracked
                # asynchronously via the FillListener / fill handler pipeline.
                # Consumers should use total_submitted to know how much was sent.

                logger.info(
                    "TWAP slice %d/%d: %s %s x%d @ %s",
                    slice_order.sequence + 1, config.num_slices,
                    config.side, config.symbol, slice_order.quantity,
                    f"{limit_price:.2f}" if limit_price else "MARKET",
                )

            except Exception as e:
                slice_order.status = "FAILED"
                logger.error("TWAP slice %d failed: %s", slice_order.sequence, e)

            # Wait for next interval
            if slice_order.sequence < config.num_slices - 1:
                await asyncio.sleep(interval)

        # Final stats — total_filled stays 0; actual fills arrive via FillListener.
        # total_submitted reflects how much was sent to the broker.
        execution.total_filled = total_filled
        execution.total_submitted = total_submitted
        execution.vwap = total_cost / total_filled if total_filled > 0 else 0.0
        if execution.status not in ("DRIFT_CANCELLED", "DRIFT_PAUSED", "CANCELLED"):
            execution.status = "SUBMITTED"
        execution.end_time = datetime.now(timezone.utc)

        # Log drift summary for post-trade analysis
        if execution.drift_events:
            logger.info(
                "TWAP drift summary: %s %s — %d drift events, max_drift=%.1f bps, "
                "final_status=%s",
                config.side, config.symbol, len(execution.drift_events),
                execution.max_drift_bps, execution.status,
            )

        logger.info(
            "TWAP done: %s %s submitted %d/%d (fills pending via FillListener)",
            config.side, config.symbol, total_submitted, config.total_quantity,
        )
        return execution

    async def cancel(self, exec_id: str) -> bool:
        """Cancel a running TWAP execution."""
        execution = self._active_executions.get(exec_id)
        if execution and execution.status == "RUNNING":
            execution.status = "CANCELLED"
            logger.info("TWAP cancelled: %s", exec_id)
            return True
        return False

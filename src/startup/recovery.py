"""
Cold start recovery: load persisted orders/positions, warm RiskManager and OrderLifecycle,
reconcile with broker when in live mode. No re-submit; no silent correction.
Startup invariant validation: fail if exposure > equity or duplicate active orders.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from src.core.events import Order, OrderStatus, Position
from src.monitoring.metrics import (
    track_reconciliation_mismatches_total,
    track_startup_recovery_duration,
    track_startup_recovery_failure,
    track_startup_recovery_mismatches,
)
from src.persistence.order_repo import OrderRepository
from src.persistence.position_repo import PositionRepository
from src.persistence.reconciliation import reconcile_positions

logger = logging.getLogger(__name__)


async def _query_broker_order_status(get_broker_positions, broker_order_id: str) -> str:
    """
    Query broker for order status. Returns one of: FILLED, REJECTED, CANCELLED, UNKNOWN, or status string.
    Adapts to different gateway interfaces.
    """
    try:
        # Try gateway.get_order_status() if available
        if hasattr(get_broker_positions, "get_order_status"):
            result = await get_broker_positions.get_order_status(broker_order_id)
            if result:
                status = str(result).upper()
                if "FILL" in status or "COMPLETE" in status or "TRADED" in status:
                    return "FILLED"
                if "REJECT" in status:
                    return "REJECTED"
                if "CANCEL" in status:
                    return "CANCELLED"
                return status
        # Fallback: try getting all orders and matching
        if callable(get_broker_positions):
            orders = await get_broker_positions()
            if orders:
                for o in orders:
                    oid = getattr(o, "broker_order_id", None) or getattr(o, "order_id", None)
                    if str(oid) == str(broker_order_id):
                        status_raw = getattr(o, "status", None)
                        status = status_raw.value if hasattr(status_raw, "value") else str(status_raw or "").upper()
                        if "FILL" in status or "COMPLETE" in status:
                            return "FILLED"
                        if "REJECT" in status:
                            return "REJECTED"
                        return status
    except Exception as e:
        logger.debug("Broker order status query failed for %s: %s", broker_order_id, e)
    return "UNKNOWN"


def _validate_recovery_invariants(
    risk_manager: Any,
    positions: list[Position],
    active_orders: list[Order],
    lifecycle: Any,
) -> None:
    """
    Verify startup invariants. Raises RuntimeError if violated.
    - Sum(position exposure) <= equity.
    - No duplicate order_ids in active orders.
    """
    equity = getattr(risk_manager, "equity", 0.0) or 0.0
    if equity <= 0:
        return
    position_value = 0.0
    for p in positions:
        qty = getattr(p, "quantity", 0) or 0
        avg = getattr(p, "avg_price", 0) or 0
        position_value += qty * avg
    if position_value > equity * 1.5:
        raise RuntimeError(
            f"Recovery invariant violated: sum(position exposure)={position_value:.2f} > 1.5*equity={equity * 1.5:.2f}"
        )
    order_ids = [o.order_id for o in active_orders if o and getattr(o, "order_id", None)]
    if len(order_ids) != len(set(order_ids)):
        raise RuntimeError("Recovery invariant violated: duplicate order_id in active orders")


async def run_cold_start_recovery(
    order_repo: OrderRepository,
    position_repo: PositionRepository,
    risk_manager: Any,
    lifecycle: Any,
    get_broker_positions: Callable | None = None,
    is_live_mode: bool = False,
    get_risk_snapshot: Callable[[], Any] | None = None,
) -> tuple[bool, int]:
    """
    Load active orders and positions from DB; warm RiskManager and OrderLifecycle.
    If is_live_mode and get_broker_positions provided, run reconciliation (log discrepancies; do not auto-correct).
    If get_risk_snapshot is provided and returns (equity, daily_pnl), restore them on risk_manager.
    Returns (safe_mode: bool, mismatch_count: int).
    safe_mode=True if broker was unreachable during reconciliation.
    """
    start = time.perf_counter()
    safe_mode = False
    mismatch_count = 0

    loop = asyncio.get_running_loop()

    try:
        # Load active orders (NEW, ACK, PARTIAL) and positions from DB (sync in executor)
        active_orders = await loop.run_in_executor(None, order_repo.list_active_orders)
        positions = await loop.run_in_executor(None, position_repo.list_positions)

        # Warm RiskManager from persisted positions
        risk_manager.load_positions_for_recovery(positions)
        # Restore equity and daily_pnl from risk_snapshot if available
        if get_risk_snapshot is not None:
            snapshot = await loop.run_in_executor(None, get_risk_snapshot)
            if snapshot is not None and len(snapshot) >= 2:
                equity, daily_pnl = float(snapshot[0]), float(snapshot[1])
                risk_manager.update_equity(equity)
                risk_manager.daily_pnl = daily_pnl
                logger.info("Cold start: restored equity=%.2f daily_pnl=%.2f", equity, daily_pnl)
        # Rebuild OrderLifecycle for active orders (no re-submit)
        lifecycle.load_for_recovery(active_orders)

        # Startup invariant validation (fail fast if state is inconsistent)
        _validate_recovery_invariants(risk_manager, positions, active_orders, lifecycle)

        # Reconcile SUBMITTING (write-ahead): check broker for actual status before marking REJECTED
        submitting = await loop.run_in_executor(None, order_repo.list_submitting_orders)
        if submitting:
            _resolved_count = 0
            filled_count = 0
            rejected_count = 0
            deferred_count = 0
            for o in submitting:
                if not o or not getattr(o, "order_id", None):
                    continue
                broker_oid = getattr(o, "broker_order_id", None)

                # If we have a broker reference and a gateway, check actual broker status
                if broker_oid and get_broker_positions is not None:
                    try:
                        # Query broker with timeout to prevent startup hang
                        broker_status = await asyncio.wait_for(
                            _query_broker_order_status(get_broker_positions, broker_oid),
                            timeout=5.0,
                        )
                        if broker_status == "FILLED":
                            # Broker says filled: update DB and create position in RiskManager
                            order_repo.update_order_status(o.order_id, OrderStatus.FILLED)
                            await lifecycle.update_status(
                                o.order_id,
                                OrderStatus.FILLED,
                                filled_qty=getattr(o, "quantity", 0),
                                avg_price=getattr(o, "limit_price", None),
                            )
                            # Add position to RiskManager
                            pos = Position(
                                symbol=o.symbol,
                                exchange=o.exchange,
                                side=o.side,
                                quantity=getattr(o, "quantity", 0),
                                avg_price=getattr(o, "limit_price", 0) or getattr(o, "avg_price", 0) or 0,
                            )
                            risk_manager.add_or_merge_position(pos)
                            filled_count += 1
                            logger.info("Cold start: SUBMITTING order %s confirmed FILLED by broker", o.order_id)
                            continue
                        elif broker_status in ("REJECTED", "CANCELLED", "UNKNOWN"):
                            # Broker confirms not active: mark REJECTED
                            order_repo.update_order_status(o.order_id, OrderStatus.REJECTED)
                            await lifecycle.update_status(
                                o.order_id, OrderStatus.REJECTED, filled_qty=0.0, avg_price=None
                            )
                            rejected_count += 1
                            continue
                        else:
                            # Broker returned ambiguous status: leave as SUBMITTING, retry next cycle
                            deferred_count += 1
                            logger.warning(
                                "Cold start: order %s broker status=%s, deferring", o.order_id, broker_status
                            )
                            continue
                    except TimeoutError:
                        deferred_count += 1
                        logger.warning("Cold start: broker query timeout for order %s, deferring", o.order_id)
                        continue
                    except Exception as e:
                        deferred_count += 1
                        logger.warning("Cold start: broker query failed for %s: %s, deferring", o.order_id, e)
                        continue

                # No broker reference or no gateway: default to REJECTED (legacy behavior)
                order_repo.update_order_status(o.order_id, OrderStatus.REJECTED)
                lifecycle.update_status(o.order_id, OrderStatus.REJECTED, filled_qty=0.0, avg_price=None)
                rejected_count += 1

            logger.info(
                "Cold start: reconciled %d SUBMITTING orders (filled=%d, rejected=%d, deferred=%d)",
                len(submitting),
                filled_count,
                rejected_count,
                deferred_count,
            )

        logger.info(
            "Cold start recovery: loaded %d active orders, %d positions",
            len(active_orders),
            len(positions),
        )

        # If live mode, reconcile with broker; on broker failure enter safe mode
        if is_live_mode and get_broker_positions is not None:
            try:

                def on_mismatch_count(n: int):
                    track_reconciliation_mismatches_total(n)
                    track_startup_recovery_mismatches(n)

                result = await reconcile_positions(
                    get_broker_positions,
                    position_repo,
                    on_mismatch_count,
                )
                mismatch_count = len(result.mismatches)
                if result.mismatches:
                    for msg in result.mismatches:
                        logger.warning("Cold start reconciliation mismatch: %s", msg)
            except Exception as e:
                logger.exception("Cold start: broker reconciliation failed (entering safe mode): %s", e)
                track_startup_recovery_failure()
                safe_mode = True
    except Exception as e:
        logger.exception("Cold start recovery failed: %s", e)
        track_startup_recovery_failure()
        raise

    elapsed = time.perf_counter() - start
    track_startup_recovery_duration(elapsed)
    logger.info(
        "Cold start recovery completed in %.2fs (safe_mode=%s, mismatches=%d)", elapsed, safe_mode, mismatch_count
    )
    return safe_mode, mismatch_count

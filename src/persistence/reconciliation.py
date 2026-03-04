"""Reconcile broker positions vs DB. Log discrepancies; do not auto-correct."""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List

from .position_repo import PositionRepository

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    in_sync: bool
    broker_positions: List[Any] = field(default_factory=list)
    db_positions: List[Any] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _broker_key_and_qty(p: Any) -> tuple:
    symbol = getattr(p, "symbol", p.get("symbol") if isinstance(p, dict) else None) or ""
    exchange = getattr(p, "exchange", p.get("exchange") if isinstance(p, dict) else "NSE") or "NSE"
    if hasattr(exchange, "value"):
        exchange = exchange.value
    side = getattr(p, "side", p.get("side") if isinstance(p, dict) else "BUY") or "BUY"
    qty = float(getattr(p, "quantity", p.get("quantity", 0) if isinstance(p, dict) else 0) or 0)
    return (symbol, str(exchange), str(side), qty)


def _db_key_and_qty(p: Any) -> tuple:
    symbol = getattr(p, "symbol", "")
    exchange = getattr(p, "exchange", "NSE")
    if hasattr(exchange, "value"):
        exchange = exchange.value
    side = getattr(p, "side", "BUY")
    if hasattr(side, "value"):
        side = side.value
    qty = float(getattr(p, "quantity", 0) or 0)
    return (symbol, str(exchange), str(side), qty)


async def reconcile_positions(
    get_broker_positions: Any,
    position_repo: PositionRepository,
    on_mismatch_count: Any,
) -> ReconciliationResult:
    """
    Compare broker positions vs DB. Log every discrepancy; call on_mismatch_count(n) with total mismatch count.
    Does NOT auto-correct. get_broker_positions is async (e.g. gateway.get_positions).
    """
    try:
        if asyncio.iscoroutinefunction(get_broker_positions):
            broker_positions = await get_broker_positions()
        else:
            broker_positions = get_broker_positions()
    except Exception as e:
        logger.exception("Reconciliation: fetch broker positions failed: %s", e)
        return ReconciliationResult(in_sync=False, mismatches=[f"broker_fetch_error: {e}"])

    loop = asyncio.get_running_loop()
    db_positions = await loop.run_in_executor(None, position_repo.list_positions)

    broker_by_key = {}
    for p in broker_positions:
        t = _broker_key_and_qty(p)
        k = (t[0], t[1], t[2])
        broker_by_key[k] = t[3]
    db_by_key = {}
    for p in db_positions:
        t = _db_key_and_qty(p)
        k = (t[0], t[1], t[2])
        db_by_key[k] = t[3]

    mismatches: List[str] = []
    all_keys = set(broker_by_key) | set(db_by_key)
    for key in all_keys:
        b_qty = broker_by_key.get(key, 0.0)
        d_qty = db_by_key.get(key, 0.0)
        if key not in broker_by_key:
            mismatches.append(f"db_extra:{key[0]}_{key[1]}_{key[2]}")
        elif key not in db_by_key:
            mismatches.append(f"broker_extra:{key[0]}_{key[1]}_{key[2]}")
        elif abs(b_qty - d_qty) > 1e-6:
            mismatches.append(f"qty_mismatch:{key[0]}_{key[1]}_{key[2]} broker={b_qty} db={d_qty}")

    if mismatches:
        for msg in mismatches:
            logger.warning("Reconciliation MISMATCH: %s", msg)
        try:
            on_mismatch_count(len(mismatches))
        except Exception as e:
            logger.warning("on_mismatch_count failed: %s", e)

    return ReconciliationResult(
        in_sync=len(mismatches) == 0,
        broker_positions=broker_positions,
        db_positions=db_positions,
        mismatches=mismatches,
    )

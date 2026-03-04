"""
Periodic broker reconciliation: fetch positions from broker, compare to RiskManager.
Mismatch detection; auto-heal for small drifts, emergency freeze for large ones.

Auto-heal logic:
  - If quantity difference < auto_heal_threshold_pct (default 5%) of position size:
    auto-adjust RiskManager position to match broker. Log adjustment with full audit trail.
    Fire INFO alert.
  - If quantity difference >= threshold: arm kill switch (critical mismatch).
    Fire CRITICAL alert.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from src.core.events import Position

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    in_sync: bool
    broker_positions: List[Any] = field(default_factory=list)
    local_positions: List[Position] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)
    auto_healed: List[str] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReconciliationJob:
    """
    Periodic job: get_positions() from gateway, compare to risk_manager.positions.
    Auto-heals small drifts (<5%); triggers kill switch for larger mismatches.
    """

    def __init__(
        self,
        fetch_broker_positions: Callable,
        get_local_positions: Callable,
        on_mismatch: Optional[Callable[[ReconciliationResult], None]] = None,
        trigger_freeze_on_mismatch: bool = True,
        auto_heal_threshold_pct: float = 5.0,
        adjust_local_position_fn: Optional[Callable[[str, str, int], None]] = None,
        # adjust_local_position_fn(symbol, exchange, broker_quantity) — adjust RiskManager
    ):
        self.fetch_broker_positions = fetch_broker_positions
        self.get_local_positions = get_local_positions
        self.on_mismatch = on_mismatch
        self.trigger_freeze_on_mismatch = trigger_freeze_on_mismatch
        self.auto_heal_threshold_pct = auto_heal_threshold_pct
        self.adjust_local_position_fn = adjust_local_position_fn
        self._audit_log: List[dict] = []

    def get_audit_log(self) -> List[dict]:
        """Return reconciliation audit trail."""
        return list(self._audit_log)

    async def run(self) -> ReconciliationResult:
        """Run one reconciliation cycle with auto-heal."""
        try:
            broker_positions = await self.fetch_broker_positions()
        except Exception as e:
            logger.exception("Reconciliation: fetch broker positions failed: %s", e)
            return ReconciliationResult(
                in_sync=False,
                mismatches=[f"broker_fetch_error: {e}"],
            )

        local_positions = self.get_local_positions()
        mismatches: List[str] = []
        auto_healed: List[str] = []

        # Normalize: broker may return list of dicts or objects; local is List[Position]
        broker_by_key = {}
        for p in broker_positions:
            if isinstance(p, dict):
                sym = p.get("symbol", "")
                exch_raw = p.get("exchange", "")
                b_qty = p.get("quantity", 0)
            else:
                sym = getattr(p, "symbol", "")
                exch_raw = getattr(p, "exchange", "")
                b_qty = getattr(p, "quantity", 0)
            exch = exch_raw.value if hasattr(exch_raw, "value") else str(exch_raw)
            broker_by_key[(sym, exch)] = (p, b_qty or 0)

        local_by_key = {(p.symbol, p.exchange.value if hasattr(p.exchange, "value") else str(p.exchange)): p for p in local_positions}

        all_keys = set(broker_by_key) | set(local_by_key)
        for key in all_keys:
            b_entry = broker_by_key.get(key)
            l = local_by_key.get(key)
            if b_entry is None:
                mismatches.append(f"local_extra:{key[0]}_{key[1]}")
            elif l is None:
                mismatches.append(f"broker_extra:{key[0]}_{key[1]}")
            else:
                _, b_qty = b_entry
                local_qty = l.quantity or 0
                diff = abs(local_qty - b_qty)

                if diff < 1e-6:
                    continue  # In sync

                # Calculate difference as % of max(local, broker) position
                max_qty = max(abs(local_qty), abs(b_qty), 1)
                diff_pct = (diff / max_qty) * 100.0

                if diff_pct < self.auto_heal_threshold_pct and self.adjust_local_position_fn:
                    # Auto-heal: small drift, adjust local to match broker
                    try:
                        self.adjust_local_position_fn(key[0], key[1], int(b_qty))
                        heal_msg = (
                            f"auto_healed:{key[0]}_{key[1]} "
                            f"local={local_qty}->broker={b_qty} (diff={diff_pct:.1f}%)"
                        )
                        auto_healed.append(heal_msg)
                        logger.info("Reconciliation AUTO-HEALED: %s", heal_msg)
                        self._audit_log.append({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "action": "auto_heal",
                            "symbol": key[0],
                            "exchange": key[1],
                            "local_qty": local_qty,
                            "broker_qty": int(b_qty),
                            "diff_pct": round(diff_pct, 2),
                        })
                    except Exception as e:
                        logger.warning("Auto-heal failed for %s: %s", key[0], e)
                        mismatches.append(f"qty_mismatch:{key[0]}_{key[1]} local={local_qty} broker={b_qty}")
                else:
                    # Large mismatch: flag for kill switch
                    mismatch_msg = (
                        f"qty_mismatch:{key[0]}_{key[1]} "
                        f"local={local_qty} broker={b_qty} (diff={diff_pct:.1f}%)"
                    )
                    mismatches.append(mismatch_msg)
                    self._audit_log.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "action": "mismatch_detected",
                        "symbol": key[0],
                        "exchange": key[1],
                        "local_qty": local_qty,
                        "broker_qty": int(b_qty),
                        "diff_pct": round(diff_pct, 2),
                        "severity": "CRITICAL",
                    })

        # Keep audit log bounded
        if len(self._audit_log) > 500:
            self._audit_log = self._audit_log[-500:]

        result = ReconciliationResult(
            in_sync=len(mismatches) == 0,
            broker_positions=broker_positions,
            local_positions=local_positions,
            mismatches=mismatches,
            auto_healed=auto_healed,
        )

        if mismatches:
            logger.warning("Reconciliation MISMATCH (critical): %s", mismatches)
            if self.on_mismatch:
                self.on_mismatch(result)
            if self.trigger_freeze_on_mismatch and self.on_mismatch:
                logger.warning("Reconciliation: freeze triggered on mismatch — calling on_mismatch callback")
        elif auto_healed:
            logger.info("Reconciliation: all drifts auto-healed (%d adjustments)", len(auto_healed))

        return result

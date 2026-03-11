"""
Position recovery manager: crash-safe position persistence, startup recovery,
broker reconciliation, and idempotency key persistence.

Production-grade: every position state change is persisted to DB before
in-memory update. On startup, positions are loaded from DB and reconciled
with the broker (if available) to detect phantom or missing positions.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.core.events import Exchange, Position, SignalSide

from .database import get_session_factory, session_scope
from .models import (
    AuditEventModel,
    Base,
    OrderModel,
    PositionModel,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PositionRecoveryManager:
    """
    Crash-safe position management:
    - Persist every position state change to DB (write-ahead)
    - Recover positions from DB on startup
    - Reconcile with broker positions (if live)
    - Persist idempotency keys to DB (survive restarts)
    - Transaction-safe state transitions with OCC
    - Full audit trail of all recovery operations
    """

    def __init__(self, session_factory=None):
        self._session_factory = session_factory or get_session_factory()
        self._recovered_positions: List[Position] = []
        self._recovery_audit: List[Dict[str, Any]] = []
        self._idempotency_keys: Dict[str, datetime] = {}

    # ── Write-Ahead Position Persistence ──

    def persist_position_change(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: float,
        avg_price: float,
        strategy_id: Optional[str] = None,
        change_type: str = "update",
        reason: str = "",
    ) -> bool:
        """
        Persist a position state change to DB BEFORE updating in-memory state.
        Write-ahead pattern: if this fails, caller must NOT update in-memory.

        Returns True on success, False on failure.
        """
        try:
            with session_scope() as sess:
                existing = sess.query(PositionModel).filter(
                    and_(
                        PositionModel.symbol == symbol,
                        PositionModel.exchange == exchange,
                        PositionModel.side == side,
                    )
                ).first()

                now = _utc_now()

                if change_type == "close" or quantity <= 0:
                    # Position closed: delete from DB
                    if existing:
                        sess.delete(existing)
                    self._audit_log(
                        sess, "position_closed", "recovery_manager",
                        {
                            "symbol": symbol, "exchange": exchange, "side": side,
                            "final_avg_price": avg_price, "reason": reason,
                        },
                    )
                elif existing is None:
                    # New position
                    sess.add(PositionModel(
                        symbol=symbol,
                        exchange=exchange,
                        side=side,
                        quantity=quantity,
                        avg_price=avg_price,
                        realized_pnl=0.0,
                        strategy_id=strategy_id,
                        version=0,
                        updated_at=now,
                    ))
                    self._audit_log(
                        sess, "position_opened", "recovery_manager",
                        {
                            "symbol": symbol, "exchange": exchange, "side": side,
                            "quantity": quantity, "avg_price": avg_price,
                            "strategy_id": strategy_id, "reason": reason,
                        },
                    )
                else:
                    # Update existing position with OCC
                    current_version = existing.version or 0
                    from sqlalchemy import update
                    result = sess.execute(
                        update(PositionModel)
                        .where(
                            and_(
                                PositionModel.id == existing.id,
                                PositionModel.version == current_version,
                            )
                        )
                        .values(
                            quantity=quantity,
                            avg_price=avg_price,
                            version=current_version + 1,
                            updated_at=now,
                            strategy_id=strategy_id or existing.strategy_id,
                        )
                    )
                    if result.rowcount == 0:
                        logger.error(
                            "OCC conflict persisting position %s/%s/%s (version=%d)",
                            symbol, exchange, side, current_version,
                        )
                        return False
                    self._audit_log(
                        sess, "position_updated", "recovery_manager",
                        {
                            "symbol": symbol, "exchange": exchange, "side": side,
                            "quantity": quantity, "avg_price": avg_price,
                            "version": current_version + 1, "reason": reason,
                        },
                    )
            return True
        except Exception as e:
            logger.error("Write-ahead position persist failed for %s: %s", symbol, e)
            return False

    # ── Startup Recovery ──

    def recover_positions(self) -> List[Position]:
        """
        Load all open positions from DB on startup.
        Returns list of Position domain objects for RiskManager initialization.
        """
        positions: List[Position] = []
        try:
            with session_scope() as sess:
                rows = sess.query(PositionModel).filter(
                    PositionModel.quantity > 0
                ).all()
                for row in rows:
                    try:
                        pos = Position(
                            symbol=row.symbol,
                            exchange=Exchange(row.exchange) if row.exchange else Exchange.NSE,
                            side=SignalSide(row.side) if row.side else SignalSide.BUY,
                            quantity=row.quantity,
                            avg_price=row.avg_price,
                            unrealized_pnl=0.0,
                            strategy_id=row.strategy_id,
                        )
                        positions.append(pos)
                    except Exception as e:
                        logger.warning(
                            "Skipping corrupt position row id=%d symbol=%s: %s",
                            row.id, row.symbol, e,
                        )
                self._audit_log(
                    sess, "position_recovery", "recovery_manager",
                    {"positions_recovered": len(positions)},
                )
            self._recovered_positions = positions
            logger.info("Position recovery: loaded %d open positions from DB", len(positions))
        except Exception as e:
            logger.error("Position recovery failed: %s", e)
        return positions

    def reconcile_with_broker(
        self,
        broker_positions: List[Dict[str, Any]],
        local_positions: Optional[List[Position]] = None,
    ) -> Dict[str, Any]:
        """
        Reconcile local DB positions with broker-reported positions.

        Returns a report dict with:
            - in_sync: bool
            - missing_locally: positions on broker but not in DB
            - phantom_local: positions in DB but not on broker
            - quantity_mismatches: positions with different quantities
            - actions_taken: list of auto-healing actions

        Broker positions are expected as list of dicts:
            {"symbol": str, "exchange": str, "side": str, "quantity": float, "avg_price": float}
        """
        report: Dict[str, Any] = {
            "in_sync": True,
            "missing_locally": [],
            "phantom_local": [],
            "quantity_mismatches": [],
            "actions_taken": [],
            "timestamp": _utc_now().isoformat(),
        }

        local = local_positions or self._recovered_positions

        # Build lookup maps
        local_map: Dict[str, Position] = {}
        for p in local:
            key = f"{p.symbol}:{getattr(p.exchange, 'value', str(p.exchange))}:{getattr(p.side, 'value', str(p.side))}"
            local_map[key] = p

        broker_map: Dict[str, Dict[str, Any]] = {}
        for bp in broker_positions:
            key = f"{bp['symbol']}:{bp.get('exchange', 'NSE')}:{bp.get('side', 'BUY')}"
            broker_map[key] = bp

        # Find positions on broker but not locally
        for key, bp in broker_map.items():
            if key not in local_map:
                report["missing_locally"].append(bp)
                report["in_sync"] = False
                # Auto-heal: add missing position to DB
                parts = key.split(":")
                self.persist_position_change(
                    symbol=parts[0],
                    exchange=parts[1],
                    side=parts[2],
                    quantity=bp["quantity"],
                    avg_price=bp.get("avg_price", 0.0),
                    change_type="update",
                    reason="reconciliation_missing_locally",
                )
                report["actions_taken"].append(
                    f"Added missing position {key} (qty={bp['quantity']})"
                )

        # Find positions locally but not on broker
        for key, lp in local_map.items():
            if key not in broker_map:
                report["phantom_local"].append({
                    "symbol": lp.symbol,
                    "exchange": getattr(lp.exchange, "value", str(lp.exchange)),
                    "side": getattr(lp.side, "value", str(lp.side)),
                    "quantity": lp.quantity,
                })
                report["in_sync"] = False
                logger.warning(
                    "PHANTOM position detected: %s exists locally but NOT on broker. "
                    "Manual review required.",
                    key,
                )

        # Check quantity mismatches
        for key in set(local_map.keys()) & set(broker_map.keys()):
            lp = local_map[key]
            bp = broker_map[key]
            if abs(lp.quantity - bp["quantity"]) > 0.01:
                report["quantity_mismatches"].append({
                    "key": key,
                    "local_qty": lp.quantity,
                    "broker_qty": bp["quantity"],
                    "diff": bp["quantity"] - lp.quantity,
                })
                report["in_sync"] = False
                # Auto-heal: trust broker quantity
                parts = key.split(":")
                self.persist_position_change(
                    symbol=parts[0],
                    exchange=parts[1],
                    side=parts[2],
                    quantity=bp["quantity"],
                    avg_price=bp.get("avg_price", lp.avg_price),
                    change_type="update",
                    reason=f"reconciliation_qty_mismatch_local={lp.quantity}_broker={bp['quantity']}",
                )
                report["actions_taken"].append(
                    f"Corrected quantity for {key}: {lp.quantity} -> {bp['quantity']}"
                )

        # Audit the reconciliation result
        try:
            with session_scope() as sess:
                self._audit_log(
                    sess, "position_reconciliation", "recovery_manager",
                    {
                        "in_sync": report["in_sync"],
                        "missing_locally": len(report["missing_locally"]),
                        "phantom_local": len(report["phantom_local"]),
                        "quantity_mismatches": len(report["quantity_mismatches"]),
                        "actions_taken": len(report["actions_taken"]),
                    },
                )
        except Exception as e:
            logger.debug("Failed to audit reconciliation: %s", e)

        if report["in_sync"]:
            logger.info("Position reconciliation: IN SYNC (%d positions)", len(local))
        else:
            logger.warning(
                "Position reconciliation: OUT OF SYNC "
                "(missing=%d, phantom=%d, qty_mismatch=%d, actions=%d)",
                len(report["missing_locally"]),
                len(report["phantom_local"]),
                len(report["quantity_mismatches"]),
                len(report["actions_taken"]),
            )

        return report

    # ── Idempotency Key Persistence ──

    def persist_idempotency_key(self, key: str, created_at: Optional[datetime] = None) -> bool:
        """
        Persist an idempotency key to DB so it survives restarts.
        Uses the orders table idempotency_key column to avoid schema changes.
        """
        try:
            with session_scope() as sess:
                # Check if key already exists
                existing = sess.query(OrderModel).filter(
                    OrderModel.idempotency_key == key
                ).first()
                if existing is not None:
                    return True  # Already persisted
                # Insert a placeholder order with this idempotency key
                sess.add(OrderModel(
                    order_id=f"idem_{key[:48]}_{_utc_now().strftime('%Y%m%d%H%M%S')}",
                    idempotency_key=key,
                    symbol="IDEMPOTENCY_KEY",
                    exchange="NSE",
                    side="NONE",
                    quantity=0,
                    order_type="LIMIT",
                    status="CANCELLED",
                    filled_qty=0,
                    created_at=created_at or _utc_now(),
                ))
            self._idempotency_keys[key] = created_at or _utc_now()
            return True
        except Exception as e:
            logger.debug("Idempotency key persist failed for %s: %s", key[:40], e)
            return False

    def load_idempotency_keys(self, max_age_hours: int = 24) -> Dict[str, datetime]:
        """
        Load recent idempotency keys from DB on startup.
        Returns dict of key -> created_at for deduplication.
        """
        keys: Dict[str, datetime] = {}
        try:
            from datetime import timedelta
            cutoff = _utc_now() - timedelta(hours=max_age_hours)
            with session_scope() as sess:
                rows = sess.query(OrderModel).filter(
                    and_(
                        OrderModel.idempotency_key.isnot(None),
                        OrderModel.created_at >= cutoff,
                    )
                ).all()
                for row in rows:
                    if row.idempotency_key:
                        keys[row.idempotency_key] = row.created_at
            self._idempotency_keys = keys
            logger.info("Loaded %d idempotency keys from DB (last %dh)", len(keys), max_age_hours)
        except Exception as e:
            logger.error("Failed to load idempotency keys: %s", e)
        return keys

    def is_idempotency_key_used(self, key: str) -> bool:
        """Check if an idempotency key has already been used (in-memory + DB fallback)."""
        if key in self._idempotency_keys:
            return True
        # DB fallback
        try:
            with session_scope() as sess:
                existing = sess.query(OrderModel).filter(
                    OrderModel.idempotency_key == key
                ).first()
                if existing is not None:
                    self._idempotency_keys[key] = existing.created_at
                    return True
        except Exception:
            pass
        return False

    # ── Transaction-Safe State Transitions ──

    def atomic_position_transition(
        self,
        symbol: str,
        exchange: str,
        side: str,
        from_qty: float,
        to_qty: float,
        avg_price: float,
        strategy_id: Optional[str] = None,
        reason: str = "",
    ) -> bool:
        """
        Atomically transition a position from one state to another.
        Uses OCC (version column) to prevent concurrent modifications.
        """
        try:
            with session_scope() as sess:
                existing = sess.query(PositionModel).filter(
                    and_(
                        PositionModel.symbol == symbol,
                        PositionModel.exchange == exchange,
                        PositionModel.side == side,
                    )
                ).first()

                if existing is None:
                    if from_qty != 0:
                        logger.error(
                            "Atomic transition failed: expected qty=%.2f but position "
                            "not found for %s/%s/%s",
                            from_qty, symbol, exchange, side,
                        )
                        return False
                    # Create new position
                    if to_qty > 0:
                        sess.add(PositionModel(
                            symbol=symbol,
                            exchange=exchange,
                            side=side,
                            quantity=to_qty,
                            avg_price=avg_price,
                            realized_pnl=0.0,
                            strategy_id=strategy_id,
                            version=0,
                            updated_at=_utc_now(),
                        ))
                    return True

                # Verify current state matches expectation
                if abs(existing.quantity - from_qty) > 0.01:
                    logger.error(
                        "Atomic transition failed: expected qty=%.2f but found qty=%.2f "
                        "for %s/%s/%s (stale state)",
                        from_qty, existing.quantity, symbol, exchange, side,
                    )
                    return False

                current_version = existing.version or 0

                if to_qty <= 0:
                    # Close position
                    from sqlalchemy import delete
                    result = sess.execute(
                        delete(PositionModel).where(
                            and_(
                                PositionModel.id == existing.id,
                                PositionModel.version == current_version,
                            )
                        )
                    )
                else:
                    # Update position
                    from sqlalchemy import update
                    result = sess.execute(
                        update(PositionModel)
                        .where(
                            and_(
                                PositionModel.id == existing.id,
                                PositionModel.version == current_version,
                            )
                        )
                        .values(
                            quantity=to_qty,
                            avg_price=avg_price,
                            version=current_version + 1,
                            updated_at=_utc_now(),
                            strategy_id=strategy_id or existing.strategy_id,
                        )
                    )

                if result.rowcount == 0:
                    logger.error(
                        "Atomic transition OCC conflict for %s/%s/%s (version=%d)",
                        symbol, exchange, side, current_version,
                    )
                    return False

                self._audit_log(
                    sess, "position_transition", "recovery_manager",
                    {
                        "symbol": symbol, "exchange": exchange, "side": side,
                        "from_qty": from_qty, "to_qty": to_qty,
                        "avg_price": avg_price, "reason": reason,
                    },
                )
            return True
        except Exception as e:
            logger.error("Atomic position transition failed for %s: %s", symbol, e)
            return False

    # ── Recovery Audit Logging ──

    def get_recovery_audit(self) -> List[Dict[str, Any]]:
        """Return the audit trail for the last recovery operation."""
        return list(self._recovery_audit)

    def _audit_log(
        self,
        session: Session,
        event_type: str,
        actor: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an audit event within an existing session/transaction."""
        try:
            payload_str = json.dumps(payload, default=str) if payload else None
            session.add(AuditEventModel(
                ts=_utc_now(),
                event_type=event_type,
                actor=actor,
                payload=payload_str,
            ))
            session.flush()
            self._recovery_audit.append({
                "ts": _utc_now().isoformat(),
                "event_type": event_type,
                "actor": actor,
                "payload": payload,
            })
        except Exception as e:
            logger.debug("Recovery audit log failed: %s", e)

    # ── Full Recovery Workflow ──

    def run_full_recovery(
        self,
        broker_get_positions: Optional[Callable] = None,
        risk_manager=None,
        is_live: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full startup recovery workflow:
        1. Load positions from DB
        2. Load idempotency keys from DB
        3. If live mode, reconcile with broker
        4. Initialize risk manager with recovered positions

        Returns recovery report.
        """
        report: Dict[str, Any] = {
            "success": True,
            "positions_recovered": 0,
            "idempotency_keys_loaded": 0,
            "reconciliation": None,
            "safe_mode_recommended": False,
            "errors": [],
        }

        # Step 1: Recover positions from DB
        try:
            positions = self.recover_positions()
            report["positions_recovered"] = len(positions)
        except Exception as e:
            report["errors"].append(f"Position recovery failed: {e}")
            report["safe_mode_recommended"] = True
            positions = []

        # Step 2: Load idempotency keys
        try:
            keys = self.load_idempotency_keys()
            report["idempotency_keys_loaded"] = len(keys)
        except Exception as e:
            report["errors"].append(f"Idempotency key load failed: {e}")

        # Step 3: Broker reconciliation (live mode only)
        if is_live and broker_get_positions is not None:
            try:
                broker_positions = broker_get_positions()
                # Handle async result
                import asyncio
                if asyncio.iscoroutine(broker_positions):
                    loop = asyncio.get_event_loop()
                    broker_positions = loop.run_until_complete(broker_positions)

                if broker_positions is not None:
                    # Normalize broker positions to dicts
                    normalized = []
                    for bp in broker_positions:
                        if isinstance(bp, dict):
                            normalized.append(bp)
                        else:
                            normalized.append({
                                "symbol": getattr(bp, "symbol", ""),
                                "exchange": getattr(
                                    getattr(bp, "exchange", None),
                                    "value",
                                    str(getattr(bp, "exchange", "NSE")),
                                ),
                                "side": getattr(
                                    getattr(bp, "side", None),
                                    "value",
                                    str(getattr(bp, "side", "BUY")),
                                ),
                                "quantity": getattr(bp, "quantity", 0),
                                "avg_price": getattr(bp, "avg_price", 0),
                            })

                    recon = self.reconcile_with_broker(normalized, positions)
                    report["reconciliation"] = recon

                    if not recon["in_sync"]:
                        if recon["phantom_local"]:
                            # Phantom positions are dangerous in live mode
                            report["safe_mode_recommended"] = True
                            logger.warning(
                                "SAFE MODE RECOMMENDED: %d phantom positions detected",
                                len(recon["phantom_local"]),
                            )
            except Exception as e:
                report["errors"].append(f"Broker reconciliation failed: {e}")
                report["safe_mode_recommended"] = True
                logger.error("Broker reconciliation failed: %s", e)

        # Step 4: Initialize risk manager
        if risk_manager is not None and positions:
            try:
                for pos in positions:
                    risk_manager.add_position(pos)
                logger.info(
                    "Risk manager initialized with %d recovered positions",
                    len(positions),
                )
            except Exception as e:
                report["errors"].append(f"Risk manager init failed: {e}")
                logger.error("Risk manager position init failed: %s", e)

        report["success"] = len(report["errors"]) == 0
        logger.info(
            "Full recovery complete: positions=%d, idem_keys=%d, sync=%s, safe_mode=%s",
            report["positions_recovered"],
            report["idempotency_keys_loaded"],
            report.get("reconciliation", {}).get("in_sync", "N/A"),
            report["safe_mode_recommended"],
        )
        return report

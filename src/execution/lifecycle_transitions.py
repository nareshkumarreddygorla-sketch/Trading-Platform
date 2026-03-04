"""
Order lifecycle state machine. Institutional: only valid transitions allowed.
Prevents illegal status transitions (e.g. FILLED -> CANCELLED).
"""
from src.core.events import OrderStatus

# DB status strings (order_repo uses these)
DB_SUBMITTING = "SUBMITTING"  # Write-ahead: persisted before broker call
DB_NEW = "NEW"
DB_ACK = "ACK"
DB_PARTIAL = "PARTIAL"
DB_FILLED = "FILLED"
DB_REJECTED = "REJECTED"
DB_CANCELLED = "CANCELLED"

# Valid transitions: SUBMITTING -> NEW, REJECTED, CANCELLED (broker response or timeout)
# NEW -> ACK, PARTIAL, FILLED, REJECTED, CANCELLED
# ACK -> PARTIAL, FILLED, REJECTED, CANCELLED
# PARTIAL -> FILLED, CANCELLED
# FILLED, REJECTED, CANCELLED are terminal
ALLOWED_DB_TRANSITIONS = {
    DB_SUBMITTING: {DB_NEW, DB_ACK, DB_REJECTED, DB_CANCELLED},
    DB_NEW: {DB_ACK, DB_PARTIAL, DB_FILLED, DB_REJECTED, DB_CANCELLED},
    DB_ACK: {DB_PARTIAL, DB_FILLED, DB_REJECTED, DB_CANCELLED},
    DB_PARTIAL: {DB_FILLED, DB_CANCELLED},
    DB_FILLED: set(),
    DB_REJECTED: set(),
    DB_CANCELLED: set(),
}

# Domain OrderStatus -> DB status
ORDER_STATUS_TO_DB = {
    OrderStatus.PENDING: DB_NEW,
    OrderStatus.LIVE: DB_ACK,
    OrderStatus.PARTIALLY_FILLED: DB_PARTIAL,
    OrderStatus.FILLED: DB_FILLED,
    OrderStatus.REJECTED: DB_REJECTED,
    OrderStatus.CANCELLED: DB_CANCELLED,
}


def is_allowed_transition(from_db_status: str, to_db_status: str) -> bool:
    """Return True if transition from_db_status -> to_db_status is allowed."""
    allowed = ALLOWED_DB_TRANSITIONS.get(from_db_status, set())
    return to_db_status in allowed


def is_allowed_transition_domain(from_status: OrderStatus, to_status: OrderStatus) -> bool:
    """Return True if transition from_status -> to_status is allowed (domain enums)."""
    from_db = ORDER_STATUS_TO_DB.get(from_status, "NEW")
    to_db = ORDER_STATUS_TO_DB.get(to_status, "NEW")
    return is_allowed_transition(from_db, to_db)

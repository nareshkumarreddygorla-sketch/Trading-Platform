"""
Error classification for broker/execution errors.

Categorises exceptions into:
  - RETRYABLE: network timeouts, 5xx server errors, temporary broker issues
  - PERMANENT: invalid order params, insufficient funds, 4xx client errors
  - UNKNOWN: anything else (treated as retryable with limited attempts)

Used by OrderEntryService and OrderRouter to decide whether to retry or reject.
"""
import asyncio
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    RETRYABLE = "retryable"      # Network timeout, 5xx, transient broker issue
    PERMANENT = "permanent"      # Invalid order, insufficient funds, 4xx, auth failure
    UNKNOWN = "unknown"          # Unclassified — default to limited retry


# Angel One error codes that are permanent (no retry)
_ANGEL_PERMANENT_CODES = frozenset({
    "AB1001",  # Invalid input
    "AB1002",  # Insufficient margin/funds
    "AB1003",  # Circuit breaker halt
    "AB1004",  # Price out of range
    "AB1005",  # Quantity validation error
    "OI8001", "OI8002", "OI8003", "OI8004", "OI8005",  # Exchange circuit
})

# Angel One error codes that are retryable
_ANGEL_RETRYABLE_CODES = frozenset({
    "AG8001", "AG8002", "AB8051",  # Session expired (retryable after refresh)
})

# Kite error codes that are permanent
_KITE_PERMANENT_CODES = frozenset({
    "ORDER_REJECTED",
    "INPUT_ERROR",
    "CIRCUIT_HALT",
    "AUTH_FAILED",
    "TOKEN_EXPIRED",
})

# Kite error codes that are retryable
_KITE_RETRYABLE_CODES = frozenset({
    "TIMEOUT",
    "RETRIES_EXHAUSTED",
})

# Exception type names that indicate permanent errors
_PERMANENT_EXCEPTION_TYPES = frozenset({
    "OrderException",      # Kite: order rejected by exchange
    "InputException",      # Kite: invalid parameters
    "PermissionException", # Kite: auth/permission issue
    "ValueError",          # Our own validation errors
})

# Exception type names that indicate retryable errors
_RETRYABLE_EXCEPTION_TYPES = frozenset({
    "TimeoutError",
    "ConnectionError",
    "NetworkException",    # Kite: network issues
    "GeneralException",    # Kite: catch-all (often transient)
})

# HTTP status code patterns in error messages
_PERMANENT_STATUS_KEYWORDS = frozenset({
    "400", "401", "403", "404", "422",
    "insufficient", "invalid", "rejected", "not allowed",
    "margin", "funds",
})

_RETRYABLE_STATUS_KEYWORDS = frozenset({
    "500", "502", "503", "504",
    "timeout", "timed out", "connection",
    "temporarily", "unavailable", "rate limit",
    "retry", "overloaded",
})


def classify_error(exc: Exception) -> ErrorCategory:
    """Classify an exception as retryable, permanent, or unknown.

    Args:
        exc: The exception to classify.

    Returns:
        ErrorCategory indicating how the caller should handle the error.
    """
    exc_type_name = type(exc).__name__

    # asyncio.TimeoutError is always retryable
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return ErrorCategory.RETRYABLE

    # ValueError is always permanent (validation failure)
    if isinstance(exc, ValueError):
        return ErrorCategory.PERMANENT

    # Check by exception type name (for broker SDK exceptions we don't import)
    if exc_type_name in _PERMANENT_EXCEPTION_TYPES:
        return ErrorCategory.PERMANENT
    if exc_type_name in _RETRYABLE_EXCEPTION_TYPES:
        return ErrorCategory.RETRYABLE

    # Check for errorcode attribute (BrokerClientError, KiteGatewayError)
    errorcode = getattr(exc, "errorcode", None) or ""
    if errorcode:
        if errorcode in _ANGEL_PERMANENT_CODES or errorcode in _KITE_PERMANENT_CODES:
            return ErrorCategory.PERMANENT
        if errorcode in _ANGEL_RETRYABLE_CODES or errorcode in _KITE_RETRYABLE_CODES:
            return ErrorCategory.RETRYABLE

    # Check HTTP response status code if available
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", 0)
        if 400 <= status < 500:
            return ErrorCategory.PERMANENT
        if status >= 500:
            return ErrorCategory.RETRYABLE

    # Keyword-based fallback from error message
    msg = str(exc).lower()
    if any(kw in msg for kw in _PERMANENT_STATUS_KEYWORDS):
        return ErrorCategory.PERMANENT
    if any(kw in msg for kw in _RETRYABLE_STATUS_KEYWORDS):
        return ErrorCategory.RETRYABLE

    return ErrorCategory.UNKNOWN


def is_retryable(exc: Exception) -> bool:
    """Convenience: return True if the error should be retried."""
    cat = classify_error(exc)
    return cat in (ErrorCategory.RETRYABLE, ErrorCategory.UNKNOWN)


def is_permanent(exc: Exception) -> bool:
    """Convenience: return True if the error should NOT be retried."""
    return classify_error(exc) == ErrorCategory.PERMANENT

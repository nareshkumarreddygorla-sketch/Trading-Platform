"""Unit tests for src.execution.order_entry.error_classification.

Covers:
- ErrorCategory enum values
- classify_error(): built-in exception types (TimeoutError, ConnectionError,
  ValueError, OSError), exception type name matching, errorcode attribute,
  HTTP response status codes, keyword-based fallback, unknown errors
- is_retryable() convenience function
- is_permanent() convenience function
- Angel One and Kite specific error codes
"""

from types import SimpleNamespace

import pytest

from src.execution.order_entry.error_classification import (
    ErrorCategory,
    classify_error,
    is_permanent,
    is_retryable,
)

# ──────────────────────────────────────────────────
# ErrorCategory enum
# ──────────────────────────────────────────────────


class TestErrorCategory:
    def test_enum_values(self):
        assert ErrorCategory.RETRYABLE == "retryable"
        assert ErrorCategory.PERMANENT == "permanent"
        assert ErrorCategory.UNKNOWN == "unknown"


# ──────────────────────────────────────────────────
# classify_error — built-in types
# ──────────────────────────────────────────────────


class TestBuiltinExceptions:
    def test_timeout_error_is_retryable(self):
        assert classify_error(TimeoutError("timed out")) == ErrorCategory.RETRYABLE

    def test_asyncio_timeout_is_retryable(self):
        assert classify_error(TimeoutError()) == ErrorCategory.RETRYABLE

    def test_connection_error_is_retryable(self):
        assert classify_error(ConnectionError("refused")) == ErrorCategory.RETRYABLE

    def test_os_error_is_retryable(self):
        assert classify_error(OSError("network unreachable")) == ErrorCategory.RETRYABLE

    def test_value_error_is_permanent(self):
        assert classify_error(ValueError("invalid order")) == ErrorCategory.PERMANENT


# ──────────────────────────────────────────────────
# classify_error — exception type name matching
# ──────────────────────────────────────────────────


class TestExceptionTypeNameMatching:
    def test_order_exception_is_permanent(self):
        # Simulate Kite OrderException (we don't import it, match by name)
        exc = type("OrderException", (Exception,), {})("rejected by exchange")
        assert classify_error(exc) == ErrorCategory.PERMANENT

    def test_input_exception_is_permanent(self):
        exc = type("InputException", (Exception,), {})("invalid params")
        assert classify_error(exc) == ErrorCategory.PERMANENT

    def test_permission_exception_is_permanent(self):
        exc = type("PermissionException", (Exception,), {})("auth failed")
        assert classify_error(exc) == ErrorCategory.PERMANENT

    def test_network_exception_is_retryable(self):
        exc = type("NetworkException", (Exception,), {})("network error")
        assert classify_error(exc) == ErrorCategory.RETRYABLE

    def test_general_exception_is_retryable(self):
        exc = type("GeneralException", (Exception,), {})("transient")
        assert classify_error(exc) == ErrorCategory.RETRYABLE


# ──────────────────────────────────────────────────
# classify_error — errorcode attribute
# ──────────────────────────────────────────────────


class TestErrorcodeAttribute:
    def _make_exc(self, code: str) -> Exception:
        exc = Exception(f"broker error {code}")
        exc.errorcode = code  # type: ignore[attr-defined]
        return exc

    # Angel One permanent codes
    @pytest.mark.parametrize("code", ["AB1001", "AB1002", "AB1003", "AB1004", "AB1005"])
    def test_angel_permanent_codes(self, code):
        assert classify_error(self._make_exc(code)) == ErrorCategory.PERMANENT

    # Angel One retryable codes
    @pytest.mark.parametrize("code", ["AG8001", "AG8002", "AB8051"])
    def test_angel_retryable_codes(self, code):
        assert classify_error(self._make_exc(code)) == ErrorCategory.RETRYABLE

    # Kite permanent codes
    @pytest.mark.parametrize("code", ["ORDER_REJECTED", "INPUT_ERROR", "CIRCUIT_HALT", "AUTH_FAILED"])
    def test_kite_permanent_codes(self, code):
        assert classify_error(self._make_exc(code)) == ErrorCategory.PERMANENT

    # Kite retryable codes
    @pytest.mark.parametrize("code", ["TIMEOUT", "RETRIES_EXHAUSTED"])
    def test_kite_retryable_codes(self, code):
        assert classify_error(self._make_exc(code)) == ErrorCategory.RETRYABLE


# ──────────────────────────────────────────────────
# classify_error — HTTP response status
# ──────────────────────────────────────────────────


class TestHTTPResponseStatus:
    def _make_http_exc(self, status_code: int) -> Exception:
        exc = Exception(f"HTTP {status_code}")
        exc.response = SimpleNamespace(status_code=status_code)  # type: ignore[attr-defined]
        return exc

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_is_permanent(self, status):
        assert classify_error(self._make_http_exc(status)) == ErrorCategory.PERMANENT

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_5xx_is_retryable(self, status):
        assert classify_error(self._make_http_exc(status)) == ErrorCategory.RETRYABLE


# ──────────────────────────────────────────────────
# classify_error — keyword fallback
# ──────────────────────────────────────────────────


class TestKeywordFallback:
    @pytest.mark.parametrize(
        "msg",
        [
            "insufficient margin",
            "order rejected",
            "invalid quantity",
            "not allowed",
            "insufficient funds",
        ],
    )
    def test_permanent_keywords(self, msg):
        assert classify_error(Exception(msg)) == ErrorCategory.PERMANENT

    @pytest.mark.parametrize(
        "msg",
        [
            "connection timeout",
            "service temporarily unavailable",
            "rate limit exceeded",
            "please retry",
            "server overloaded",
        ],
    )
    def test_retryable_keywords(self, msg):
        assert classify_error(Exception(msg)) == ErrorCategory.RETRYABLE

    def test_unknown_message_returns_unknown(self):
        assert classify_error(Exception("something weird happened")) == ErrorCategory.UNKNOWN


# ──────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────


class TestConvenienceFunctions:
    def test_is_retryable_for_timeout(self):
        assert is_retryable(TimeoutError()) is True

    def test_is_retryable_for_unknown(self):
        # UNKNOWN is treated as retryable (with limited attempts)
        assert is_retryable(Exception("something weird")) is True

    def test_is_retryable_false_for_permanent(self):
        assert is_retryable(ValueError("bad input")) is False

    def test_is_permanent_for_value_error(self):
        assert is_permanent(ValueError("invalid")) is True

    def test_is_permanent_false_for_retryable(self):
        assert is_permanent(TimeoutError()) is False

    def test_is_permanent_false_for_unknown(self):
        assert is_permanent(Exception("something")) is False

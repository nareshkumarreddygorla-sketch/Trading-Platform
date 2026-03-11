"""
Structured logging configuration.

- Production (ENV=production): JSON-formatted log lines for machine parsing,
  with timestamp_iso, level, logger_name, message, and correlation_id fields.
- Development: human-readable log format with timestamps.
- Suppresses noisy third-party loggers.
- Sensitive data (passwords, tokens, API keys) is redacted from all log output.
- Trading-critical events (order placed, fill received, circuit breaker) include
  correlation_id when set via CorrelationContext.
"""

import contextvars
import json
import logging
import os
import re
import sys
import uuid
from datetime import UTC, datetime

# ── Sensitive data redaction ──
# Patterns that match common sensitive data in log messages.
# These are applied to every log message to prevent credential leaks.
_SENSITIVE_PATTERNS = [
    # password=..., password: ..., "password": "..."
    (re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+"), r"\1=***REDACTED***"),
    # token=..., access_token=..., refresh_token=..., bearer ...
    (re.compile(r"(?i)((?:access_|refresh_|auth_|bearer\s*)?token)\s*[=:]\s*\S+"), r"\1=***REDACTED***"),
    # api_key=..., apikey=..., secret_key=..., api_secret=...
    (re.compile(r"(?i)(api[_-]?key|api[_-]?secret|secret[_-]?key)\s*[=:]\s*\S+"), r"\1=***REDACTED***"),
    # Authorization: Bearer eyJ...
    (re.compile(r"(?i)(Authorization)\s*[=:]\s*\S+"), r"\1=***REDACTED***"),
    # JWT tokens (eyJ...)
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "***JWT_REDACTED***"),
    # TOTP secrets / OTP codes
    (re.compile(r"(?i)(totp[_-]?secret|otp)\s*[=:]\s*\S+"), r"\1=***REDACTED***"),
]


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data (passwords, tokens, API keys) from log records.

    Applied globally to prevent accidental credential leaks in log output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg and isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._redact(str(a)) if isinstance(a, str) else a for a in record.args)
        return True

    @staticmethod
    def _redact(text: str) -> str:
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


# ── Correlation ID context ──
# Set this in request middleware or at the start of a trading operation
# so every log line in that call-chain carries the same correlation_id.
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Return the current correlation ID, or None if not set."""
    return correlation_id_var.get()


def set_correlation_id(cid: str | None = None) -> str:
    """Set (or generate) a correlation ID for the current async/thread context."""
    if cid is None:
        cid = uuid.uuid4().hex[:16]
    correlation_id_var.set(cid)
    return cid


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON object (one per line).

    Fields: timestamp_iso, level, logger_name, message, module, function, line.
    Includes correlation_id when set in the current context.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp_iso": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Attach correlation_id from context (if set)
        cid = correlation_id_var.get()
        if cid is not None:
            log_obj["correlation_id"] = cid
        # Also pick up correlation_id if explicitly passed via extra={}
        extra_cid = getattr(record, "correlation_id", None)
        if extra_cid is not None:
            log_obj["correlation_id"] = extra_cid
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class HumanReadableFormatter(logging.Formatter):
    """Development-friendly formatter that includes correlation_id when available."""

    def format(self, record: logging.LogRecord) -> str:
        cid = correlation_id_var.get() or getattr(record, "correlation_id", None)
        if cid:
            self._fmt = "%(asctime)s %(levelname)-8s [%(correlation_id)s] %(name)s: %(message)s"
            record.correlation_id = cid  # type: ignore[attr-defined]
        else:
            self._fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
        self._style = logging.PercentStyle(self._fmt)
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        return super().format(record)


def configure_logging() -> None:
    """Set up root logger based on environment variables.

    Environment variables:
        ENV         - "production" selects JSON output; anything else uses human-readable.
        LOG_LEVEL   - Standard Python log level name (default: INFO).
    """
    env = os.environ.get("ENV", "development")
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Clear existing handlers to avoid duplicate output
    root.handlers.clear()

    # Apply sensitive data redaction filter globally (all handlers)
    sensitive_filter = SensitiveDataFilter()
    root.addFilter(sensitive_filter)

    handler = logging.StreamHandler(sys.stdout)
    if env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter())
    handler.addFilter(sensitive_filter)
    root.addHandler(handler)

    # ── Rotating file handler: logs/trading.log (50 MB, 5 backups) ──
    # Only added if the logs/ directory exists (avoids creating dirs in containers
    # where log collection is handled differently).
    _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    if os.path.isdir(_log_dir):
        try:
            from logging.handlers import RotatingFileHandler

            _log_file = os.path.join(_log_dir, "trading.log")
            file_handler = RotatingFileHandler(
                _log_file,
                maxBytes=50 * 1024 * 1024,  # 50 MB
                backupCount=5,
                encoding="utf-8",
            )
            # Always use JSON for file logs (machine-parseable)
            file_handler.setFormatter(JSONFormatter())
            file_handler.addFilter(sensitive_filter)
            root.addHandler(file_handler)
        except Exception as e:
            # Non-fatal: file logging is optional
            root.warning("Failed to configure file log handler: %s", e)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "httpx", "httpcore", "watchfiles", "matplotlib"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

"""
Structured logging configuration.

- Production (ENV=production): JSON-formatted log lines for machine parsing.
- Development: human-readable log format with timestamps.
- Suppresses noisy third-party loggers.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON object (one per line)."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


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

    handler = logging.StreamHandler(sys.stdout)
    if env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "httpx", "httpcore", "watchfiles", "matplotlib"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

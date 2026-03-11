"""
Sentry error tracking initialization.

Activates only when SENTRY_DSN is set. Gracefully degrades if sentry-sdk
is not installed.
"""

import logging
import os

logger = logging.getLogger(__name__)


def init_sentry():
    """Initialize Sentry SDK for error tracking and performance monitoring."""
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        logger.info("SENTRY_DSN not set — Sentry disabled")
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("ENV", "development"),
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            integrations=[FastApiIntegration(), StarletteIntegration()],
            send_default_pii=False,
        )
        logger.info("Sentry initialized (env=%s)", os.environ.get("ENV"))
    except ImportError:
        logger.warning("sentry-sdk not installed — Sentry disabled")

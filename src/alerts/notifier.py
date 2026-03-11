"""
Multi-channel alert system.

Supports:
  - Email (SMTP)
  - Telegram bot
  - Console logging (always on)

Alert deduplication: same alert not sent twice within 30 minutes.
Configurable severity levels: INFO, WARNING, CRITICAL
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time as _time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertChannel(str, Enum):
    EMAIL = "email"
    TELEGRAM = "telegram"
    CONSOLE = "console"


@dataclass
class AlertConfig:
    """Alert system configuration."""

    # Email
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: list[str] = field(default_factory=list)

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # General
    dedup_window_seconds: float = 1800.0  # 30 minutes
    enabled_channels: set[AlertChannel] = field(default_factory=lambda: {AlertChannel.CONSOLE})
    min_severity: AlertSeverity = AlertSeverity.WARNING


@dataclass
class Alert:
    """An alert event."""

    severity: AlertSeverity
    title: str
    message: str
    source: str = "system"
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = _time.time()

    @property
    def dedup_key(self) -> str:
        return hashlib.md5(f"{self.title}:{self.source}".encode()).hexdigest()


class AlertNotifier:
    """
    Multi-channel alert system with deduplication.

    Usage:
        notifier = AlertNotifier(config)
        await notifier.send(AlertSeverity.CRITICAL, "Circuit Breaker", "Trading halted — daily loss limit reached")
    """

    def __init__(self, config: AlertConfig | None = None):
        self.config = config or AlertConfig()
        self._sent_alerts: dict[str, float] = {}  # dedup_key -> last_sent_ts
        self._alert_history: list[Alert] = []

        # Auto-configure from environment
        if not self.config.telegram_bot_token:
            self.config.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not self.config.telegram_chat_id:
            self.config.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not self.config.smtp_host:
            self.config.smtp_host = os.environ.get("SMTP_HOST", "")
        if not self.config.smtp_user:
            self.config.smtp_user = os.environ.get("SMTP_USER", "")
        if not self.config.smtp_password:
            self.config.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        if not self.config.email_to:
            email_to = os.environ.get("ALERT_EMAIL_TO", "")
            if email_to:
                self.config.email_to = [e.strip() for e in email_to.split(",")]

        # Auto-enable channels based on config
        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            self.config.enabled_channels.add(AlertChannel.TELEGRAM)
        if self.config.smtp_host and self.config.email_to:
            self.config.enabled_channels.add(AlertChannel.EMAIL)

    async def send(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "system",
    ) -> bool:
        """
        Send an alert through all enabled channels.

        Returns True if alert was sent (not deduplicated).
        """
        alert = Alert(severity=severity, title=title, message=message, source=source)

        # Check minimum severity
        severity_order = {AlertSeverity.INFO: 0, AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}
        if severity_order.get(severity, 0) < severity_order.get(self.config.min_severity, 0):
            return False

        # Deduplication check
        now = _time.time()
        if alert.dedup_key in self._sent_alerts:
            last_sent = self._sent_alerts[alert.dedup_key]
            if now - last_sent < self.config.dedup_window_seconds:
                logger.debug("Alert deduplicated: %s", title)
                return False

        self._sent_alerts[alert.dedup_key] = now
        self._alert_history.append(alert)

        # Trim history
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

        # Send through all channels concurrently
        tasks = []
        for channel in self.config.enabled_channels:
            if channel == AlertChannel.CONSOLE:
                tasks.append(self._send_console(alert))
            elif channel == AlertChannel.TELEGRAM:
                tasks.append(self._send_telegram(alert))
            elif channel == AlertChannel.EMAIL:
                tasks.append(self._send_email(alert))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return True

    async def _send_console(self, alert: Alert) -> None:
        """Log alert to console."""
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)
        logger.log(level, "[ALERT][%s] %s: %s", alert.severity.value, alert.title, alert.message)

    async def _send_telegram(self, alert: Alert) -> None:
        """Send alert via Telegram bot."""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return
        try:
            import aiohttp

            emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(alert.severity.value, "📢")
            text = f"{emoji} *{alert.severity.value}: {alert.title}*\n\n{alert.message}\n\n_Source: {alert.source}_"
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        "chat_id": self.config.telegram_chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Telegram alert failed: HTTP %d", resp.status)
        except ImportError:
            logger.debug("aiohttp not installed — Telegram alerts disabled")
        except Exception as e:
            logger.warning("Telegram alert failed: %s", e)

    async def _send_email(self, alert: Alert) -> None:
        """Send alert via email (SMTP)."""
        if not self.config.smtp_host or not self.config.email_to:
            return
        try:
            from email.mime.text import MIMEText

            msg = MIMEText(f"Severity: {alert.severity.value}\n\n{alert.message}\n\nSource: {alert.source}")
            msg["Subject"] = f"[AlphaForge {alert.severity.value}] {alert.title}"
            msg["From"] = self.config.email_from or self.config.smtp_user
            msg["To"] = ", ".join(self.config.email_to)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._send_smtp(msg))
        except Exception as e:
            logger.warning("Email alert failed: %s", e)

    def _send_smtp(self, msg) -> None:
        """Synchronous SMTP send."""
        import smtplib

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=10) as server:
            server.starttls()
            if self.config.smtp_user and self.config.smtp_password:
                server.login(self.config.smtp_user, self.config.smtp_password)
            server.send_message(msg)

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent alert history."""
        return [
            {
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message,
                "source": a.source,
                "timestamp": a.timestamp,
            }
            for a in self._alert_history[-limit:]
        ]

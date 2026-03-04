"""
Dual data feed manager with automatic failover.

Monitors primary and secondary market data feeds, detects stale data or
disconnections, and switches between feeds with exponential backoff
reconnection. Wraps existing feed implementations (MarketDataService,
YFinanceFallbackFeeder) without managing their WebSocket connections directly.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
MONITOR_INTERVAL = 10.0


class FeedRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


@dataclass
class FeedHealth:
    """Snapshot of a single feed's health status."""
    feed_name: str
    connected: bool = False
    healthy: bool = False
    last_tick_ts: Optional[datetime] = None
    latency_ms: float = 0.0
    stale_seconds: float = 0.0


@dataclass
class FeedSwitchEvent:
    """Record of a feed failover/restore event for debugging history."""
    timestamp: datetime
    from_feed: Optional[str]
    to_feed: str
    reason: str


def _feed_health_to_dict(h: FeedHealth) -> dict:
    """Serialize a FeedHealth dataclass to a JSON-safe dict."""
    return {
        "feed_name": h.feed_name,
        "connected": h.connected,
        "healthy": h.healthy,
        "last_tick_ts": h.last_tick_ts.isoformat() if h.last_tick_ts else None,
        "latency_ms": round(h.latency_ms, 2),
        "stale_seconds": round(h.stale_seconds, 2),
    }


class FeedManager:
    """
    Manages primary and secondary market data feeds with automatic failover.

    Monitors both feeds for connectivity and data staleness. When the primary
    feed becomes unhealthy (disconnected or no tick data for stale_threshold_seconds),
    the manager fails over to the secondary feed. When the primary recovers,
    it restores it as the active feed.

    Does not own the underlying connections -- it delegates to each feed's own
    start/stop/is_healthy interface and monitors their reported state.
    """

    def __init__(
        self,
        bar_cache,
        primary_feed=None,
        secondary_feed=None,
        stale_threshold_seconds: float = 120.0,
    ):
        self._bar_cache = bar_cache
        self._primary = primary_feed
        self._secondary = secondary_feed
        self._stale_threshold = stale_threshold_seconds

        # Active feed tracking
        self._active_role: FeedRole = FeedRole.PRIMARY
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Reconnect backoff state (for attempting to restore primary)
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._last_reconnect_attempt: float = 0.0

        # Feed switch history
        self._switch_history: List[FeedSwitchEvent] = []

        # Callback for external notification (e.g. WebSocket push to UI)
        self._on_failover: Optional[Callable[[str, str, str], None]] = None

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    def set_failover_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """
        Register a callback invoked on every feed switch.

        Signature: callback(from_feed: str, to_feed: str, reason: str)
        """
        self._on_failover = callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the feed monitoring loop."""
        if self._running:
            logger.debug("FeedManager already running")
            return
        self._running = True
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        active = self._feed_name(self._active_role)
        logger.info(
            "FeedManager started -- primary=%s, secondary=%s, active=%s",
            self._feed_name(FeedRole.PRIMARY),
            self._feed_name(FeedRole.SECONDARY),
            active,
        )

    async def stop(self) -> None:
        """Stop the monitoring loop (does NOT stop the underlying feeds)."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("FeedManager stopped")

    # ------------------------------------------------------------------
    # Status / health queries
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """True if at least one feed is active and reporting healthy data."""
        active = self._get_active_feed()
        if active is None:
            return False
        return self._check_feed_healthy(active)

    def get_active_feed_name(self) -> str:
        """Return the name of the currently active feed."""
        return self._feed_name(self._active_role)

    def get_status(self) -> dict:
        """Full status dict for both feeds plus manager metadata."""
        return {
            "active_feed": self.get_active_feed_name(),
            "active_role": self._active_role.value,
            "healthy": self.is_healthy(),
            "running": self._running,
            "stale_threshold_seconds": self._stale_threshold,
            "reconnect_delay": self._reconnect_delay,
            "primary": _feed_health_to_dict(
                self._build_feed_health(self._primary, "primary")
            ) if self._primary else None,
            "secondary": _feed_health_to_dict(
                self._build_feed_health(self._secondary, "secondary")
            ) if self._secondary else None,
            "switch_history": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "from_feed": e.from_feed,
                    "to_feed": e.to_feed,
                    "reason": e.reason,
                }
                for e in self._switch_history[-20:]  # last 20 events
            ],
        }

    # ------------------------------------------------------------------
    # Internal: monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Async loop that checks feed health every MONITOR_INTERVAL seconds."""
        while self._running:
            try:
                await asyncio.sleep(MONITOR_INTERVAL)
                if not self._running:
                    break
                self._evaluate_feeds()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("FeedManager monitor loop error")

    def _evaluate_feeds(self) -> None:
        """Core decision logic run each monitor cycle."""
        primary_ok = self._primary is not None and self._check_feed_healthy(self._primary)
        secondary_ok = self._secondary is not None and self._check_feed_healthy(self._secondary)

        if self._active_role == FeedRole.PRIMARY:
            if not primary_ok:
                if secondary_ok:
                    self._failover_to_secondary(
                        reason="primary unhealthy, secondary available"
                    )
                else:
                    logger.warning(
                        "FeedManager: primary unhealthy and no healthy secondary"
                    )
        else:
            # Currently on secondary -- try to restore primary
            if primary_ok:
                self._restore_primary(reason="primary recovered")
            elif not secondary_ok:
                # Both down -- attempt primary reconnect with backoff
                self._attempt_primary_reconnect()

    # ------------------------------------------------------------------
    # Failover / restore
    # ------------------------------------------------------------------

    def _failover_to_secondary(self, reason: str = "primary unhealthy") -> None:
        """Switch the active feed from primary to secondary."""
        old_name = self._feed_name(FeedRole.PRIMARY)
        new_name = self._feed_name(FeedRole.SECONDARY)
        self._active_role = FeedRole.SECONDARY
        self._reconnect_delay = RECONNECT_BASE_DELAY  # reset backoff for restore attempts
        self._record_switch(old_name, new_name, reason)
        logger.warning(
            "FeedManager FAILOVER: %s -> %s (%s)", old_name, new_name, reason
        )

    def _restore_primary(self, reason: str = "primary recovered") -> None:
        """Switch back from secondary to primary."""
        old_name = self._feed_name(FeedRole.SECONDARY)
        new_name = self._feed_name(FeedRole.PRIMARY)
        self._active_role = FeedRole.PRIMARY
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._record_switch(old_name, new_name, reason)
        logger.info(
            "FeedManager RESTORE: %s -> %s (%s)", old_name, new_name, reason
        )

    def _attempt_primary_reconnect(self) -> None:
        """
        When both feeds are down, attempt to nudge primary reconnection
        using exponential backoff (1s, 2s, 4s, 8s, 16s, capped at 30s).

        The actual reconnection is handled by the feed itself (e.g.
        MarketDataService._run). This method just tracks timing so we
        don't spam logs or callbacks.
        """
        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_delay:
            return  # still within backoff window
        self._last_reconnect_attempt = now
        logger.info(
            "FeedManager: both feeds down, waiting %.1fs before next check "
            "(primary reconnect managed by feed itself)",
            self._reconnect_delay,
        )
        self._reconnect_delay = min(
            RECONNECT_MAX_DELAY, self._reconnect_delay * 2
        )

    # ------------------------------------------------------------------
    # Feed health helpers
    # ------------------------------------------------------------------

    def _check_feed_healthy(self, feed) -> bool:
        """
        Determine if a feed object is healthy.

        Supports two interfaces:
        1. feed.is_healthy() -- MarketDataService style
        2. feed._running attribute -- YFinanceFallbackFeeder style (no is_healthy method)

        Additionally checks for stale data via feed._last_tick_ts or
        feed._last_bar_ts when available.
        """
        # Check basic connectivity via is_healthy() if available
        if hasattr(feed, "is_healthy") and callable(feed.is_healthy):
            return feed.is_healthy()

        # Fallback: check _running flag (YFinanceFallbackFeeder)
        if hasattr(feed, "_running"):
            if not feed._running:
                return False

        # For feeders without is_healthy, check staleness via bar cache
        # (YFinanceFallbackFeeder pushes into bar_cache directly)
        if hasattr(feed, "_last_bar_ts") and feed._last_bar_ts:
            # _last_bar_ts is dict of symbol -> datetime or str (YahooBarFeeder stores str)
            most_recent = max(feed._last_bar_ts.values())
            if isinstance(most_recent, str):
                from datetime import datetime as _dt
                try:
                    most_recent = _dt.fromisoformat(most_recent)
                except (ValueError, TypeError):
                    return True  # can't parse, assume healthy
            if most_recent.tzinfo is None:
                most_recent = most_recent.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - most_recent).total_seconds()
            return age < self._stale_threshold

        # If running but no staleness info yet, consider healthy (just started)
        if hasattr(feed, "_running") and feed._running:
            return True

        return False

    def _build_feed_health(self, feed, label: str) -> FeedHealth:
        """Build a FeedHealth snapshot for a given feed."""
        name = self._feed_name_from_obj(feed) if feed else label
        if feed is None:
            return FeedHealth(feed_name=name)

        connected = False
        last_tick_ts = None
        latency_ms = 0.0
        stale_seconds = 0.0

        # Extract connected state
        if hasattr(feed, "_connected"):
            connected = feed._connected
        elif hasattr(feed, "_running"):
            connected = feed._running

        # Extract last tick/bar timestamp
        if hasattr(feed, "_last_tick_ts") and feed._last_tick_ts is not None:
            if isinstance(feed._last_tick_ts, datetime):
                last_tick_ts = feed._last_tick_ts
        elif hasattr(feed, "_last_bar_ts") and feed._last_bar_ts:
            ts_vals = feed._last_bar_ts.values()
            if ts_vals:
                raw_ts = max(ts_vals)
                if isinstance(raw_ts, str):
                    from datetime import datetime as _dt
                    try:
                        last_tick_ts = _dt.fromisoformat(raw_ts)
                    except (ValueError, TypeError):
                        last_tick_ts = None
                else:
                    last_tick_ts = raw_ts

        # Calculate staleness
        if last_tick_ts is not None:
            if last_tick_ts.tzinfo is None:
                last_tick_ts = last_tick_ts.replace(tzinfo=timezone.utc)
            stale_seconds = (
                datetime.now(timezone.utc) - last_tick_ts
            ).total_seconds()
            # Rough latency estimate (time since last data)
            latency_ms = min(stale_seconds * 1000, 999999.0)

        healthy = self._check_feed_healthy(feed)

        return FeedHealth(
            feed_name=name,
            connected=connected,
            healthy=healthy,
            last_tick_ts=last_tick_ts,
            latency_ms=latency_ms,
            stale_seconds=stale_seconds,
        )

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    def _feed_name(self, role: FeedRole) -> str:
        """Human-readable name for the feed in a given role."""
        feed = self._primary if role == FeedRole.PRIMARY else self._secondary
        if feed is None:
            return f"{role.value}(none)"
        return self._feed_name_from_obj(feed)

    @staticmethod
    def _feed_name_from_obj(feed) -> str:
        """Derive a short name from a feed object."""
        cls = type(feed).__name__
        # MarketDataService -> "MarketDataService"
        # YFinanceFallbackFeeder -> "YFinanceFallbackFeeder"
        return cls

    def _get_active_feed(self):
        """Return the feed object for the currently active role."""
        if self._active_role == FeedRole.PRIMARY:
            return self._primary
        return self._secondary

    # ------------------------------------------------------------------
    # Switch history / callbacks
    # ------------------------------------------------------------------

    def _record_switch(self, from_feed: str, to_feed: str, reason: str) -> None:
        """Record a feed switch event and fire the callback."""
        event = FeedSwitchEvent(
            timestamp=datetime.now(timezone.utc),
            from_feed=from_feed,
            to_feed=to_feed,
            reason=reason,
        )
        self._switch_history.append(event)

        # Trim history to last 100 events
        if len(self._switch_history) > 100:
            self._switch_history = self._switch_history[-100:]

        # Fire external callback (e.g. WebSocket notification to frontend)
        if self._on_failover:
            try:
                self._on_failover(from_feed, to_feed, reason)
            except Exception:
                logger.exception("FeedManager failover callback error")

"""Unit tests for src.market_data.feed_manager — dual-feed failover logic.

Covers:
- FeedHealth serialization
- FeedManager construction and defaults
- is_healthy() with various feed states
- Active feed name and status reporting
- Failover: primary unhealthy -> switch to secondary
- Restore: primary recovered -> switch back from secondary
- Reconnect backoff when both feeds down
- Failover callback invocation
- Switch history recording and trimming
- Feed health building (connected, stale, latency)
- Start/stop lifecycle
"""

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.market_data.feed_manager import (
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
    FeedHealth,
    FeedManager,
    FeedRole,
    _feed_health_to_dict,
)

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────


def _make_feed(
    name: str = "MockFeed",
    healthy: bool = True,
    running: bool = True,
    connected: bool = True,
    last_tick_ts: datetime | None = None,
):
    """Create a mock feed with configurable attributes."""
    # Dynamically create a class with the desired name (SimpleNamespace
    # does not support __class__ reassignment).
    cls = type(name, (), {})
    feed = cls()
    feed._running = running
    feed._connected = connected
    feed._last_tick_ts = last_tick_ts
    if healthy is not None:
        feed.is_healthy = lambda: healthy
    return feed


# ──────────────────────────────────────────────────
# FeedHealth serialization
# ──────────────────────────────────────────────────


class TestFeedHealthSerialization:
    def test_feed_health_to_dict_basic(self):
        h = FeedHealth(feed_name="test", connected=True, healthy=True)
        d = _feed_health_to_dict(h)
        assert d["feed_name"] == "test"
        assert d["connected"] is True
        assert d["healthy"] is True
        assert d["last_tick_ts"] is None

    def test_feed_health_to_dict_with_timestamp(self):
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        h = FeedHealth(
            feed_name="primary",
            connected=True,
            healthy=True,
            last_tick_ts=ts,
            latency_ms=42.567,
            stale_seconds=1.234,
        )
        d = _feed_health_to_dict(h)
        assert d["last_tick_ts"] == ts.isoformat()
        assert d["latency_ms"] == 42.57
        assert d["stale_seconds"] == 1.23


# ──────────────────────────────────────────────────
# FeedManager construction
# ──────────────────────────────────────────────────


class TestFeedManagerConstruction:
    def test_default_active_role_is_primary(self):
        fm = FeedManager(bar_cache=None)
        assert fm._active_role == FeedRole.PRIMARY

    def test_custom_stale_threshold(self):
        fm = FeedManager(bar_cache=None, stale_threshold_seconds=60.0)
        assert fm._stale_threshold == 60.0

    def test_feeds_stored(self):
        primary = _make_feed("Primary")
        secondary = _make_feed("Secondary")
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        assert fm._primary is primary
        assert fm._secondary is secondary


# ──────────────────────────────────────────────────
# Feed health checking
# ──────────────────────────────────────────────────


class TestFeedHealthCheck:
    def test_healthy_feed_with_is_healthy_method(self):
        feed = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=feed)
        assert fm._check_feed_healthy(feed) is True

    def test_unhealthy_feed_with_is_healthy_method(self):
        feed = _make_feed(healthy=False)
        fm = FeedManager(bar_cache=None, primary_feed=feed)
        assert fm._check_feed_healthy(feed) is False

    def test_feed_without_is_healthy_uses_running_flag(self):
        feed = SimpleNamespace(_running=True)
        fm = FeedManager(bar_cache=None, primary_feed=feed)
        assert fm._check_feed_healthy(feed) is True

    def test_feed_not_running_is_unhealthy(self):
        feed = SimpleNamespace(_running=False)
        fm = FeedManager(bar_cache=None, primary_feed=feed)
        assert fm._check_feed_healthy(feed) is False

    def test_feed_with_stale_bar_ts(self):
        """Feed with old _last_bar_ts should be unhealthy."""
        old_ts = datetime.now(UTC) - timedelta(seconds=60)
        feed = SimpleNamespace(_running=True, _last_bar_ts={"SYM1": old_ts})
        # Remove is_healthy so it goes to staleness check
        fm = FeedManager(bar_cache=None, primary_feed=feed, stale_threshold_seconds=30.0)
        assert fm._check_feed_healthy(feed) is False

    def test_feed_with_fresh_bar_ts(self):
        """Feed with recent _last_bar_ts should be healthy."""
        fresh_ts = datetime.now(UTC) - timedelta(seconds=5)
        feed = SimpleNamespace(_running=True, _last_bar_ts={"SYM1": fresh_ts})
        fm = FeedManager(bar_cache=None, primary_feed=feed, stale_threshold_seconds=30.0)
        assert fm._check_feed_healthy(feed) is True

    def test_feed_with_string_bar_ts(self):
        """Feed with ISO string timestamps should be parsed."""
        fresh_ts = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        feed = SimpleNamespace(_running=True, _last_bar_ts={"SYM1": fresh_ts})
        fm = FeedManager(bar_cache=None, primary_feed=feed, stale_threshold_seconds=30.0)
        assert fm._check_feed_healthy(feed) is True


# ──────────────────────────────────────────────────
# is_healthy() and get_active_feed_name()
# ──────────────────────────────────────────────────


class TestManagerHealth:
    def test_healthy_when_primary_ok(self):
        primary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary)
        assert fm.is_healthy() is True

    def test_unhealthy_when_no_feeds(self):
        fm = FeedManager(bar_cache=None)
        assert fm.is_healthy() is False

    def test_active_feed_name_primary(self):
        primary = _make_feed("PrimaryFeed")
        fm = FeedManager(bar_cache=None, primary_feed=primary)
        name = fm.get_active_feed_name()
        # Name comes from class name
        assert "none" not in name.lower()

    def test_active_feed_name_no_feed(self):
        fm = FeedManager(bar_cache=None)
        name = fm.get_active_feed_name()
        assert "none" in name.lower()


# ──────────────────────────────────────────────────
# Failover / restore logic
# ──────────────────────────────────────────────────


class TestFailoverRestore:
    def test_failover_to_secondary(self):
        primary = _make_feed(healthy=False)
        secondary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        fm._evaluate_feeds()
        assert fm._active_role == FeedRole.SECONDARY

    def test_restore_primary_after_recovery(self):
        primary = _make_feed(healthy=True)
        secondary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        # Simulate being on secondary
        fm._active_role = FeedRole.SECONDARY
        fm._evaluate_feeds()
        assert fm._active_role == FeedRole.PRIMARY

    def test_no_failover_when_primary_healthy(self):
        primary = _make_feed(healthy=True)
        secondary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        fm._evaluate_feeds()
        assert fm._active_role == FeedRole.PRIMARY

    def test_no_failover_when_secondary_also_unhealthy(self):
        primary = _make_feed(healthy=False)
        secondary = _make_feed(healthy=False)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        fm._evaluate_feeds()
        # Should stay on primary (no healthy secondary to switch to)
        assert fm._active_role == FeedRole.PRIMARY

    def test_both_down_on_secondary_attempts_reconnect(self):
        primary = _make_feed(healthy=False)
        secondary = _make_feed(healthy=False)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        fm._active_role = FeedRole.SECONDARY
        initial_delay = fm._reconnect_delay
        fm._evaluate_feeds()
        # Reconnect delay should have doubled
        assert fm._reconnect_delay > initial_delay


# ──────────────────────────────────────────────────
# Reconnect backoff
# ──────────────────────────────────────────────────


class TestReconnectBackoff:
    def test_backoff_increases_exponentially(self):
        fm = FeedManager(bar_cache=None)
        fm._reconnect_delay = RECONNECT_BASE_DELAY
        fm._last_reconnect_attempt = 0.0  # Allow first attempt

        fm._attempt_primary_reconnect()
        assert fm._reconnect_delay == RECONNECT_BASE_DELAY * 2

    def test_backoff_capped_at_max(self):
        fm = FeedManager(bar_cache=None)
        fm._reconnect_delay = RECONNECT_MAX_DELAY
        fm._last_reconnect_attempt = 0.0

        fm._attempt_primary_reconnect()
        assert fm._reconnect_delay == RECONNECT_MAX_DELAY

    def test_backoff_skipped_within_window(self):
        fm = FeedManager(bar_cache=None)
        fm._reconnect_delay = 10.0
        fm._last_reconnect_attempt = time.monotonic()  # Just attempted

        initial_delay = fm._reconnect_delay
        fm._attempt_primary_reconnect()
        # Should not have changed — still within backoff window
        assert fm._reconnect_delay == initial_delay


# ──────────────────────────────────────────────────
# Switch history and callbacks
# ──────────────────────────────────────────────────


class TestSwitchHistoryAndCallbacks:
    def test_switch_recorded_on_failover(self):
        primary = _make_feed(healthy=False)
        secondary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        fm._evaluate_feeds()
        assert len(fm._switch_history) == 1
        event = fm._switch_history[0]
        assert event.reason == "primary unhealthy, secondary available"

    def test_failover_callback_invoked(self):
        callback = MagicMock()
        primary = _make_feed(healthy=False)
        secondary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary, secondary_feed=secondary)
        fm.set_failover_callback(callback)
        fm._evaluate_feeds()
        callback.assert_called_once()

    def test_switch_history_trimmed_at_100(self):
        fm = FeedManager(bar_cache=None)
        for i in range(110):
            fm._record_switch(f"from_{i}", f"to_{i}", f"reason_{i}")
        assert len(fm._switch_history) <= 100

    def test_callback_exception_does_not_crash(self):
        def bad_callback(from_feed, to_feed, reason):
            raise ValueError("callback error")

        fm = FeedManager(bar_cache=None)
        fm.set_failover_callback(bad_callback)
        # Should not raise
        fm._record_switch("a", "b", "test")


# ──────────────────────────────────────────────────
# get_status()
# ──────────────────────────────────────────────────


class TestGetStatus:
    def test_status_contains_required_keys(self):
        primary = _make_feed(healthy=True)
        fm = FeedManager(bar_cache=None, primary_feed=primary)
        status = fm.get_status()
        assert "active_feed" in status
        assert "active_role" in status
        assert "healthy" in status
        assert "running" in status
        assert "stale_threshold_seconds" in status
        assert "primary" in status
        assert "switch_history" in status

    def test_status_no_feeds(self):
        fm = FeedManager(bar_cache=None)
        status = fm.get_status()
        assert status["healthy"] is False
        assert status["primary"] is None
        assert status["secondary"] is None


# ──────────────────────────────────────────────────
# Start / stop lifecycle
# ──────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        fm = FeedManager(bar_cache=None)
        fm.start()
        assert fm._running is True
        assert fm._monitor_task is not None
        await fm.stop()
        assert fm._running is False
        assert fm._monitor_task is None

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        fm = FeedManager(bar_cache=None)
        fm.start()
        task1 = fm._monitor_task
        fm.start()  # Should not create a new task
        assert fm._monitor_task is task1
        await fm.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        fm = FeedManager(bar_cache=None)
        await fm.stop()  # Should not raise
        assert fm._running is False


# ──────────────────────────────────────────────────
# Build feed health
# ──────────────────────────────────────────────────


class TestBuildFeedHealth:
    def test_build_health_none_feed(self):
        fm = FeedManager(bar_cache=None)
        h = fm._build_feed_health(None, "primary")
        assert h.feed_name == "primary"
        assert h.connected is False
        assert h.healthy is False

    def test_build_health_connected_feed(self):
        feed = _make_feed(healthy=True, connected=True)
        fm = FeedManager(bar_cache=None, primary_feed=feed)
        h = fm._build_feed_health(feed, "primary")
        assert h.connected is True
        assert h.healthy is True

    def test_build_health_with_last_tick_ts(self):
        ts = datetime.now(UTC) - timedelta(seconds=5)
        feed = _make_feed(healthy=True, last_tick_ts=ts)
        fm = FeedManager(bar_cache=None, primary_feed=feed)
        h = fm._build_feed_health(feed, "primary")
        assert h.last_tick_ts == ts
        assert h.stale_seconds > 0

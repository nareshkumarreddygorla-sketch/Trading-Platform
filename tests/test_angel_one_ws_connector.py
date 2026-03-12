"""Unit tests for src.market_data.angel_one_ws_connector — WebSocket connector.

Covers:
- _parse_smartapi_tick: LTP parsing, token reverse-resolution, missing fields
- _backoff_delay: exponential backoff schedule
- AngelOneWsConnector construction and defaults
- is_healthy(): connected/disconnected/stale/noop states
- get_reconnect_metrics(): counters and downtime tracking
- _set_connected(): state transitions and callback invocation
- _payload_to_tick(): nested payloads, lists, non-dict
- subscribe() sync alias
- stop() lifecycle
"""

import time
from datetime import UTC, datetime, timedelta

import pytest

from src.core.events import Exchange
from src.market_data.angel_one_ws_connector import (
    _BACKOFF_SCHEDULE,
    AngelOneWsConnector,
    _backoff_delay,
    _parse_smartapi_tick,
)

# ──────────────────────────────────────────────────
# _parse_smartapi_tick
# ──────────────────────────────────────────────────


class TestParseSmartapiTick:
    def test_basic_ltp_payload(self):
        payload = {"ltp": 2500.0, "tradingsymbol": "RELIANCE-EQ", "volume": 1000}
        tick = _parse_smartapi_tick(payload, Exchange.NSE)
        assert tick is not None
        assert tick.symbol == "RELIANCE"
        assert tick.price == 2500.0
        assert tick.exchange == Exchange.NSE

    def test_last_price_key(self):
        payload = {"last_price": 1500.0, "symbol": "INFY", "last_quantity": 500}
        tick = _parse_smartapi_tick(payload, Exchange.NSE)
        assert tick is not None
        assert tick.price == 1500.0

    def test_missing_ltp_returns_none(self):
        payload = {"symbol": "TCS", "volume": 100}
        tick = _parse_smartapi_tick(payload, Exchange.NSE)
        assert tick is None

    def test_token_reverse_resolution(self):
        payload = {"ltp": 2500.0, "symbol": "2885", "volume": 100}
        token_map = {"2885": "RELIANCE"}
        tick = _parse_smartapi_tick(payload, Exchange.NSE, token_to_symbol=token_map)
        assert tick is not None
        assert tick.symbol == "RELIANCE"

    def test_no_reverse_resolution_when_not_numeric(self):
        payload = {"ltp": 2500.0, "symbol": "RELIANCE", "volume": 100}
        token_map = {"2885": "RELIANCE"}
        tick = _parse_smartapi_tick(payload, Exchange.NSE, token_to_symbol=token_map)
        assert tick.symbol == "RELIANCE"

    def test_eq_suffix_stripped(self):
        payload = {"ltp": 100.0, "tradingsymbol": "TCS-EQ", "volume": 50}
        tick = _parse_smartapi_tick(payload, Exchange.NSE)
        assert tick.symbol == "TCS"

    def test_tk_key_used_as_symbol(self):
        payload = {"ltp": 100.0, "tk": "2885", "volume": 50}
        tick = _parse_smartapi_tick(payload, Exchange.NSE)
        assert tick is not None
        assert tick.symbol == "2885"

    def test_zero_volume_default(self):
        payload = {"ltp": 100.0, "tradingsymbol": "X"}
        tick = _parse_smartapi_tick(payload, Exchange.NSE)
        assert tick is not None
        assert tick.size == 0


# ──────────────────────────────────────────────────
# _backoff_delay
# ──────────────────────────────────────────────────


class TestBackoffDelay:
    def test_first_attempt(self):
        assert _backoff_delay(0) == float(_BACKOFF_SCHEDULE[0])

    def test_schedule_progression(self):
        for i, expected in enumerate(_BACKOFF_SCHEDULE):
            assert _backoff_delay(i) == float(expected)

    def test_beyond_schedule_caps_at_last(self):
        # Attempts beyond schedule length should cap at last value
        last = float(_BACKOFF_SCHEDULE[-1])
        assert _backoff_delay(100) == last
        assert _backoff_delay(len(_BACKOFF_SCHEDULE)) == last


# ──────────────────────────────────────────────────
# AngelOneWsConnector construction
# ──────────────────────────────────────────────────


class TestConnectorConstruction:
    def test_default_exchange(self):
        c = AngelOneWsConnector("key", "secret", "token")
        assert c._exchange == Exchange.NSE

    def test_custom_exchange(self):
        c = AngelOneWsConnector("key", "secret", "token", exchange="BSE")
        assert c._exchange == Exchange.BSE

    def test_invalid_exchange_defaults_to_nse(self):
        c = AngelOneWsConnector("key", "secret", "token", exchange="INVALID")
        assert c._exchange == Exchange.NSE

    def test_initial_state(self):
        c = AngelOneWsConnector("key", "secret", "token")
        assert c._connected is False
        assert c._closed is False
        assert c._intentional_stop is False
        assert c._reconnect_count == 0
        assert c._last_tick_ts is None

    def test_custom_heartbeat_timeout(self):
        c = AngelOneWsConnector("key", "secret", "token", heartbeat_timeout_seconds=120)
        assert c.heartbeat_timeout == 120


# ──────────────────────────────────────────────────
# is_healthy()
# ──────────────────────────────────────────────────


class TestIsHealthy:
    def test_not_connected_is_unhealthy(self):
        c = AngelOneWsConnector("key", "secret", "token")
        assert c.is_healthy() is False

    def test_connected_with_recent_message_is_healthy(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._connected = True
        c._last_message_ts = datetime.now(UTC)
        assert c.is_healthy() is True

    def test_connected_with_stale_message_is_unhealthy(self):
        c = AngelOneWsConnector("key", "secret", "token", heartbeat_timeout_seconds=30)
        c._connected = True
        c._last_message_ts = datetime.now(UTC) - timedelta(seconds=60)
        assert c.is_healthy() is False

    def test_connected_no_message_yet_is_healthy(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._connected = True
        c._last_message_ts = None
        assert c.is_healthy() is True

    def test_noop_ws_always_unhealthy(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._connected = True
        c._using_noop_ws = True
        assert c.is_healthy() is False


# ──────────────────────────────────────────────────
# _set_connected state transitions
# ──────────────────────────────────────────────────


class TestSetConnected:
    def test_connected_to_disconnected_starts_downtime_tracking(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._connected = True
        c._set_connected(False)
        assert c._connected is False
        assert c._disconnect_started_at is not None

    def test_disconnected_to_connected_accumulates_downtime(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._connected = False
        c._disconnect_started_at = time.monotonic() - 5.0  # 5 seconds ago
        c._set_connected(True)
        assert c._connected is True
        assert c._total_disconnect_seconds >= 4.0  # Allow some slack
        assert c._disconnect_started_at is None

    def test_on_feed_unhealthy_callback_on_disconnect(self):
        called = []
        c = AngelOneWsConnector(
            "key",
            "secret",
            "token",
            on_feed_unhealthy=lambda: called.append(True),
        )
        c._connected = True
        c._set_connected(False)
        assert len(called) == 1

    def test_on_feed_unhealthy_exception_does_not_crash(self):
        def bad_callback():
            raise ValueError("callback error")

        c = AngelOneWsConnector("key", "secret", "token", on_feed_unhealthy=bad_callback)
        c._connected = True
        c._set_connected(False)  # Should not raise


# ──────────────────────────────────────────────────
# get_reconnect_metrics()
# ──────────────────────────────────────────────────


class TestReconnectMetrics:
    def test_initial_metrics(self):
        c = AngelOneWsConnector("key", "secret", "token")
        m = c.get_reconnect_metrics()
        assert m["reconnect_count"] == 0
        assert m["is_connected"] is False
        assert m["is_healthy"] is False
        assert m["total_disconnect_seconds"] >= 0

    def test_metrics_after_reconnect(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._reconnect_count = 3
        c._connected = True
        c._last_message_ts = datetime.now(UTC)
        c._total_disconnect_seconds = 15.0
        m = c.get_reconnect_metrics()
        assert m["reconnect_count"] == 3
        assert m["is_connected"] is True
        assert m["total_disconnect_seconds"] == 15.0

    def test_metrics_include_current_downtime(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._disconnect_started_at = time.monotonic() - 10.0
        m = c.get_reconnect_metrics()
        assert m["total_disconnect_seconds"] >= 9.0


# ──────────────────────────────────────────────────
# _payload_to_tick
# ──────────────────────────────────────────────────


class TestPayloadToTick:
    def test_direct_tick_payload(self):
        c = AngelOneWsConnector("key", "secret", "token")
        ticks = c._payload_to_tick({"ltp": 100.0, "tradingsymbol": "TCS", "volume": 50})
        assert len(ticks) == 1
        assert ticks[0].price == 100.0

    def test_nested_data_key(self):
        c = AngelOneWsConnector("key", "secret", "token")
        ticks = c._payload_to_tick({"data": {"ltp": 200.0, "tradingsymbol": "INFY", "volume": 10}})
        assert len(ticks) == 1
        assert ticks[0].price == 200.0

    def test_list_of_payloads(self):
        c = AngelOneWsConnector("key", "secret", "token")
        ticks = c._payload_to_tick(
            [
                {"ltp": 100.0, "tradingsymbol": "A", "volume": 1},
                {"ltp": 200.0, "tradingsymbol": "B", "volume": 2},
            ]
        )
        assert len(ticks) == 2

    def test_non_dict_returns_empty(self):
        c = AngelOneWsConnector("key", "secret", "token")
        assert c._payload_to_tick("not a dict") == []
        assert c._payload_to_tick(42) == []

    def test_empty_dict_returns_empty(self):
        c = AngelOneWsConnector("key", "secret", "token")
        assert c._payload_to_tick({}) == []


# ──────────────────────────────────────────────────
# subscribe() sync alias
# ──────────────────────────────────────────────────


class TestSubscribeSync:
    def test_subscribe_stores_symbols(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c.subscribe(["RELIANCE", "TCS", "INFY"])
        assert c._symbols == ["RELIANCE", "TCS", "INFY"]


# ──────────────────────────────────────────────────
# stop() lifecycle
# ──────────────────────────────────────────────────


class TestStopLifecycle:
    @pytest.mark.asyncio
    async def test_stop_sets_closed(self):
        c = AngelOneWsConnector("key", "secret", "token")
        await c.stop()
        assert c._closed is True
        assert c._intentional_stop is True
        assert c._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_alias(self):
        c = AngelOneWsConnector("key", "secret", "token")
        await c.disconnect()
        assert c._closed is True

    @pytest.mark.asyncio
    async def test_stop_finalizes_downtime(self):
        c = AngelOneWsConnector("key", "secret", "token")
        c._disconnect_started_at = time.monotonic() - 5.0
        await c.stop()
        assert c._total_disconnect_seconds >= 4.0
        assert c._disconnect_started_at is None

    @pytest.mark.asyncio
    async def test_connect_after_close_raises(self):
        c = AngelOneWsConnector("key", "secret", "token")
        await c.stop()
        with pytest.raises(RuntimeError, match="closed"):
            await c.connect()


# ──────────────────────────────────────────────────
# get_last_tick_ts
# ──────────────────────────────────────────────────


class TestGetLastTickTs:
    def test_initial_none(self):
        c = AngelOneWsConnector("key", "secret", "token")
        assert c.get_last_tick_ts() is None

    def test_returns_last_ts(self):
        c = AngelOneWsConnector("key", "secret", "token")
        ts = datetime.now(UTC)
        c._last_tick_ts = ts
        assert c.get_last_tick_ts() == ts

"""
Phase 1 Critical Safety Fixes -- Comprehensive Test Suite
==========================================================

Tests the four Phase 1 safety fixes:

1. Kill Switch NameError Fix
   - _MANUAL_DISARM_ONLY accessible at module level
   - check_auto_disarm() does not crash with NameError

2. Kill Switch Reason Overwrite Fix
   - Severity-based arm rejection prevents dangerous downgrades
   - Manual reasons cannot be overwritten by auto-disarmable reasons
   - Auto-disarm returns False for manual-only reasons

3. Redis Fail-Open Fix
   - IdempotencyStore.redis_connected is False when Redis unavailable
   - IdempotencyStore.is_available() is True (backward compat)
   - RedisClusterReservation fail_open=True returns True when Redis down
   - RedisClusterReservation fail_open=False returns False when Redis down

4. Audit Trail Write-Through + Hash Chain
   - Each appended event triggers immediate flush (not batched)
   - Hash chain is sequential (each event has a different hash)
   - Hash chain includes previous hash (tampering invalidates chain)
"""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from src.compliance.audit_trail import (
    SEBIAuditTrail,
)
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.kill_switch import (
    _MANUAL_DISARM_ONLY,
    KillReason,
    KillSwitch,
)
from src.execution.order_entry.redis_cluster_reservation import RedisClusterReservation

# ========================================================================
# Section 1: Kill Switch NameError Fix
# ========================================================================


class TestKillSwitchNameErrorFix:
    """Verify _MANUAL_DISARM_ONLY is a module-level constant and accessible
    everywhere that needs it, preventing the NameError that previously
    crashed check_auto_disarm()."""

    def test_manual_disarm_only_exists_at_module_level(self):
        """_MANUAL_DISARM_ONLY should be importable from the kill_switch module."""
        assert _MANUAL_DISARM_ONLY is not None
        assert isinstance(_MANUAL_DISARM_ONLY, frozenset)

    def test_manual_disarm_only_contains_expected_reasons(self):
        """The frozenset must contain all reasons that require human intervention."""
        expected = {
            KillReason.MANUAL,
            KillReason.MAX_DAILY_LOSS,
            KillReason.FILL_MISMATCH,
            KillReason.MAX_DRAWDOWN,
        }
        assert _MANUAL_DISARM_ONLY == expected

    def test_auto_disarmable_reasons_not_in_manual_only(self):
        """Auto-disarmable reasons must NOT be in _MANUAL_DISARM_ONLY."""
        auto_disarmable = {
            KillReason.BROKER_LATENCY_SPIKE,
            KillReason.INDIA_VIX_SPIKE,
            KillReason.MARKET_FEED_FAILURE,
            KillReason.DRIFT_SPIKE,
            KillReason.REJECTION_SPIKE,
        }
        for reason in auto_disarmable:
            assert reason not in _MANUAL_DISARM_ONLY, (
                f"{reason} should be auto-disarmable but is in _MANUAL_DISARM_ONLY"
            )

    @pytest.mark.asyncio
    async def test_check_auto_disarm_no_nameerror_when_armed_manual(self):
        """check_auto_disarm() must not raise NameError when armed with a
        manual-only reason. This was the original bug."""
        ks = KillSwitch()
        await ks.arm(KillReason.MANUAL, "operator pressed button")
        # Previously this crashed with NameError: '_MANUAL_DISARM_ONLY' not defined
        result = await ks.check_auto_disarm(broker_healthy=True)
        assert result is False  # MANUAL never auto-disarms

    @pytest.mark.asyncio
    async def test_check_auto_disarm_no_nameerror_when_armed_max_drawdown(self):
        """check_auto_disarm() must not crash for MAX_DRAWDOWN reason."""
        ks = KillSwitch()
        await ks.arm(KillReason.MAX_DRAWDOWN, "drawdown exceeded 5%")
        result = await ks.check_auto_disarm(broker_healthy=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_auto_disarm_no_nameerror_when_armed_fill_mismatch(self):
        """check_auto_disarm() must not crash for FILL_MISMATCH reason."""
        ks = KillSwitch()
        await ks.arm(KillReason.FILL_MISMATCH, "broker vs local fill mismatch")
        result = await ks.check_auto_disarm(broker_healthy=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_auto_disarm_no_nameerror_when_armed_max_daily_loss(self):
        """check_auto_disarm() must not crash for MAX_DAILY_LOSS reason."""
        ks = KillSwitch()
        await ks.arm(KillReason.MAX_DAILY_LOSS, "daily loss exceeded 2%")
        result = await ks.check_auto_disarm(broker_healthy=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_auto_disarm_no_crash_when_not_armed(self):
        """check_auto_disarm() should return False cleanly when not armed."""
        ks = KillSwitch()
        result = await ks.check_auto_disarm(broker_healthy=True)
        assert result is False


# ========================================================================
# Section 2: Kill Switch Reason Overwrite Fix
# ========================================================================


class TestKillSwitchReasonOverwriteFix:
    """Verify that arming with a lower-severity reason cannot overwrite
    a higher-severity (manual-only) reason."""

    @pytest.mark.asyncio
    async def test_max_drawdown_then_market_feed_failure_keeps_max_drawdown(self):
        """Arming with MAX_DRAWDOWN then re-arming with MARKET_FEED_FAILURE
        must keep MAX_DRAWDOWN (manual-only cannot be downgraded to auto-disarmable)."""
        ks = KillSwitch()
        await ks.arm(KillReason.MAX_DRAWDOWN, "drawdown hit 5%")
        state_before = await ks.get_state()
        assert state_before.reason == KillReason.MAX_DRAWDOWN

        # Attempt to overwrite with auto-disarmable reason
        await ks.arm(KillReason.MARKET_FEED_FAILURE, "feed stalled")
        state_after = await ks.get_state()

        # Must still be MAX_DRAWDOWN -- the downgrade was rejected
        assert state_after.reason == KillReason.MAX_DRAWDOWN
        assert state_after.armed is True

    @pytest.mark.asyncio
    async def test_market_feed_failure_then_max_drawdown_upgrades(self):
        """Arming with MARKET_FEED_FAILURE then re-arming with MAX_DRAWDOWN
        should upgrade the reason (auto-disarmable -> manual-only is allowed)."""
        ks = KillSwitch()
        await ks.arm(KillReason.MARKET_FEED_FAILURE, "feed stalled")
        state_before = await ks.get_state()
        assert state_before.reason == KillReason.MARKET_FEED_FAILURE

        # Upgrade to a more severe, manual-only reason
        await ks.arm(KillReason.MAX_DRAWDOWN, "drawdown hit 5%")
        state_after = await ks.get_state()

        assert state_after.reason == KillReason.MAX_DRAWDOWN
        assert state_after.armed is True

    @pytest.mark.asyncio
    async def test_manual_then_broker_latency_spike_rejected(self):
        """Arming with MANUAL then re-arming with BROKER_LATENCY_SPIKE
        must be rejected (manual-only cannot be overwritten by auto-disarmable)."""
        ks = KillSwitch()
        await ks.arm(KillReason.MANUAL, "operator decision")
        state_before = await ks.get_state()
        assert state_before.reason == KillReason.MANUAL

        # Attempt to overwrite with auto-disarmable reason
        await ks.arm(KillReason.BROKER_LATENCY_SPIKE, "latency > 5s")
        state_after = await ks.get_state()

        # Must still be MANUAL
        assert state_after.reason == KillReason.MANUAL
        assert state_after.armed is True

    @pytest.mark.asyncio
    async def test_auto_disarm_returns_false_for_manual_only_reasons(self):
        """check_auto_disarm must return False for every manual-only reason."""
        for reason in _MANUAL_DISARM_ONLY:
            ks = KillSwitch()
            await ks.arm(reason, f"test {reason.value}")
            result = await ks.check_auto_disarm(broker_healthy=True, vix_value=10.0)
            assert result is False, f"check_auto_disarm returned True for manual-only reason {reason.value}"

    @pytest.mark.asyncio
    async def test_manual_only_to_manual_only_overwrites(self):
        """Re-arming from one manual-only reason to another manual-only reason
        should succeed (both are equally severe)."""
        ks = KillSwitch()
        await ks.arm(KillReason.MANUAL, "operator decision")
        state = await ks.get_state()
        assert state.reason == KillReason.MANUAL

        await ks.arm(KillReason.MAX_DRAWDOWN, "drawdown exceeded")
        state = await ks.get_state()
        assert state.reason == KillReason.MAX_DRAWDOWN

    @pytest.mark.asyncio
    async def test_auto_to_auto_overwrites(self):
        """Re-arming from one auto-disarmable reason to another should succeed."""
        ks = KillSwitch()
        await ks.arm(KillReason.BROKER_LATENCY_SPIKE, "latency spike")
        state = await ks.get_state()
        assert state.reason == KillReason.BROKER_LATENCY_SPIKE

        await ks.arm(KillReason.INDIA_VIX_SPIKE, "VIX spiked")
        state = await ks.get_state()
        assert state.reason == KillReason.INDIA_VIX_SPIKE

    @pytest.mark.asyncio
    async def test_max_daily_loss_then_drift_spike_rejected(self):
        """MAX_DAILY_LOSS (manual-only) cannot be overwritten by DRIFT_SPIKE (auto)."""
        ks = KillSwitch()
        await ks.arm(KillReason.MAX_DAILY_LOSS, "daily loss exceeded")
        await ks.arm(KillReason.DRIFT_SPIKE, "model drift detected")
        state = await ks.get_state()
        assert state.reason == KillReason.MAX_DAILY_LOSS

    @pytest.mark.asyncio
    async def test_fill_mismatch_then_rejection_spike_rejected(self):
        """FILL_MISMATCH (manual-only) cannot be overwritten by REJECTION_SPIKE (auto)."""
        ks = KillSwitch()
        await ks.arm(KillReason.FILL_MISMATCH, "fill reconciliation failed")
        await ks.arm(KillReason.REJECTION_SPIKE, "5 rejections in 20 orders")
        state = await ks.get_state()
        assert state.reason == KillReason.FILL_MISMATCH

    @pytest.mark.asyncio
    async def test_broker_latency_auto_disarms_after_healthy_checks(self):
        """BROKER_LATENCY_SPIKE should auto-disarm after 3 consecutive healthy checks."""
        ks = KillSwitch()
        await ks.arm(KillReason.BROKER_LATENCY_SPIKE, "latency > 5000ms")
        assert await ks.is_armed() is True

        # 3 consecutive healthy checks required (default _required_healthy_checks=3)
        result1 = await ks.check_auto_disarm(broker_healthy=True)
        assert result1 is False  # only 1 healthy check so far
        result2 = await ks.check_auto_disarm(broker_healthy=True)
        assert result2 is False  # only 2
        result3 = await ks.check_auto_disarm(broker_healthy=True)
        assert result3 is True  # 3 -- auto-disarmed

        assert await ks.is_armed() is False


# ========================================================================
# Section 3: Redis Fail-Open Fix
# ========================================================================


class TestRedisFailOpenFix:
    """Verify that Redis-dependent components degrade gracefully when
    Redis is unavailable, and that the fail_open flag is respected."""

    # ---- IdempotencyStore ----

    @pytest.mark.asyncio
    async def test_idempotency_redis_connected_false_when_unavailable(self):
        """redis_connected must be False when Redis is not reachable."""
        store = IdempotencyStore(redis_url="redis://localhost:59999/0")
        # Force the connection check
        await store.is_available()
        assert store.redis_connected is False

    @pytest.mark.asyncio
    async def test_idempotency_is_available_always_true(self):
        """is_available() must return True regardless of Redis status
        (backward compat -- falls back to in-memory)."""
        store = IdempotencyStore(redis_url="redis://localhost:59999/0")
        available = await store.is_available()
        assert available is True

    @pytest.mark.asyncio
    async def test_idempotency_fallback_to_memory_set_and_get(self):
        """With Redis down, set/get should work via in-memory fallback."""
        store = IdempotencyStore(redis_url="redis://localhost:59999/0")
        await store.is_available()  # trigger connection check

        ok = await store.set("test-key-1", "order-abc", status="PENDING")
        assert ok is True

        result = await store.get("test-key-1")
        assert result is not None
        assert result["order_id"] == "order-abc"
        assert result["status"] == "PENDING"

    @pytest.mark.asyncio
    async def test_idempotency_fallback_nx_semantics(self):
        """In-memory fallback must respect NX (set-if-not-exists) semantics."""
        store = IdempotencyStore(redis_url="redis://localhost:59999/0")
        await store.is_available()

        ok1 = await store.set("dup-key", "order-1")
        assert ok1 is True

        ok2 = await store.set("dup-key", "order-2")
        assert ok2 is False  # key already exists, NX fails

        result = await store.get("dup-key")
        assert result["order_id"] == "order-1"  # original preserved

    # ---- RedisClusterReservation ----

    @pytest.mark.asyncio
    async def test_reservation_fail_open_true_returns_true_when_redis_down(self):
        """With fail_open=True and Redis unavailable, reserve() must return True
        (single-pod safe degradation)."""
        reservation = RedisClusterReservation(
            redis_url="redis://localhost:59999/0",
            fail_open=True,
        )
        result = await reservation.reserve("order-123", max_allowed=5)
        assert result is True

    @pytest.mark.asyncio
    async def test_reservation_fail_open_false_returns_false_when_redis_down(self):
        """With fail_open=False and Redis unavailable, reserve() must return False
        (multi-pod safety -- block rather than risk exceeding limits)."""
        reservation = RedisClusterReservation(
            redis_url="redis://localhost:59999/0",
            fail_open=False,
        )
        result = await reservation.reserve("order-456", max_allowed=5)
        assert result is False

    @pytest.mark.asyncio
    async def test_reservation_fail_open_true_is_default(self):
        """Default behavior should be fail_open=True for backward compat."""
        reservation = RedisClusterReservation(redis_url="redis://localhost:59999/0")
        assert reservation._fail_open is True

    @pytest.mark.asyncio
    async def test_reservation_max_allowed_zero_returns_false_regardless(self):
        """When max_allowed < 1, reserve() should always return False,
        even with fail_open=True."""
        reservation = RedisClusterReservation(
            redis_url="redis://localhost:59999/0",
            fail_open=True,
        )
        result = await reservation.reserve("order-789", max_allowed=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_reservation_release_no_crash_when_redis_down(self):
        """release() should not raise when Redis is unavailable."""
        reservation = RedisClusterReservation(
            redis_url="redis://localhost:59999/0",
            fail_open=True,
        )
        # Should not raise
        await reservation.release("order-xyz")

    @pytest.mark.asyncio
    async def test_reservation_fail_open_false_with_redis_import_error(self):
        """If the redis package is not importable, fail_open=False blocks."""
        reservation = RedisClusterReservation(
            redis_url="redis://localhost:59999/0",
            fail_open=False,
        )
        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            # Force re-init of _redis to None
            reservation._redis = None
            result = await reservation.reserve("order-no-redis", max_allowed=5)
            assert result is False


# ========================================================================
# Section 4: Audit Trail Write-Through + Hash Chain
# ========================================================================


class TestAuditTrailWriteThroughAndHashChain:
    """Verify that audit events are flushed immediately (write-through)
    and that the SHA-256 hash chain provides tamper detection."""

    def _make_trail_with_mock_db(self):
        """Create an SEBIAuditTrail with a mock DB session factory
        so we can verify flush calls."""
        mock_session = MagicMock()
        mock_session.execute = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.rollback = MagicMock()
        mock_session.close = MagicMock()

        factory = MagicMock(return_value=mock_session)
        trail = SEBIAuditTrail(db_session_factory=factory, flush_threshold=100)
        return trail, factory, mock_session

    # ---- Write-through (immediate flush) ----

    def test_each_event_triggers_immediate_flush(self):
        """Every single event appended must trigger a DB flush call,
        not wait for a batch threshold."""
        trail, factory, session = self._make_trail_with_mock_db()

        trail.record_signal(
            symbol="RELIANCE.NS",
            direction="BUY",
            confidence=0.85,
            model_source="xgb_v2",
        )
        # Factory should have been called exactly once (one flush)
        assert factory.call_count == 1
        assert session.commit.call_count == 1

        trail.record_order(
            symbol="TCS.NS",
            side="SELL",
            qty=50,
            order_type="LIMIT",
            price=3500.0,
            strategy_id="momentum_v3",
            risk_checks_passed={"max_position": True, "sector_limit": True},
        )
        assert factory.call_count == 2
        assert session.commit.call_count == 2

        trail.record_fill(
            symbol="INFY.NS",
            side="BUY",
            qty=100,
            price=1450.0,
            costs_breakdown={"brokerage": 20.0, "stt": 15.0},
            slippage=0.5,
        )
        assert factory.call_count == 3
        assert session.commit.call_count == 3

    def test_flush_called_per_event_not_batched(self):
        """Verify that with flush_threshold=100, events are still flushed
        one-by-one (write-through), NOT waiting until 100 events accumulate."""
        trail, factory, session = self._make_trail_with_mock_db()

        # Append 5 events
        for i in range(5):
            trail.record_signal(
                symbol=f"SYM{i}.NS",
                direction="BUY",
                confidence=0.5 + i * 0.05,
                model_source="test_model",
            )

        # Each of the 5 events should have caused its own flush
        assert factory.call_count == 5
        assert session.commit.call_count == 5

    def test_no_flush_when_no_db_configured(self):
        """With no DB session factory, events are stored in memory only.
        No crash, no flush attempt."""
        trail = SEBIAuditTrail(db_session_factory=None)
        event = trail.record_signal(
            symbol="HDFC.NS",
            direction="SELL",
            confidence=0.72,
            model_source="lstm_v1",
        )
        assert event is not None
        assert trail.event_count == 1
        assert trail.pending_flush_count == 0

    def test_failed_flush_retains_pending_event(self):
        """If the DB flush fails, the event stays in the pending buffer
        for retry on the next append."""
        mock_session = MagicMock()
        mock_session.execute = MagicMock(side_effect=Exception("DB connection lost"))
        mock_session.rollback = MagicMock()
        mock_session.close = MagicMock()

        factory = MagicMock(return_value=mock_session)
        trail = SEBIAuditTrail(db_session_factory=factory)

        # This should not crash even though flush fails
        event = trail.record_signal(
            symbol="WIPRO.NS",
            direction="BUY",
            confidence=0.65,
            model_source="xgb_v1",
        )
        assert event is not None
        assert trail.event_count == 1
        # Pending flush buffer should still have the failed event
        assert trail.pending_flush_count == 1

    # ---- Hash chain: sequential and unique ----

    def test_hash_chain_is_sequential_each_event_different_hash(self):
        """Each event must produce a different hash in the chain."""
        trail = SEBIAuditTrail()
        hashes = []

        for i in range(10):
            trail.record_signal(
                symbol=f"SYM{i}.NS",
                direction="BUY",
                confidence=0.5,
                model_source="model_a",
            )
            hashes.append(trail._last_hash)

        # All hashes should be unique
        assert len(set(hashes)) == 10, f"Expected 10 unique hashes, got {len(set(hashes))}"

        # Hashes should be valid hex SHA-256 (64 chars)
        for h in hashes:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_hash_chain_includes_previous_hash(self):
        """Each hash must include the previous hash as input, forming a chain.
        Changing one event should invalidate all subsequent hashes."""
        import hmac as _hmac

        trail = SEBIAuditTrail()

        # Record 3 events and capture the chain
        events = []
        hashes = []

        for i in range(3):
            evt = trail.record_signal(
                symbol=f"CHAIN{i}.NS",
                direction="BUY",
                confidence=0.5 + i * 0.1,
                model_source="chain_model",
            )
            events.append(evt)
            hashes.append(trail._last_hash)

        # Manually recompute hash for event[1] using the hash from event[0]
        # to verify the chain dependency (must match HMAC-SHA256 with details)
        prev_hash_for_event_1 = hashes[0]
        evt1 = events[1]
        payload = (
            f"{prev_hash_for_event_1}|{evt1.event_id}|"
            f"{evt1.timestamp.isoformat()}|{evt1.event_type.value}|{evt1.symbol}|"
            f"{json.dumps(evt1.details, sort_keys=True, default=str)}"
        )
        expected_hash_1 = _hmac.new(trail._HMAC_KEY, payload.encode(), hashlib.sha256).hexdigest()
        assert hashes[1] == expected_hash_1, (
            "Hash for event[1] does not match manual recomputation -- chain does not include previous hash"
        )

        # Verify event[2] depends on event[1]'s hash
        prev_hash_for_event_2 = hashes[1]
        evt2 = events[2]
        payload2 = (
            f"{prev_hash_for_event_2}|{evt2.event_id}|"
            f"{evt2.timestamp.isoformat()}|{evt2.event_type.value}|{evt2.symbol}|"
            f"{json.dumps(evt2.details, sort_keys=True, default=str)}"
        )
        expected_hash_2 = _hmac.new(trail._HMAC_KEY, payload2.encode(), hashlib.sha256).hexdigest()
        assert hashes[2] == expected_hash_2

    def test_hash_chain_genesis_for_first_event(self):
        """The first event in the chain should use 'genesis' as the previous hash."""
        import hmac as _hmac

        trail = SEBIAuditTrail()
        evt = trail.record_signal(
            symbol="FIRST.NS",
            direction="BUY",
            confidence=0.9,
            model_source="genesis_test",
        )
        first_hash = trail._last_hash

        # Manually compute with "genesis" as previous (HMAC-SHA256 with details)
        payload = (
            f"genesis|{evt.event_id}|{evt.timestamp.isoformat()}|{evt.event_type.value}|{evt.symbol}|"
            f"{json.dumps(evt.details, sort_keys=True, default=str)}"
        )
        expected = _hmac.new(trail._HMAC_KEY, payload.encode(), hashlib.sha256).hexdigest()
        assert first_hash == expected

    def test_tampering_detection_changing_event_invalidates_chain(self):
        """If we replay the chain computation with a modified event in the
        middle, the resulting hash will differ from the stored chain hash.
        This proves tamper detection works."""
        import hmac as _hmac

        trail = SEBIAuditTrail()

        events = []
        hashes = []
        for i in range(3):
            evt = trail.record_signal(
                symbol=f"TAMPER{i}.NS",
                direction="BUY",
                confidence=0.5,
                model_source="tamper_test",
            )
            events.append(evt)
            hashes.append(trail._last_hash)

        # Simulate tampering: change event[0]'s symbol in a replay
        tampered_evt0 = events[0]
        # Recompute what hash[0] WOULD be with a different symbol
        tampered_payload_0 = (
            f"genesis|{tampered_evt0.event_id}|"
            f"{tampered_evt0.timestamp.isoformat()}|{tampered_evt0.event_type.value}|TAMPERED_SYMBOL|"
            f"{json.dumps(tampered_evt0.details, sort_keys=True, default=str)}"
        )
        tampered_hash_0 = _hmac.new(trail._HMAC_KEY, tampered_payload_0.encode(), hashlib.sha256).hexdigest()

        # This tampered hash should differ from the real hash[0]
        assert tampered_hash_0 != hashes[0]

        # Now recompute hash[1] using the tampered hash[0]
        evt1 = events[1]
        tampered_payload_1 = (
            f"{tampered_hash_0}|{evt1.event_id}|{evt1.timestamp.isoformat()}|{evt1.event_type.value}|{evt1.symbol}|"
            f"{json.dumps(evt1.details, sort_keys=True, default=str)}"
        )
        tampered_hash_1 = _hmac.new(trail._HMAC_KEY, tampered_payload_1.encode(), hashlib.sha256).hexdigest()

        # The cascaded hash for event[1] should now differ from the real chain
        assert tampered_hash_1 != hashes[1], "Tampering event[0] should cascade and invalidate event[1]'s hash"

    def test_hash_chain_different_event_types_produce_unique_hashes(self):
        """Different event types (signal, order, fill, risk) should all
        produce unique hashes even with similar metadata."""
        trail = SEBIAuditTrail()
        hashes = set()

        trail.record_signal(
            symbol="MULTI.NS",
            direction="BUY",
            confidence=0.8,
            model_source="m1",
        )
        hashes.add(trail._last_hash)

        trail.record_order(
            symbol="MULTI.NS",
            side="BUY",
            qty=100,
            order_type="MARKET",
            price=None,
            strategy_id="s1",
            risk_checks_passed={"check1": True},
        )
        hashes.add(trail._last_hash)

        trail.record_fill(
            symbol="MULTI.NS",
            side="BUY",
            qty=100,
            price=500.0,
            costs_breakdown={"brokerage": 10.0},
            slippage=0.1,
        )
        hashes.add(trail._last_hash)

        trail.record_risk_decision(
            check_type="max_position_size",
            result=True,
            values={"current": 5},
            thresholds={"max": 10},
        )
        hashes.add(trail._last_hash)

        assert len(hashes) == 4, "All four event types should produce unique hashes"

    def test_write_through_db_receives_chain_hash(self):
        """Verify the chain_hash is passed to the DB flush so it can be
        persisted alongside the event payload."""
        trail, factory, session = self._make_trail_with_mock_db()

        trail.record_signal(
            symbol="HASH_DB.NS",
            direction="BUY",
            confidence=0.75,
            model_source="hash_db_test",
        )

        # The session.execute should have been called with a payload
        # containing the chain_hash
        assert session.execute.call_count == 1
        call_args = session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        payload_json = params.get("payload", "{}")
        payload_data = json.loads(payload_json)
        assert "chain_hash" in payload_data
        assert payload_data["chain_hash"] is not None
        assert len(payload_data["chain_hash"]) == 64  # SHA-256 hex


# ========================================================================
# Section 5: Integration / Cross-Cutting Concerns
# ========================================================================


class TestPhase1Integration:
    """Cross-cutting tests that verify multiple safety fixes work together."""

    @pytest.mark.asyncio
    async def test_kill_switch_arm_disarm_cycle_with_audit(self):
        """Verify that arming/disarming the kill switch can be audited
        without crashes (combines kill switch fix + audit trail)."""
        ks = KillSwitch()
        trail = SEBIAuditTrail()

        await ks.arm(KillReason.BROKER_LATENCY_SPIKE, "latency > 5s")
        state = await ks.get_state()
        assert state.armed is True

        # Record the kill switch event in audit trail (no crash)
        event = trail.record_signal(
            symbol="SYSTEM",
            direction="KILL_SWITCH_ARMED",
            confidence=1.0,
            model_source="circuit_and_kill",
        )
        assert event is not None
        assert trail.event_count == 1

    @pytest.mark.asyncio
    async def test_kill_switch_all_reasons_can_arm_without_crash(self):
        """Every KillReason enum value must be armable without error."""
        for reason in KillReason:
            ks = KillSwitch()
            # Disarm first to clear any persisted state from previous iteration
            # (severity-based rejection prevents arming a lower-severity reason
            # when a higher-severity one is already persisted).
            await ks.disarm()
            await ks.arm(reason, f"test arming with {reason.value}")
            state = await ks.get_state()
            assert state.armed is True
            assert state.reason == reason

    @pytest.mark.asyncio
    async def test_idempotency_store_works_end_to_end_without_redis(self):
        """Full lifecycle: set -> get -> set_if_new_or_get -> update,
        all without Redis."""
        store = IdempotencyStore(redis_url="redis://localhost:59999/0")
        await store.is_available()
        assert store.redis_connected is False

        # set
        ok = await store.set("e2e-key", "order-100", status="PENDING")
        assert ok is True

        # get
        result = await store.get("e2e-key")
        assert result["order_id"] == "order-100"

        # set_if_new_or_get (should find existing)
        is_new, existing = await store.set_if_new_or_get("e2e-key", "order-200", None, "PENDING")
        assert is_new is False
        assert existing["order_id"] == "order-100"

        # update
        await store.update("e2e-key", "order-100", broker_order_id="BROKER-999", status="FILLED")
        result2 = await store.get("e2e-key")
        assert result2["status"] == "FILLED"
        assert result2["broker_order_id"] == "BROKER-999"

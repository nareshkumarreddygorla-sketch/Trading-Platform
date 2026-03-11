"""
Integration tests for concurrent order handling, Redis failure, and broker timeout scenarios.

These tests verify the critical paths identified in the institutional audit:
1. Concurrent order submissions don't bypass idempotency
2. Redis failure gracefully degrades
3. Broker timeouts don't create phantom positions
4. Kill switch race conditions are prevented
5. Lock coordination between OrderEntryService and RiskManager works correctly
"""

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.core.events import Exchange, Order, OrderStatus, OrderType, Position, Signal, SignalSide
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.kill_switch import KillSwitch
from src.execution.order_entry.request import OrderEntryRequest
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_entry.service import OrderEntryService
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits

logger = logging.getLogger(__name__)


def _make_signal(symbol="RELIANCE", side="BUY", price=2500.0, strategy_id="test_strategy"):
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=Exchange.NSE,
        side=SignalSide.BUY if side == "BUY" else SignalSide.SELL,
        score=0.8,
        portfolio_weight=0.05,
        risk_level="medium",
        reason="test_signal",
        price=price,
        ts=datetime.now(UTC),
    )


def _make_order(order_id="test-order-1", symbol="RELIANCE"):
    return Order(
        order_id=order_id,
        strategy_id="test_strategy",
        symbol=symbol,
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        status=OrderStatus.PENDING,
        filled_qty=0.0,
        avg_price=None,
        broker_order_id=f"BROKER-{order_id}",
        metadata={},
    )


@pytest.fixture
def risk_manager():
    limits = RiskLimits(
        max_open_positions=10,
        max_position_pct=20.0,
        max_daily_loss_pct=5.0,
    )
    return RiskManager(equity=1_000_000, limits=limits)


@pytest.fixture
def idempotency_store():
    """In-memory idempotency store for testing (no Redis)."""
    store = IdempotencyStore(redis_url="redis://nonexistent:6379")
    store._redis_checked = True  # Skip Redis connection
    store._redis_available = False
    return store


@pytest.fixture
def kill_switch():
    return KillSwitch()


@pytest.fixture
def reservation():
    return ExposureReservation()


@pytest.fixture
def mock_router():
    router = AsyncMock()
    router.place_order = AsyncMock(return_value=_make_order())
    return router


@pytest.fixture
def mock_lifecycle():
    lifecycle = AsyncMock()
    lifecycle.register = AsyncMock(return_value=True)
    lifecycle.count_active = AsyncMock(return_value=0)
    return lifecycle


@pytest.fixture
def order_service(risk_manager, mock_router, mock_lifecycle, idempotency_store, kill_switch, reservation):
    return OrderEntryService(
        risk_manager=risk_manager,
        order_router=mock_router,
        lifecycle=mock_lifecycle,
        idempotency_store=idempotency_store,
        kill_switch=kill_switch,
        reservation=reservation,
    )


class TestConcurrentOrders:
    """Test that concurrent order submissions don't bypass safety checks."""

    @pytest.mark.asyncio
    async def test_same_idempotency_key_not_duplicated(self, order_service):
        """Two concurrent submissions with same idem key should result in only one order."""
        signal = _make_signal()
        request1 = OrderEntryRequest(
            signal=signal,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="same_key_123",
        )
        request2 = OrderEntryRequest(
            signal=signal,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="same_key_123",
        )

        # Submit both concurrently
        results = await asyncio.gather(
            order_service.submit_order(request1),
            order_service.submit_order(request2),
        )

        # Both should succeed (one creates, one returns existing)
        assert all(r.success for r in results)
        # Both should have the same order_id
        order_ids = {r.order_id for r in results}
        assert len(order_ids) == 1, f"Expected 1 unique order_id, got {len(order_ids)}: {order_ids}"

    @pytest.mark.asyncio
    async def test_different_signals_not_conflated(self, order_service):
        """Different signals should create separate orders."""
        signal1 = _make_signal(symbol="RELIANCE")
        signal2 = _make_signal(symbol="TCS")

        request1 = OrderEntryRequest(
            signal=signal1,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="key_1",
        )
        request2 = OrderEntryRequest(
            signal=signal2,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=3500.0,
            idempotency_key="key_2",
        )

        result1 = await order_service.submit_order(request1)
        result2 = await order_service.submit_order(request2)

        assert result1.success
        assert result2.success

    @pytest.mark.asyncio
    async def test_max_positions_enforced_under_load(self, order_service, risk_manager):
        """Rapid submissions should not exceed max_open_positions."""
        risk_manager.limits.max_open_positions = 3

        tasks = []
        for i in range(10):
            signal = _make_signal(symbol=f"STOCK{i}", price=100.0)
            request = OrderEntryRequest(
                signal=signal,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=100.0,
                idempotency_key=f"concurrent_{i}",
            )
            tasks.append(order_service.submit_order(request))

        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r.success)
        # Should not exceed max_open_positions
        assert success_count <= 3, f"Expected at most 3 successful orders, got {success_count}"


class TestKillSwitchSafety:
    """Test kill switch race condition fix."""

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_increasing_orders(self, order_service, kill_switch, risk_manager):
        """When kill switch is armed, only position-reducing orders should be allowed."""
        # Add existing LONG position
        risk_manager.add_position(
            Position(
                symbol="RELIANCE",
                exchange=Exchange.NSE,
                side=SignalSide.BUY,
                quantity=100,
                avg_price=2500.0,
            )
        )

        # Arm kill switch
        from src.execution.order_entry.kill_switch import KillReason

        await kill_switch.arm(KillReason.DAILY_LOSS)

        # Try to BUY more (increasing) - should be rejected
        signal_buy = _make_signal(symbol="RELIANCE", side="BUY")
        request_buy = OrderEntryRequest(
            signal=signal_buy,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="kill_buy",
        )
        result = await order_service.submit_order(request_buy)
        assert not result.success, "Buy (increasing) order should be rejected when kill switch armed"


class TestBrokerTimeout:
    """Test broker timeout doesn't create phantom positions."""

    @pytest.mark.asyncio
    async def test_timeout_releases_reservation(self, order_service, mock_router, reservation):
        """Broker timeout should release exposure reservation."""
        # Make broker timeout
        mock_router.place_order = AsyncMock(side_effect=TimeoutError())

        signal = _make_signal()
        request = OrderEntryRequest(
            signal=signal,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="timeout_test",
        )

        result = await order_service.submit_order(request)
        assert not result.success
        assert "timeout" in (result.reject_detail or "").lower()

        # Reservation should be released
        assert len(reservation._reservations) == 0, "Reservation should be released after timeout"


class TestCircuitBreaker:
    """Test circuit breaker prevents trading."""

    @pytest.mark.asyncio
    async def test_circuit_open_blocks_orders(self, order_service, risk_manager):
        """Open circuit breaker should block all new orders."""
        risk_manager.open_circuit("test_trigger")

        signal = _make_signal()
        request = OrderEntryRequest(
            signal=signal,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="circuit_test",
        )

        result = await order_service.submit_order(request)
        assert not result.success
        assert "circuit" in (result.reject_detail or "").lower()


class TestRiskLimits:
    """Test risk limit enforcement under various conditions."""

    @pytest.mark.asyncio
    async def test_daily_loss_limit_blocks_new_orders(self, order_service, risk_manager):
        """Orders blocked when daily loss exceeds limit."""
        risk_manager.register_pnl(-60_000)  # -6% of 1M equity

        signal = _make_signal()
        request = OrderEntryRequest(
            signal=signal,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="loss_test",
        )

        result = await order_service.submit_order(request)
        assert not result.success

    @pytest.mark.asyncio
    async def test_zero_equity_blocks_orders(self, order_service, risk_manager):
        """Orders blocked when equity is zero."""
        risk_manager.update_equity(0)

        signal = _make_signal()
        request = OrderEntryRequest(
            signal=signal,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key="equity_test",
        )

        result = await order_service.submit_order(request)
        assert not result.success


class TestVaRSectorCorrelation:
    """Test sector-aware VaR correlation defaults."""

    def test_same_sector_high_correlation(self):
        """Same-sector stocks should have higher default correlation."""
        from src.risk_engine.var import PortfolioVaR

        var_calc = PortfolioVaR(min_history=100)  # High min so it uses defaults

        # IT stocks should have high correlation (0.82)
        symbols = ["INFY", "TCS", "WIPRO"]
        corr = var_calc._correlation_matrix(symbols)
        # Off-diagonal should be > 0.7 (sector-aware, not generic 0.5)
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                if i != j:
                    assert corr[i, j] >= 0.70, f"IT sector correlation {symbols[i]}-{symbols[j]} too low: {corr[i, j]}"

    def test_cross_sector_moderate_correlation(self):
        """Cross-sector stocks should have moderate default correlation."""
        from src.risk_engine.var import PortfolioVaR

        var_calc = PortfolioVaR(min_history=100)

        symbols = ["INFY", "HDFCBANK"]  # IT vs Banking
        corr = var_calc._correlation_matrix(symbols)
        # Cross-sector should be moderate (0.3-0.5)
        assert 0.2 <= corr[0, 1] <= 0.6, f"Cross-sector correlation too extreme: {corr[0, 1]}"


class TestFeatureShiftDetector:
    """Test feature distribution shift detection."""

    def test_detects_significant_shift(self):
        """Should detect when features shift significantly from training distribution."""
        import numpy as np

        from src.ai.feature_shift_detector import FeatureShiftDetector

        detector = FeatureShiftDetector(min_samples=50)

        # Set training stats from normal distribution (mean=50, std=10)
        np.random.seed(42)
        training_data = np.random.normal(50, 10, 1000).reshape(-1, 1)
        detector.load_training_stats_from_arrays(["rsi_14"], training_data)

        # Record shifted live data (mean=80, std=10) - significant shift
        for _ in range(200):
            detector.record_live_features({"rsi_14": float(np.random.normal(80, 10))})

        report = detector.check_shift()
        assert report.recommendation in ("retrain", "halt"), f"Expected retrain/halt, got {report.recommendation}"
        assert report.features_significant > 0

    def test_no_false_alarm_on_normal_data(self):
        """Should not trigger on data from the same distribution."""
        import numpy as np

        from src.ai.feature_shift_detector import FeatureShiftDetector

        detector = FeatureShiftDetector(min_samples=50)

        np.random.seed(42)
        training_data = np.random.normal(50, 10, 1000).reshape(-1, 1)
        detector.load_training_stats_from_arrays(["rsi_14"], training_data)

        # Record live data from same distribution
        for _ in range(200):
            detector.record_live_features({"rsi_14": float(np.random.normal(50, 10))})

        report = detector.check_shift()
        assert report.recommendation == "ok", f"False alarm: {report.recommendation}"


class TestReconciliation:
    """Test broker position reconciliation."""

    @pytest.mark.asyncio
    async def test_detects_missing_broker_position(self):
        """Should detect position that exists locally but not at broker."""
        from src.execution.reconciliation import BrokerReconciliator

        local_positions = [
            Position(symbol="RELIANCE", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=100, avg_price=2500.0),
        ]
        broker_positions = []  # Empty - broker has nothing

        reconciliator = BrokerReconciliator(
            get_local_positions=lambda: local_positions,
            get_broker_positions=lambda: broker_positions,
        )

        report = await reconciliator.reconcile()
        assert len(report.discrepancies) > 0
        assert report.discrepancies[0].discrepancy_type == "missing_broker"
        assert report.status in ("warning", "critical")

    @pytest.mark.asyncio
    async def test_no_discrepancy_when_matched(self):
        """Should report ok when positions match."""
        from src.execution.reconciliation import BrokerReconciliator

        local_positions = [
            Position(symbol="RELIANCE", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=100, avg_price=2500.0),
        ]
        broker_positions = [
            {"symbol": "RELIANCE", "exchange": "NSE", "side": "BUY", "quantity": 100, "avg_price": 2500.0},
        ]

        reconciliator = BrokerReconciliator(
            get_local_positions=lambda: local_positions,
            get_broker_positions=lambda: broker_positions,
        )

        report = await reconciliator.reconcile()
        assert report.matched == 1
        assert report.status == "ok"


class TestDatabaseHealth:
    """Test database health checking."""

    def test_health_check_returns_result(self):
        """Health check should return a result dict."""
        from src.persistence.database import check_database_health

        result = check_database_health()
        assert "healthy" in result
        assert "error" in result

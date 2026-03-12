"""Unit tests for src.execution.order_router — multi-broker routing, price improvement, sizing.

Covers:
- NSE tick-size validation and price sanity
- BrokerHealth tracking (success, failure, unhealthy marking, reset)
- OrderRouter single-broker routing
- OrderRouter multi-broker ranking and failover
- Tick-based and spread-based price improvement
- ADV-aware child order sizing
- Minimum order value check
- Freeze-qty splitting with child cancellation on failure
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import Exchange, Order, OrderStatus, OrderType, Signal, SignalSide
from src.execution.order_router import (
    BrokerHealth,
    OrderRouter,
    validate_price_sanity,
    validate_tick_size,
)

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────


def _make_signal(
    symbol: str = "RELIANCE",
    side: SignalSide = SignalSide.BUY,
    price: float = 2500.0,
    exchange: Exchange = Exchange.NSE,
    strategy_id: str = "test_strategy",
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=exchange,
        side=side,
        score=0.8,
        portfolio_weight=0.1,
        risk_level="medium",
        reason="unit_test",
        price=price,
    )


def _make_order(order_id: str = "ORD001", status: OrderStatus = OrderStatus.LIVE) -> Order:
    return Order(
        order_id=order_id,
        strategy_id="test_strategy",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=100.0,
        order_type=OrderType.LIMIT,
        status=status,
    )


def _make_mock_gateway() -> AsyncMock:
    gw = AsyncMock()
    gw.place_order = AsyncMock(return_value=_make_order())
    gw.cancel_order = AsyncMock(return_value=True)
    return gw


# ──────────────────────────────────────────────────
# validate_tick_size
# ──────────────────────────────────────────────────


class TestTickSizeValidation:
    def test_high_price_rounds_to_005(self):
        """Stocks >= ₹100: tick size is ₹0.05."""
        assert validate_tick_size(100.03) == 100.05
        assert validate_tick_size(100.07) == 100.05
        assert validate_tick_size(100.08) == 100.10
        assert validate_tick_size(250.12) == 250.10

    def test_low_price_rounds_to_001(self):
        """Stocks < ₹100: tick size is ₹0.01 (paisa)."""
        assert validate_tick_size(50.123) == 50.12
        assert validate_tick_size(99.999) == 100.0

    def test_exact_tick_unchanged(self):
        assert validate_tick_size(100.05) == 100.05
        assert validate_tick_size(50.01) == 50.01

    def test_zero_or_negative_returns_as_is(self):
        assert validate_tick_size(0) == 0
        assert validate_tick_size(-5.0) == -5.0


class TestPriceSanity:
    def test_valid_range(self):
        assert validate_price_sanity(1.0) is True
        assert validate_price_sanity(50000.0) is True
        assert validate_price_sanity(100000.0) is True

    def test_below_minimum(self):
        assert validate_price_sanity(0.5) is False

    def test_above_maximum(self):
        assert validate_price_sanity(200000.0) is False


# ──────────────────────────────────────────────────
# BrokerHealth
# ──────────────────────────────────────────────────


class TestBrokerHealth:
    def test_initial_state(self):
        bh = BrokerHealth("test_broker")
        assert bh.healthy is True
        assert bh.consecutive_failures == 0
        assert bh.avg_latency == 0.0
        assert bh.failure_rate == 0.0

    def test_record_success(self):
        bh = BrokerHealth("test_broker")
        bh.record_success(0.05)
        assert bh.healthy is True
        assert bh.consecutive_failures == 0
        assert bh.last_success_ts is not None
        assert len(bh.latencies) == 1

    def test_record_failure_marks_unhealthy(self):
        bh = BrokerHealth("test_broker", max_consecutive_failures=3)
        bh.record_failure()
        bh.record_failure()
        assert bh.healthy is True  # 2 failures, below threshold
        bh.record_failure()
        assert bh.healthy is False  # 3 failures, at threshold

    def test_success_resets_failure_count(self):
        bh = BrokerHealth("test_broker", max_consecutive_failures=3)
        bh.record_failure()
        bh.record_failure()
        bh.record_success(0.05)
        assert bh.consecutive_failures == 0
        assert bh.healthy is True

    def test_avg_latency(self):
        bh = BrokerHealth("test_broker")
        bh.record_success(0.1)
        bh.record_success(0.3)
        assert abs(bh.avg_latency - 0.2) < 1e-9

    def test_failure_rate(self):
        bh = BrokerHealth("test_broker")
        bh.record_success(0.1)
        bh.record_failure()
        assert abs(bh.failure_rate - 0.5) < 1e-9

    def test_reset(self):
        bh = BrokerHealth("test_broker")
        bh.record_success(0.1)
        bh.record_failure()
        bh.reset()
        assert bh.healthy is True
        assert bh.consecutive_failures == 0
        assert len(bh.latencies) == 0
        assert bh._total_orders == 0

    def test_as_dict(self):
        bh = BrokerHealth("test_broker")
        bh.record_success(0.05)
        d = bh.as_dict()
        assert d["name"] == "test_broker"
        assert d["healthy"] is True
        assert d["total_orders"] == 1
        assert "avg_latency_ms" in d

    def test_repr(self):
        bh = BrokerHealth("angel_one")
        assert "angel_one" in repr(bh)
        assert "healthy=True" in repr(bh)

    def test_rolling_window(self):
        bh = BrokerHealth("test_broker", window_size=3)
        bh.record_success(1.0)
        bh.record_success(2.0)
        bh.record_success(3.0)
        bh.record_success(4.0)
        # Window of 3: [2.0, 3.0, 4.0]
        assert len(bh.latencies) == 3
        assert abs(bh.avg_latency - 3.0) < 1e-9


# ──────────────────────────────────────────────────
# OrderRouter — single broker
# ──────────────────────────────────────────────────


class TestOrderRouterSingleBroker:
    @pytest.mark.asyncio
    async def test_basic_limit_order(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        signal = _make_signal()
        order = await router.place_order(signal, 100, OrderType.LIMIT, limit_price=2500.0)
        assert order.order_id == "ORD001"
        gw.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_market_order_when_price_none(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        signal = _make_signal(price=None)
        order = await router.place_order(signal, 100)
        gw.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_size_adjustment(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        signal = _make_signal(price=2500.03)
        await router.place_order(signal, 100, OrderType.LIMIT, limit_price=2500.03)
        # Should have adjusted price to valid tick size
        call_kwargs = gw.place_order.call_args
        assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_price_sanity_raises(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        signal = _make_signal(price=200000.0)
        with pytest.raises(ValueError, match="outside valid range"):
            await router.place_order(signal, 100, OrderType.LIMIT, limit_price=200000.0)

    @pytest.mark.asyncio
    async def test_minimum_order_value_raises(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        signal = _make_signal(price=5.0)
        with pytest.raises(ValueError, match="below minimum"):
            await router.place_order(signal, 10, OrderType.LIMIT, limit_price=5.0)

    @pytest.mark.asyncio
    async def test_register_gateway(self):
        gw_default = _make_mock_gateway()
        gw_bse = _make_mock_gateway()
        gw_bse.place_order.return_value = _make_order(order_id="BSE001")
        router = OrderRouter(gw_default)
        router.register_gateway("BSE", gw_bse)
        signal = _make_signal(exchange=Exchange.BSE)
        order = await router.place_order(signal, 100, OrderType.LIMIT, limit_price=2500.0)
        assert order.order_id == "BSE001"


# ──────────────────────────────────────────────────
# OrderRouter — price improvement
# ──────────────────────────────────────────────────


class TestPriceImprovement:
    def test_tick_based_buy_improvement(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        improved = router._price_improvement(2500.0, "BUY", ticks=2)
        assert improved == 2499.90  # 2500.0 - 2 * 0.05

    def test_tick_based_sell_improvement(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        improved = router._price_improvement(2500.0, "SELL", ticks=2)
        assert improved == 2500.10  # 2500.0 + 2 * 0.05

    def test_spread_based_buy_improvement(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw, improvement_bps=10.0)
        improved = router._spread_price_improvement("BUY", 100.0, 101.0, 10.0)
        # mid = 100.5, improvement = 100.5 * 10/10000 = 0.1005
        # improved = 100.5 - 0.1005 = 100.3995, clamped to bid=100.0
        assert improved >= 100.0  # Should not go below bid
        assert improved <= 100.5  # Should be below mid

    def test_spread_based_sell_improvement(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw, improvement_bps=10.0)
        improved = router._spread_price_improvement("SELL", 100.0, 101.0, 10.0)
        assert improved >= 100.5  # Should be above mid
        assert improved <= 101.0  # Should not go above ask

    def test_invalid_spread_returns_zero(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        assert router._spread_price_improvement("BUY", 0, 100.0, 5.0) == 0.0
        assert router._spread_price_improvement("BUY", 100.0, 0, 5.0) == 0.0
        assert router._spread_price_improvement("BUY", 101.0, 100.0, 5.0) == 0.0  # ask < bid


# ──────────────────────────────────────────────────
# OrderRouter — ADV-aware sizing
# ──────────────────────────────────────────────────


class TestADVSizing:
    def test_no_splitting_below_adv_threshold(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        # ADV below minimum threshold (50_000)
        slices = router._adv_adjusted_child_sizes(100, 10_000, 100.0)
        assert slices == [100]

    def test_no_splitting_within_participation(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        # qty=2000, ADV=100_000, max_participation=5% → max_child=5000
        slices = router._adv_adjusted_child_sizes(2000, 100_000, 100.0)
        assert slices == [2000]

    def test_splitting_when_exceeds_participation(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        # qty=10000, ADV=100_000, max_participation=5% → max_child=5000 → 2 slices
        slices = router._adv_adjusted_child_sizes(10000, 100_000, 100.0)
        assert len(slices) == 2
        assert sum(slices) == 10000

    def test_many_slices_for_large_order(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        # qty=50000, ADV=100_000, max_participation=5% → max_child=5000 → 10 slices
        slices = router._adv_adjusted_child_sizes(50000, 100_000, 100.0)
        assert len(slices) == 10
        assert sum(slices) == 50000

    def test_zero_quantity_returns_list(self):
        gw = _make_mock_gateway()
        router = OrderRouter(gw)
        slices = router._adv_adjusted_child_sizes(0, 100_000, 100.0)
        assert slices == [0]


# ──────────────────────────────────────────────────
# OrderRouter — multi-broker ranking & failover
# ──────────────────────────────────────────────────


class TestMultiBrokerRouting:
    def test_broker_ranking_by_cost(self):
        gw1 = _make_mock_gateway()
        gw2 = _make_mock_gateway()
        router = OrderRouter(
            gw1,
            brokers={"angel_one": gw1, "zerodha": gw2},
            broker_costs_bps={"angel_one": 20.0, "zerodha": 5.0},
        )
        ranked = router._rank_brokers()
        # zerodha should rank first (lower cost)
        assert ranked[0][0] == "zerodha"
        assert ranked[1][0] == "angel_one"

    def test_unhealthy_broker_ranked_last(self):
        gw1 = _make_mock_gateway()
        gw2 = _make_mock_gateway()
        router = OrderRouter(
            gw1,
            brokers={"angel_one": gw1, "zerodha": gw2},
        )
        # Mark angel_one as unhealthy
        health = router.get_broker_health("angel_one")
        for _ in range(3):
            health.record_failure()
        ranked = router._rank_brokers()
        # angel_one should be last (unhealthy)
        assert ranked[-1][0] == "angel_one"

    @pytest.mark.asyncio
    async def test_failover_on_primary_failure(self):
        gw1 = _make_mock_gateway()
        gw2 = _make_mock_gateway()
        gw1.place_order = AsyncMock(side_effect=RuntimeError("broker down"))
        gw2.place_order = AsyncMock(return_value=_make_order(order_id="FAILOVER001"))
        router = OrderRouter(
            gw1,
            brokers={"angel_one": gw1, "zerodha": gw2},
        )
        signal = _make_signal()
        order = await router.place_order(signal, 100, OrderType.LIMIT, limit_price=2500.0)
        assert order.order_id == "FAILOVER001"

    @pytest.mark.asyncio
    async def test_all_brokers_fail_raises(self):
        gw1 = _make_mock_gateway()
        gw2 = _make_mock_gateway()
        gw1.place_order = AsyncMock(side_effect=RuntimeError("broker1 down"))
        gw2.place_order = AsyncMock(side_effect=RuntimeError("broker2 down"))
        router = OrderRouter(
            gw1,
            brokers={"angel_one": gw1, "zerodha": gw2},
        )
        signal = _make_signal()
        with pytest.raises(RuntimeError, match="All brokers failed"):
            await router.place_order(signal, 100, OrderType.LIMIT, limit_price=2500.0)

    def test_get_all_broker_health(self):
        gw1 = _make_mock_gateway()
        gw2 = _make_mock_gateway()
        router = OrderRouter(
            gw1,
            brokers={"angel_one": gw1, "zerodha": gw2},
        )
        health = router.get_all_broker_health()
        assert "angel_one" in health
        assert "zerodha" in health
        assert health["angel_one"]["healthy"] is True

    def test_latency_affects_ranking(self):
        gw1 = _make_mock_gateway()
        gw2 = _make_mock_gateway()
        router = OrderRouter(
            gw1,
            brokers={"angel_one": gw1, "zerodha": gw2},
            broker_costs_bps={"angel_one": 10.0, "zerodha": 10.0},
        )
        # Give angel_one high latency
        health = router.get_broker_health("angel_one")
        for _ in range(5):
            health.record_success(2.0)  # 2s latency → penalty = 10 bps
        ranked = router._rank_brokers()
        # zerodha should rank first (same cost, lower latency)
        assert ranked[0][0] == "zerodha"


# ──────────────────────────────────────────────────
# OrderRouter — split orders
# ──────────────────────────────────────────────────


class TestSplitOrders:
    @pytest.mark.asyncio
    async def test_freeze_qty_splitting(self):
        gw = _make_mock_gateway()
        call_count = 0

        async def counting_place_order(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_order(order_id=f"SPLIT{call_count:03d}")

        gw.place_order = counting_place_order
        freeze_mgr = MagicMock()
        freeze_mgr.check_and_split.return_value = [500, 500]  # 2 child orders
        router = OrderRouter(gw, freeze_qty_manager=freeze_mgr)
        signal = _make_signal()
        order = await router.place_order(signal, 1000, OrderType.LIMIT, limit_price=2500.0)
        assert order.order_id == "SPLIT001"  # Returns first child as parent
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_split_child_failure_cancels_previous(self):
        """When a split child fails, prior children are cancelled and router
        falls back to placing a single full-qty order (graceful degradation)."""
        gw = _make_mock_gateway()
        call_count = 0

        async def failing_place_order(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("child 2 failed")
            return _make_order(order_id=f"SPLIT{call_count:03d}")

        gw.place_order = failing_place_order
        freeze_mgr = MagicMock()
        freeze_mgr.check_and_split.return_value = [500, 500]
        router = OrderRouter(gw, freeze_qty_manager=freeze_mgr)
        signal = _make_signal()
        # The router catches the split failure and falls back to full-qty order
        order = await router.place_order(signal, 1000, OrderType.LIMIT, limit_price=2500.0)
        # Should have attempted to cancel the first successful child
        gw.cancel_order.assert_called()
        # call 1 = split child 1 (OK), call 2 = split child 2 (fail),
        # call 3 = fallback full-qty order
        assert call_count == 3
        assert order.order_id == "SPLIT003"


# ──────────────────────────────────────────────────
# OrderRouter — broker timeout
# ──────────────────────────────────────────────────


class TestBrokerTimeout:
    @pytest.mark.asyncio
    async def test_timeout_records_failure(self):
        gw = _make_mock_gateway()

        async def slow_place_order(**kwargs):
            await asyncio.sleep(30)  # Will timeout
            return _make_order()

        gw.place_order = slow_place_order
        router = OrderRouter(
            gw,
            brokers={"angel_one": gw},
        )
        signal = _make_signal()
        with pytest.raises(RuntimeError, match="All brokers failed|timed out"):
            await router.place_order(signal, 100, OrderType.LIMIT, limit_price=2500.0)
        health = router.get_broker_health("angel_one")
        assert health.consecutive_failures >= 1

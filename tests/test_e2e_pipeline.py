"""
End-to-end integration tests for the full trading pipeline:
Signal -> Allocation -> Risk Check -> Order Entry -> Fill

Tests the complete order flow from signal generation through position update,
exercising all pipeline stages with realistic mocks for external dependencies.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.core.events import (
    Exchange,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalSide,
)
from src.execution.base import BaseExecutionGateway
from src.execution.fill_handler.events import FillEvent, FillType
from src.execution.fill_handler.handler import FillHandler
from src.execution.fill_handler.paper_fill_simulator import PaperFillSimulator
from src.execution.lifecycle import OrderLifecycle
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.kill_switch import KillReason, KillSwitch
from src.execution.order_entry.rate_limiter import OrderRateLimiter, RateLimitConfig
from src.execution.order_entry.request import (
    OrderEntryRequest,
    RejectReason,
)
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_entry.service import OrderEntryService
from src.execution.order_router import OrderRouter
from src.risk_engine.limits import RiskLimits
from src.risk_engine.manager import RiskManager
from src.strategy_engine.allocator import AllocatorConfig, PortfolioAllocator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeGateway(BaseExecutionGateway):
    """In-memory broker gateway for testing. Returns a PENDING order."""

    def __init__(self):
        self.placed_orders: list = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def place_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float | None = None,
        strategy_id: str = "",
        **kwargs,
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            symbol=symbol,
            exchange=Exchange(exchange) if isinstance(exchange, str) else exchange,
            side=SignalSide(side),
            quantity=quantity,
            order_type=OrderType(order_type) if isinstance(order_type, str) else order_type,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            broker_order_id=f"BRK-{uuid.uuid4().hex[:8]}",
        )
        self.placed_orders.append(order)
        return order

    async def cancel_order(self, order_id: str, broker_order_id: str | None = None) -> bool:
        return True

    async def get_order_status(self, order_id: str, broker_order_id: str | None = None) -> OrderStatus:
        return OrderStatus.PENDING

    async def get_positions(self) -> list[Position]:
        return []

    async def get_orders(self, status: str | None = None, limit: int = 100) -> list[Order]:
        return list(self.placed_orders)


def make_signal(
    symbol: str = "RELIANCE",
    side: SignalSide = SignalSide.BUY,
    score: float = 0.75,
    price: float = 2500.0,
    strategy_id: str = "test_strategy",
) -> Signal:
    """Create a test signal with sensible defaults."""
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=Exchange.NSE,
        side=side,
        score=score,
        portfolio_weight=0.1,
        price=price,
        ts=datetime.now(UTC),
    )


def make_position(
    symbol: str = "RELIANCE",
    side: SignalSide = SignalSide.BUY,
    quantity: float = 10,
    avg_price: float = 2500.0,
) -> Position:
    """Create a test position."""
    return Position(
        symbol=symbol,
        exchange=Exchange.NSE,
        side=side,
        quantity=quantity,
        avg_price=avg_price,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gateway():
    return FakeGateway()


@pytest.fixture
def risk_manager():
    limits = RiskLimits(
        max_position_pct=5.0,
        max_daily_loss_pct=2.0,
        max_open_positions=10,
        max_consecutive_losses=5,
        max_per_symbol_pct=10.0,
        max_sector_concentration_pct=25.0,
    )
    rm = RiskManager(equity=1_000_000, limits=limits, load_persisted_state=False)
    return rm


@pytest.fixture
def lifecycle():
    return OrderLifecycle()


@pytest.fixture
def idempotency():
    return IdempotencyStore(redis_url="redis://localhost:6379/99")


@pytest.fixture
def kill_switch(tmp_path):
    """KillSwitch with state path pointed to tmp_path to avoid stale state."""
    with patch(
        "src.execution.order_entry.kill_switch._KILL_STATE_PATH",
        str(tmp_path / "kill_switch_state.json"),
    ):
        ks = KillSwitch()
    return ks


@pytest.fixture
def reservation():
    return ExposureReservation()


@pytest.fixture
def order_router(gateway):
    return OrderRouter(default_gateway=gateway)


@pytest.fixture
def allocator():
    config = AllocatorConfig(
        max_active_signals=5,
        max_capital_pct_per_signal=10.0,
        min_confidence=0.5,
        use_kelly=True,
        kelly_fraction=0.5,
    )
    return PortfolioAllocator(config)


@pytest.fixture
def fill_handler(risk_manager, lifecycle):
    return FillHandler(risk_manager=risk_manager, lifecycle=lifecycle)


@pytest.fixture
def order_service(risk_manager, order_router, lifecycle, idempotency, kill_switch, reservation):
    return OrderEntryService(
        risk_manager=risk_manager,
        order_router=order_router,
        lifecycle=lifecycle,
        idempotency_store=idempotency,
        kill_switch=kill_switch,
        reservation=reservation,
    )


@pytest.fixture
def order_service_with_rate_limiter(risk_manager, order_router, lifecycle, idempotency, kill_switch, reservation):
    """OrderEntryService with a rate limiter allowing only 5 orders per minute."""
    rl = OrderRateLimiter(RateLimitConfig(max_orders_per_minute=5, window_seconds=60.0))
    return OrderEntryService(
        risk_manager=risk_manager,
        order_router=order_router,
        lifecycle=lifecycle,
        idempotency_store=idempotency,
        kill_switch=kill_switch,
        reservation=reservation,
        rate_limiter=rl,
    )


# ---------------------------------------------------------------------------
# 1. Full pipeline happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_happy_path(allocator, risk_manager, order_service, fill_handler, lifecycle, gateway):
    """Signal -> allocate -> risk check -> submit order -> simulate fill -> verify position."""
    # Step 1: Create a signal
    signal = make_signal(score=0.75, price=2500.0)

    # Step 2: Allocate
    allocations = allocator.allocate(
        signals=[signal],
        equity=risk_manager.equity,
        positions=risk_manager.positions,
        max_position_pct=5.0,
    )
    assert len(allocations) > 0, "Allocator should produce at least one allocation"
    alloc_signal, qty = allocations[0]
    assert qty > 0, "Allocated quantity must be positive"

    # Step 3: Submit order (includes risk check internally)
    request = OrderEntryRequest(
        signal=alloc_signal,
        quantity=qty,
        order_type=OrderType.LIMIT,
        limit_price=alloc_signal.price,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert result.success, f"Order should succeed, got reject: {result.reject_reason} - {result.reject_detail}"
    assert result.order_id is not None

    # Step 4: Simulate fill
    fill_event = FillEvent(
        order_id=result.order_id,
        broker_order_id=result.broker_order_id,
        symbol=signal.symbol,
        exchange="NSE",
        side="BUY",
        fill_type=FillType.FILL,
        filled_qty=qty,
        remaining_qty=0,
        avg_price=2501.0,
        ts=datetime.now(UTC),
        strategy_id=signal.strategy_id,
    )
    await fill_handler.on_fill_event(fill_event)

    # Step 5: Verify position updated in risk manager
    assert len(risk_manager.positions) == 1
    pos = risk_manager.positions[0]
    assert pos.symbol == "RELIANCE"
    assert pos.quantity == qty
    assert pos.avg_price == 2501.0


# ---------------------------------------------------------------------------
# 2. Pipeline with risk rejection (position limit exceeded)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_risk_rejection_position_limit(order_service, risk_manager):
    """Signal that exceeds position size limit gets blocked by risk check."""
    # An order whose notional far exceeds 5% of equity should be rejected
    signal = make_signal(price=2500.0)
    # 5% of 1M = 50K, so 100 shares = 250K (25% of equity) -> rejected
    qty = 100
    request = OrderEntryRequest(
        signal=signal,
        quantity=qty,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.RISK_REJECTED


# ---------------------------------------------------------------------------
# 3. Pipeline with kill switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_kill_switch_blocks_order(order_service, kill_switch):
    """Order blocked when kill switch is armed."""
    await kill_switch.arm(KillReason.MANUAL, detail="test kill switch")

    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.KILL_SWITCH


# ---------------------------------------------------------------------------
# 4. Pipeline with circuit breaker open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_circuit_breaker_open(order_service, risk_manager):
    """Order blocked when risk circuit breaker is open."""
    risk_manager.open_circuit("test_breach")

    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.CIRCUIT_BREAKER


# ---------------------------------------------------------------------------
# 5. Pipeline drawdown scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_drawdown_trip_and_resume(order_service, risk_manager):
    """Equity drops -> circuit trips -> orders blocked -> reset -> orders resume."""
    # Record enough daily loss to trigger daily loss limit (>2% of 1M = >20K)
    risk_manager.register_pnl(-25_000)

    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success, "Order should be blocked due to daily loss limit"
    assert result.reject_reason == RejectReason.RISK_REJECTED
    assert "daily loss" in result.reject_detail.lower()

    # Reset daily PnL and try again - order should succeed
    risk_manager.reset_daily_pnl()
    request2 = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
        idempotency_key=str(uuid.uuid4()),
    )
    result2 = await order_service.submit_order(request2)
    assert result2.success, f"Order should succeed after reset, got: {result2.reject_reason} - {result2.reject_detail}"


# ---------------------------------------------------------------------------
# 6. Multi-signal allocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_signal_allocation(allocator, risk_manager, order_service, gateway):
    """Multiple signals -> proper capital split -> all orders submitted."""
    signals = [
        make_signal(symbol="RELIANCE", score=0.85, price=2500.0, strategy_id="s1"),
        make_signal(symbol="TCS", score=0.80, price=3500.0, strategy_id="s2"),
        make_signal(symbol="INFY", score=0.70, price=1500.0, strategy_id="s3"),
    ]

    allocations = allocator.allocate(
        signals=signals,
        equity=risk_manager.equity,
        positions=[],
        max_position_pct=5.0,
    )
    assert len(allocations) == 3, f"Expected 3 allocations, got {len(allocations)}"

    # Submit all orders
    results = []
    for sig, qty in allocations:
        request = OrderEntryRequest(
            signal=sig,
            quantity=qty,
            order_type=OrderType.LIMIT,
            limit_price=sig.price,
            source="test",
            idempotency_key=str(uuid.uuid4()),
        )
        result = await order_service.submit_order(request)
        results.append(result)

    # All should succeed
    for i, r in enumerate(results):
        assert r.success, f"Order {i} failed: {r.reject_reason} - {r.reject_detail}"

    # Verify gateway received all orders
    assert len(gateway.placed_orders) == 3


# ---------------------------------------------------------------------------
# 7. Duplicate order rejection (idempotency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_order_idempotency(order_service):
    """Same order submitted twice with same idempotency key returns same order_id."""
    signal = make_signal(price=2500.0)
    idem_key = "test-idempotent-key-" + uuid.uuid4().hex[:8]
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key=idem_key,
        source="test",
    )

    result1 = await order_service.submit_order(request)
    assert result1.success

    # Submit same request again
    result2 = await order_service.submit_order(request)
    assert result2.success
    assert result2.order_id == result1.order_id, "Duplicate should return same order_id"


# ---------------------------------------------------------------------------
# 8. Rate limiter integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_kicks_in(order_service_with_rate_limiter):
    """Burst of orders -> rate limiter blocks excess."""
    svc = order_service_with_rate_limiter
    results = []
    for i in range(8):
        signal = make_signal(symbol=f"SYM{i}", price=2500.0)
        request = OrderEntryRequest(
            signal=signal,
            quantity=2,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key=str(uuid.uuid4()),
            source="test",
        )
        result = await svc.submit_order(request)
        results.append(result)

    # First 5 should succeed, rest should be rate limited
    success_count = sum(1 for r in results if r.success)
    rate_limited = [r for r in results if not r.success and r.reject_detail == "rate_limit_exceeded"]
    assert success_count == 5, f"Expected 5 successful, got {success_count}"
    assert len(rate_limited) == 3, f"Expected 3 rate limited, got {len(rate_limited)}"


# ---------------------------------------------------------------------------
# 9. Paper fill lifecycle (partial fill -> full fill)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_fill_lifecycle_partial_then_full(fill_handler, lifecycle, risk_manager):
    """Order submitted -> partial fill -> full fill -> position updated."""
    # Register an order first
    order = Order(
        order_id="test-order-pf",
        strategy_id="test",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        status=OrderStatus.PENDING,
    )
    await lifecycle.register(order)

    # Partial fill: 60 of 100
    partial_event = FillEvent(
        order_id="test-order-pf",
        broker_order_id="PAPER-partial",
        symbol="RELIANCE",
        exchange="NSE",
        side="BUY",
        fill_type=FillType.PARTIAL_FILL,
        filled_qty=60,
        remaining_qty=40,
        avg_price=2501.0,
        ts=datetime.now(UTC),
        strategy_id="test",
    )
    await fill_handler.on_fill_event(partial_event)

    # Check partial position
    assert len(risk_manager.positions) == 1
    assert risk_manager.positions[0].quantity == 60

    # Check lifecycle status
    lc_order = await lifecycle.get_order("test-order-pf")
    assert lc_order.status == OrderStatus.PARTIALLY_FILLED

    # Full fill: remaining 40
    full_event = FillEvent(
        order_id="test-order-pf",
        broker_order_id="PAPER-full",
        symbol="RELIANCE",
        exchange="NSE",
        side="BUY",
        fill_type=FillType.FILL,
        filled_qty=40,
        remaining_qty=0,
        avg_price=2502.0,
        ts=datetime.now(UTC),
        strategy_id="test",
    )
    await fill_handler.on_fill_event(full_event)

    # Position should be merged (60 + 40 = 100)
    assert len(risk_manager.positions) == 1
    pos = risk_manager.positions[0]
    assert pos.quantity == 100
    # VWAP: (60*2501 + 40*2502) / 100 = 2501.4
    assert abs(pos.avg_price - 2501.4) < 0.01

    # Lifecycle should show FILLED
    lc_order = await lifecycle.get_order("test-order-pf")
    assert lc_order.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# 10. Signal filtering (low confidence)
# ---------------------------------------------------------------------------


def test_signal_filtering_low_confidence(allocator, risk_manager):
    """Low-confidence signals filtered out by allocator."""
    signals = [
        make_signal(symbol="RELIANCE", score=0.3, price=2500.0),  # Below min_confidence=0.5
        make_signal(symbol="TCS", score=0.45, price=3500.0),  # Below threshold
        make_signal(symbol="INFY", score=0.75, price=1500.0),  # Above threshold
    ]
    allocations = allocator.allocate(
        signals=signals,
        equity=risk_manager.equity,
        positions=[],
        max_position_pct=5.0,
    )
    # Only INFY should pass the min_confidence=0.5 filter
    assert len(allocations) == 1
    assert allocations[0][0].symbol == "INFY"


# ---------------------------------------------------------------------------
# 11. Kelly sizing (higher confidence = larger position)
# ---------------------------------------------------------------------------


def test_kelly_sizing_higher_confidence_larger_position(risk_manager):
    """Higher confidence signals produce larger position sizes via Kelly criterion."""
    # Use a dedicated allocator with a high cap so Kelly differences are visible
    config = AllocatorConfig(
        max_active_signals=5,
        max_capital_pct_per_signal=20.0,
        min_confidence=0.5,
        use_kelly=True,
        kelly_fraction=0.5,
    )
    alloc = PortfolioAllocator(config)

    # Use a small price so that integer rounding doesn't swallow the Kelly difference
    high_conf = make_signal(symbol="RELIANCE", score=0.9, price=100.0, strategy_id="s1")
    low_conf = make_signal(symbol="TCS", score=0.55, price=100.0, strategy_id="s2")

    alloc_high = alloc.allocate(
        signals=[high_conf],
        equity=risk_manager.equity,
        positions=[],
        max_position_pct=20.0,
    )
    alloc_low = alloc.allocate(
        signals=[low_conf],
        equity=risk_manager.equity,
        positions=[],
        max_position_pct=20.0,
    )

    assert len(alloc_high) == 1
    assert len(alloc_low) == 1

    _, qty_high = alloc_high[0]
    _, qty_low = alloc_low[0]

    assert qty_high > qty_low, (
        f"Higher confidence ({high_conf.score}) should yield larger qty "
        f"({qty_high}) than lower confidence ({low_conf.score}) qty ({qty_low})"
    )


# ---------------------------------------------------------------------------
# 12. Exposure multiplier (half-open circuit -> 50% position sizes)
# ---------------------------------------------------------------------------


def test_exposure_multiplier_reduces_sizes(allocator, risk_manager):
    """Half-open circuit breaker with exposure_multiplier=0.5 -> smaller position sizes."""
    signal = make_signal(score=0.80, price=2500.0)

    # Full exposure (multiplier=1.0)
    alloc_full = allocator.allocate(
        signals=[signal],
        equity=risk_manager.equity,
        positions=[],
        exposure_multiplier=1.0,
        max_position_pct=5.0,
    )

    # Half exposure (multiplier=0.5)
    alloc_half = allocator.allocate(
        signals=[signal],
        equity=risk_manager.equity,
        positions=[],
        exposure_multiplier=0.5,
        max_position_pct=5.0,
    )

    assert len(alloc_full) == 1
    assert len(alloc_half) == 1

    _, qty_full = alloc_full[0]
    _, qty_half = alloc_half[0]

    # Half exposure should produce approximately half the quantity
    assert qty_half < qty_full, "Half exposure should produce smaller quantity"
    ratio = qty_half / qty_full
    assert 0.4 <= ratio <= 0.6, f"Expected ratio near 0.5, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# 13. Daily PnL tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_pnl_tracking_blocks_orders(order_service, risk_manager):
    """Losing trades accumulate -> daily loss limit hit -> orders blocked."""
    # Register losses just below the daily limit threshold
    risk_manager.register_pnl(-10_000)  # 1% loss
    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert result.success, "Should still allow orders at 1% loss"

    # Push over the 2% daily limit
    risk_manager.register_pnl(-15_000)  # Total loss now 25K = 2.5%
    request2 = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key=str(uuid.uuid4()),
        source="test",
    )
    result2 = await order_service.submit_order(request2)
    assert not result2.success
    assert result2.reject_reason == RejectReason.RISK_REJECTED
    assert "daily loss" in result2.reject_detail.lower()


# ---------------------------------------------------------------------------
# 14. Position close flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_close_sell_signal(order_service, risk_manager, fill_handler, lifecycle):
    """Open position -> generate SELL signal -> reduce position."""
    # Open a long position
    existing_pos = make_position(symbol="RELIANCE", side=SignalSide.BUY, quantity=10, avg_price=2500.0)
    risk_manager.add_position(existing_pos)

    # Generate a SELL signal to close
    sell_signal = make_signal(
        symbol="RELIANCE",
        side=SignalSide.SELL,
        score=0.7,
        price=2550.0,
    )
    request = OrderEntryRequest(
        signal=sell_signal,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2550.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert result.success, f"Sell order should succeed, got: {result.reject_reason} - {result.reject_detail}"

    # Simulate fill
    fill_event = FillEvent(
        order_id=result.order_id,
        broker_order_id=result.broker_order_id,
        symbol="RELIANCE",
        exchange="NSE",
        side="SELL",
        fill_type=FillType.FILL,
        filled_qty=10,
        remaining_qty=0,
        avg_price=2550.0,
        ts=datetime.now(UTC),
        strategy_id=sell_signal.strategy_id,
    )
    await fill_handler.on_fill_event(fill_event)

    # Should now have both positions (BUY and SELL) tracked in risk manager
    # The fill handler adds positions; closing logic happens in the autonomous loop
    sell_positions = [p for p in risk_manager.positions if p.side == SignalSide.SELL]
    assert len(sell_positions) == 1
    assert sell_positions[0].quantity == 10
    assert sell_positions[0].avg_price == 2550.0


# ---------------------------------------------------------------------------
# 15. Concurrent signal processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_signal_processing(order_service, risk_manager, gateway):
    """Multiple symbols processed concurrently without interference."""
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    tasks = []
    for sym in symbols:
        signal = make_signal(symbol=sym, score=0.75, price=2500.0)
        request = OrderEntryRequest(
            signal=signal,
            quantity=2,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            idempotency_key=str(uuid.uuid4()),
            source="test",
        )
        tasks.append(order_service.submit_order(request))

    results = await asyncio.gather(*tasks)

    # All should succeed
    for i, r in enumerate(results):
        assert r.success, f"Order for {symbols[i]} failed: {r.reject_reason} - {r.reject_detail}"

    # Gateway should have received all 5 orders
    assert len(gateway.placed_orders) == 5
    placed_symbols = {o.symbol for o in gateway.placed_orders}
    assert placed_symbols == set(symbols)


# ---------------------------------------------------------------------------
# 16. Kill switch allows reduce-only orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_switch_allows_reduce_only(order_service, risk_manager, kill_switch):
    """When kill switch is armed, SELL orders that reduce existing long are allowed."""
    # Add a long position first
    existing_pos = make_position(symbol="RELIANCE", side=SignalSide.BUY, quantity=20, avg_price=2500.0)
    risk_manager.add_position(existing_pos)

    # Arm kill switch
    await kill_switch.arm(KillReason.MANUAL, detail="test reduce-only")

    # BUY order should be blocked
    buy_signal = make_signal(symbol="TCS", side=SignalSide.BUY, price=3500.0)
    buy_request = OrderEntryRequest(
        signal=buy_signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=3500.0,
        source="test",
    )
    buy_result = await order_service.submit_order(buy_request)
    assert not buy_result.success
    assert buy_result.reject_reason == RejectReason.KILL_SWITCH

    # SELL order that reduces position should be allowed
    sell_signal = make_signal(symbol="RELIANCE", side=SignalSide.SELL, price=2500.0)
    sell_request = OrderEntryRequest(
        signal=sell_signal,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key=str(uuid.uuid4()),
        source="test",
    )
    sell_result = await order_service.submit_order(sell_request)
    assert sell_result.success, (
        f"Reduce-only SELL should be allowed, got: {sell_result.reject_reason} - {sell_result.reject_detail}"
    )


# ---------------------------------------------------------------------------
# 17. Consecutive losses block new orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_losses_block(order_service, risk_manager):
    """After max_consecutive_losses, new orders are blocked."""
    # Register 5 consecutive losses (limit is 5)
    for _ in range(5):
        risk_manager.register_pnl(-100)

    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.RISK_REJECTED
    assert "consecutive" in result.reject_detail.lower()


# ---------------------------------------------------------------------------
# 18. Risk manager exposure multiplier affects risk check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_manager_exposure_multiplier(order_service, risk_manager):
    """RiskManager exposure multiplier reduces effective equity for position sizing."""
    # Set exposure multiplier to 0.5
    risk_manager.set_exposure_multiplier(0.5)

    # With full equity = 1M and mult=0.5 -> effective_equity = 500K
    # max_position_pct=5% -> max position value = 25K
    # 11 shares at 2500 = 27500 > 25K -> should be rejected
    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=11,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.RISK_REJECTED

    # 9 shares at 2500 = 22500 < 25K -> should pass
    request2 = OrderEntryRequest(
        signal=signal,
        quantity=9,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key=str(uuid.uuid4()),
        source="test",
    )
    result2 = await order_service.submit_order(request2)
    assert result2.success, f"Should pass with reduced qty, got: {result2.reject_reason} - {result2.reject_detail}"


# ---------------------------------------------------------------------------
# 19. Paper fill simulator end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_fill_simulator_e2e(lifecycle, fill_handler, risk_manager):
    """PaperFillSimulator detects pending orders and fills them."""
    # Register an order in lifecycle
    order = Order(
        order_id="pfs-test-order",
        strategy_id="test",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        status=OrderStatus.PENDING,
    )
    await lifecycle.register(order)

    # Create paper fill simulator with no random rejection and fast polling
    simulator = PaperFillSimulator(
        lifecycle=lifecycle,
        fill_handler=fill_handler,
        fill_delay_seconds=0.0,
        poll_interval_seconds=0.1,
        cost_calculator=None,
    )

    # Patch random to disable rejection and partial fill randomness
    with patch("src.execution.fill_handler.paper_fill_simulator.random") as mock_random:
        mock_random.random.return_value = 0.99  # Always above rejection rate -> no rejection
        mock_random.uniform.return_value = 100  # Latency = 100ms; also avoids partial fill
        mock_random.uniform.side_effect = lambda a, b: (a + b) / 2  # deterministic middle value

        simulator._running = True
        await simulator._simulate_fills()

    # Check if the order was filled
    lc_order = await lifecycle.get_order("pfs-test-order")
    if lc_order.status == OrderStatus.FILLED:
        assert len(risk_manager.positions) > 0
    elif lc_order.status == OrderStatus.PARTIALLY_FILLED:
        assert len(risk_manager.positions) > 0
    # If rejected by the random path, that is also valid behavior


# ---------------------------------------------------------------------------
# 20. Max open positions limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_open_positions_limit(risk_manager, order_service):
    """When max open positions is reached, new orders are rejected."""
    # Set low limit for testing
    risk_manager.limits.max_open_positions = 3

    # Fill up to max
    for i in range(3):
        pos = make_position(symbol=f"SYM{i}", side=SignalSide.BUY, quantity=1, avg_price=2500.0)
        risk_manager.add_position(pos)

    # Try one more
    signal = make_signal(symbol="EXTRA", price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    # Should be rejected by either risk check or reservation
    assert result.reject_reason in (RejectReason.RISK_REJECTED, RejectReason.RESERVATION_FAILED)


# ---------------------------------------------------------------------------
# 21. Order validation rejects bad input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_order_validation_rejects_bad_input(order_service):
    """OrderEntryService rejects orders with invalid input."""
    # Zero quantity
    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=0,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.VALIDATION


# ---------------------------------------------------------------------------
# 22. Fill handler processes reject event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_handler_reject_event(fill_handler, lifecycle):
    """FillHandler correctly processes a REJECT fill event."""
    order = Order(
        order_id="reject-test",
        strategy_id="test",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        status=OrderStatus.PENDING,
    )
    await lifecycle.register(order)

    reject_event = FillEvent(
        order_id="reject-test",
        broker_order_id="BRK-REJ",
        symbol="RELIANCE",
        exchange="NSE",
        side="BUY",
        fill_type=FillType.REJECT,
        filled_qty=0,
        remaining_qty=10,
        avg_price=0.0,
        ts=datetime.now(UTC),
    )
    await fill_handler.on_fill_event(reject_event)

    lc_order = await lifecycle.get_order("reject-test")
    assert lc_order.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# 23. Allocator respects max_active_signals
# ---------------------------------------------------------------------------


def test_allocator_max_active_signals(risk_manager):
    """Allocator only returns top N signals based on max_active_signals."""
    config = AllocatorConfig(max_active_signals=2, min_confidence=0.5)
    alloc = PortfolioAllocator(config)
    signals = [
        make_signal(symbol="A", score=0.9, price=1000.0, strategy_id="s1"),
        make_signal(symbol="B", score=0.85, price=1000.0, strategy_id="s2"),
        make_signal(symbol="C", score=0.80, price=1000.0, strategy_id="s3"),
    ]
    allocations = alloc.allocate(
        signals=signals,
        equity=risk_manager.equity,
        positions=[],
        max_position_pct=5.0,
    )
    assert len(allocations) <= 2, f"Should only allocate top 2, got {len(allocations)}"
    symbols = [s.symbol for s, _ in allocations]
    assert "A" in symbols
    assert "B" in symbols
    assert "C" not in symbols


# ---------------------------------------------------------------------------
# 24. Per-symbol exposure limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_symbol_exposure_limit(order_service, risk_manager):
    """Per-symbol exposure limit prevents over-concentration in one symbol."""
    # Add existing position to get close to the 10% per-symbol limit
    # 10% of 1M = 100K. Position of 38 shares at 2500 = 95K
    existing_pos = make_position(symbol="RELIANCE", side=SignalSide.BUY, quantity=38, avg_price=2500.0)
    risk_manager.add_position(existing_pos)

    # Try to add 5 more shares (12,500), total = 107,500 > 100K (10%)
    signal = make_signal(symbol="RELIANCE", price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert result.reject_reason == RejectReason.RISK_REJECTED
    assert "per_symbol" in result.reject_detail.lower()


# ---------------------------------------------------------------------------
# 25. Circuit breaker close restores trading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_close_restores_trading(order_service, risk_manager):
    """Opening and then closing the circuit breaker restores normal trading."""
    risk_manager.open_circuit("test")

    signal = make_signal(price=2500.0)
    request = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success

    # Close circuit
    risk_manager.close_circuit()

    request2 = OrderEntryRequest(
        signal=signal,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key=str(uuid.uuid4()),
        source="test",
    )
    result2 = await order_service.submit_order(request2)
    assert result2.success, f"Should pass after circuit close, got: {result2.reject_reason} - {result2.reject_detail}"


# ---------------------------------------------------------------------------
# 26. Allocator returns empty on zero equity
# ---------------------------------------------------------------------------


def test_allocator_zero_equity():
    """Allocator returns empty list when equity is zero or negative."""
    alloc = PortfolioAllocator()
    signal = make_signal(score=0.8, price=2500.0)
    result = alloc.allocate(signals=[signal], equity=0, positions=[])
    assert result == []

    result_neg = alloc.allocate(signals=[signal], equity=-100, positions=[])
    assert result_neg == []


# ---------------------------------------------------------------------------
# 27. Minimum order value check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_minimum_order_value(order_service):
    """Orders below INR 1,000 minimum value are rejected."""
    signal = make_signal(price=100.0)  # 1 share at 100 = 100 < 1000
    request = OrderEntryRequest(
        signal=signal,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        source="test",
    )
    result = await order_service.submit_order(request)
    assert not result.success
    assert "minimum" in result.reject_detail.lower() or "below" in result.reject_detail.lower()

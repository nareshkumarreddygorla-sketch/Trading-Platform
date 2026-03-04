"""
Broker realism tests: timeout, session expiry, cancel, order status mapping.
Invariant: no fake Order in live mode (live must return Order with broker_order_id from broker or raise).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.events import OrderStatus
from src.execution.angel_one_gateway import AngelOneExecutionGateway, _to_order_status
from src.execution.broker.angel_one_http_client import BrokerClientError


def test_order_status_mapping():
    assert _to_order_status("pending") == OrderStatus.PENDING
    assert _to_order_status("open") == OrderStatus.LIVE
    assert _to_order_status("cancelled") == OrderStatus.CANCELLED
    assert _to_order_status("rejected") == OrderStatus.REJECTED
    assert _to_order_status("complete") == OrderStatus.FILLED
    assert _to_order_status("completed") == OrderStatus.FILLED
    assert _to_order_status("traded") == OrderStatus.FILLED
    assert _to_order_status("") == OrderStatus.PENDING
    assert _to_order_status("unknown") == OrderStatus.PENDING


@pytest.mark.asyncio
async def test_paper_mode_returns_in_memory_order():
    gw = AngelOneExecutionGateway(api_key="x", api_secret="", access_token="", paper=True)
    order = await gw.place_order("INFY", "NSE", "BUY", 10.0, "MARKET", strategy_id="s1")
    assert order is not None
    assert order.symbol == "INFY"
    assert order.quantity == 10.0
    assert order.metadata.get("paper") is True
    # Paper has no broker_order_id from broker; that's allowed for paper
    assert order.broker_order_id is None or order.order_id


@pytest.mark.asyncio
async def test_live_mode_never_returns_fake_order_on_failure():
    """Live mode must raise on broker failure, never return in-memory Order without broker response."""
    gw = AngelOneExecutionGateway(
        api_key="key",
        api_secret="",
        access_token="",
        paper=False,
        refresh_token="ref",
    )
    with patch.object(gw._client, "place_order", side_effect=BrokerClientError("Broker down")):
        with pytest.raises(BrokerClientError):
            await gw.place_order("INFY", "NSE", "BUY", 10.0, "MARKET", strategy_id="s1")


@pytest.mark.asyncio
async def test_live_mode_returns_order_with_broker_order_id_on_success():
    """When broker responds, Order must have broker_order_id from response."""
    gw = AngelOneExecutionGateway(
        api_key="key",
        api_secret="",
        access_token="tok",
        paper=False,
        refresh_token="ref",
    )
    gw._client._access_token = "tok"
    gw._client._refresh_token = "ref"
    with patch.object(
        gw._client,
        "place_order",
        return_value={"orderid": "201020000000080", "uniqueorderid": "05ebf91b-bea4-4a1d-b0f2-4259606570e3"},
    ):
        order = await gw.place_order("INFY", "NSE", "BUY", 10.0, "MARKET", strategy_id="s1")
    assert order.broker_order_id == "201020000000080"
    assert order.order_id  # uniqueorderid or orderid
    assert order.status == OrderStatus.LIVE
    assert order.metadata.get("paper") is not True


@pytest.mark.asyncio
async def test_broker_timeout_raises():
    """Timeout on place_order must raise and not return fake Order."""
    gw = AngelOneExecutionGateway(
        api_key="key",
        api_secret="",
        access_token="tok",
        paper=False,
        refresh_token="ref",
        request_timeout=0.5,
    )
    gw._client._access_token = "tok"
    gw._client._refresh_token = "ref"

    def slow_place(*args, **kwargs):
        import time
        time.sleep(5)
        return {"orderid": "1", "uniqueorderid": "u1"}

    with patch.object(gw._client, "place_order", side_effect=slow_place):
        with pytest.raises(BrokerClientError):
            await gw.place_order("INFY", "NSE", "BUY", 10.0, "MARKET", strategy_id="s1")


@pytest.mark.asyncio
async def test_cancel_order_paper_returns_true():
    gw = AngelOneExecutionGateway(api_key="x", api_secret="", access_token="", paper=True)
    ok = await gw.cancel_order("oid1")
    assert ok is True


@pytest.mark.asyncio
async def test_cancel_order_live_success():
    gw = AngelOneExecutionGateway(
        api_key="key",
        api_secret="",
        access_token="tok",
        paper=False,
        refresh_token="ref",
    )
    gw._client._access_token = "tok"
    gw._client._refresh_token = "ref"
    with patch.object(gw._client, "cancel_order", return_value={"status": "ok"}):
        ok = await gw.cancel_order("oid1", broker_order_id="201020000000080")
    assert ok is True


@pytest.mark.asyncio
async def test_get_order_status_mapping_from_live():
    gw = AngelOneExecutionGateway(
        api_key="key",
        api_secret="",
        access_token="tok",
        paper=False,
        refresh_token="ref",
    )
    gw._client._access_token = "tok"
    gw._client._refresh_token = "ref"
    with patch.object(
        gw._client,
        "get_order_details",
        return_value={"orderstatus": "rejected", "orderid": "123"},
    ):
        status = await gw.get_order_status("uuid-1")
    assert status == OrderStatus.REJECTED

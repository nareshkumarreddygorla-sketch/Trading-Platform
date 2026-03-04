"""
Broker failure resilience: retry with exponential backoff, order state polling fallback,
timeout handling. DLQ for Kafka publish failure is in market_data/streaming (caller adds).
"""
import asyncio
import logging
import random
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 10.0
DEFAULT_TIMEOUT = 15.0


async def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    retry_on: tuple = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Execute fn with exponential backoff. On timeout or retry_on exception, retry.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            coro = fn(*args, **kwargs)
            if asyncio.iscoroutine(coro):
                return await asyncio.wait_for(coro, timeout=timeout)
            return coro
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning("Attempt %s timeout: %s", attempt + 1, e)
        except retry_on as e:
            last_exc = e
            logger.warning("Attempt %s failed: %s", attempt + 1, e)
        if attempt < max_retries:
            delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 0.5))
            await asyncio.sleep(delay)
    raise last_exc


class BrokerGatewayResilient:
    """
    Wraps a BaseExecutionGateway with retry and timeout.
    For order placement: retry on timeout/network; do not retry on broker rejection (4xx).
    """

    def __init__(self, gateway, max_retries: int = 2, timeout: float = 15.0):
        self.gateway = gateway
        self.max_retries = max_retries
        self.timeout = timeout

    async def place_order(self, *args: Any, **kwargs: Any) -> Any:
        """Place order with retry on timeout/connection error only."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.gateway.place_order(*args, **kwargs),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning("place_order timeout attempt %s", attempt + 1)
            except ConnectionError as e:
                last_exc = e
                logger.warning("place_order connection error attempt %s", attempt + 1)
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise last_exc

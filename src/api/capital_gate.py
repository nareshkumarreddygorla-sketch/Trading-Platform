"""
Capital deployment gate: validate Redis, broker, market data, stress/restart pass.
If validate() fails, system must refuse to enable autonomous live mode. Manual paper still allowed.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class CapitalGate:
    """
    validate() runs checks: Redis, broker, market data health.
    Optional: stress tests pass, restart simulation pass (caller can run and set).
    If any critical check fails, autonomous live mode must not be enabled.
    """

    def __init__(
        self,
        check_redis: Callable[[], Any] | None = None,
        check_broker: Callable[[], Any] | None = None,
        check_market_data: Callable[[], Any] | None = None,
        stress_tests_passed: bool = False,
        restart_simulation_passed: bool = False,
    ):
        self.check_redis = check_redis
        self.check_broker = check_broker
        self.check_market_data = check_market_data
        self.stress_tests_passed = stress_tests_passed
        self.restart_simulation_passed = restart_simulation_passed

    async def validate(self) -> dict[str, Any]:
        """
        Run all checks. Returns dict with keys: ok (bool), checks (dict), message (str).
        If ok is False, autonomous live mode must not be enabled.
        """
        checks: dict[str, Any] = {}
        all_ok = True

        if self.check_redis:
            try:
                result = self.check_redis()
                if asyncio.iscoroutine(result):
                    result = await result
                checks["redis"] = "ok" if result else "fail"
                if not result:
                    all_ok = False
            except Exception as e:
                checks["redis"] = str(e)
                all_ok = False
        else:
            checks["redis"] = "skipped"

        if self.check_broker:
            try:
                result = self.check_broker()
                if asyncio.iscoroutine(result):
                    result = await result
                checks["broker"] = "ok" if result else "fail"
                if not result:
                    all_ok = False
            except Exception as e:
                checks["broker"] = str(e)
                all_ok = False
        else:
            checks["broker"] = "skipped"

        if self.check_market_data:
            try:
                result = self.check_market_data()
                if asyncio.iscoroutine(result):
                    result = await result
                checks["market_data"] = "ok" if result else "fail"
                if not result:
                    all_ok = False
            except Exception as e:
                checks["market_data"] = str(e)
                all_ok = False
        else:
            checks["market_data"] = "skipped"

        checks["stress_tests_passed"] = self.stress_tests_passed
        checks["restart_simulation_passed"] = self.restart_simulation_passed
        if not self.stress_tests_passed or not self.restart_simulation_passed:
            all_ok = False

        message = "All checks passed" if all_ok else "One or more checks failed; do not enable autonomous live mode"
        return {"ok": all_ok, "checks": checks, "message": message}

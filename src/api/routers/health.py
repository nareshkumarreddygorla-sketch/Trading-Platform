from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness: API process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """
    Readiness: check Redis (and optionally DB, broker); return 200 when core app is up.
    Redis is optional in development mode.
    """
    import os
    checks = {}
    is_dev = os.getenv("ENV", "development") == "development"

    # Redis check (optional in dev mode)
    try:
        from src.core.config import get_settings
        redis_url = get_settings().market_data.redis_url
    except Exception:
        redis_url = "redis://localhost:6379/0"
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url, decode_responses=True)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except ImportError:
        checks["redis"] = "skipped (redis package not installed)"
    except Exception as e:
        checks["redis"] = f"unavailable: {e}" if is_dev else str(e)

    # Database check
    persistence = getattr(request.app.state, "persistence_service", None)
    if persistence is not None:
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(loop.run_in_executor(None, persistence.list_positions_sync), timeout=3.0)
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = str(e)

    # Core services
    checks["risk_manager"] = "ok" if getattr(request.app.state, "risk_manager", None) else "not configured"
    checks["bar_cache"] = "ok" if getattr(request.app.state, "bar_cache", None) else "not configured"

    # In dev mode, don't fail on Redis
    if not is_dev and checks.get("redis") not in ("ok", "skipped (redis package not installed)"):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": checks, "message": "Redis unavailable"},
        )
    return {"status": "ready", "checks": checks}


@router.get("/health/self-test")
async def self_test(request: Request):
    """
    Self-test: DB, Redis, broker connectivity and optional invariants.
    Returns 200 with check results; 503 if any critical check fails.
    """
    import asyncio
    checks = {}
    critical_ok = True

    # Redis (use configured URL when available)
    try:
        import redis.asyncio as redis
        try:
            from src.core.config import get_settings
            redis_url = get_settings().market_data.redis_url
        except Exception:
            redis_url = "redis://localhost:6379/0"
        r = redis.from_url(redis_url, decode_responses=True)
        await asyncio.wait_for(r.ping(), timeout=2.0)
        await r.aclose()
        checks["redis"] = "ok"
    except asyncio.TimeoutError:
        checks["redis"] = "timeout"
        critical_ok = False
    except Exception as e:
        checks["redis"] = str(e)
        critical_ok = False

    # DB (if persistence configured)
    persistence = getattr(request.app.state, "persistence_service", None)
    if persistence is not None:
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, persistence.list_positions_sync),
                timeout=5.0,
            )
            checks["database"] = "ok"
        except asyncio.TimeoutError:
            checks["database"] = "timeout"
            critical_ok = False
        except Exception as e:
            checks["database"] = str(e)
            critical_ok = False
    else:
        checks["database"] = "skipped (no persistence)"

    # Broker: optional (gateway may not be reachable)
    order_entry = getattr(request.app.state, "order_entry_service", None)
    if order_entry is not None and getattr(order_entry, "order_router", None):
        try:
            gateway = getattr(order_entry.order_router, "default_gateway", None) or getattr(order_entry.order_router, "gateway", None)
            if gateway and hasattr(gateway, "get_positions"):
                await asyncio.wait_for(gateway.get_positions(), timeout=5.0)
                checks["broker"] = "ok"
            else:
                checks["broker"] = "skipped (no get_positions)"
        except asyncio.TimeoutError:
            checks["broker"] = "timeout"
        except Exception as e:
            checks["broker"] = str(e)
    else:
        checks["broker"] = "skipped (no gateway)"

    if not critical_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "checks": checks},
        )
    return {"status": "ok", "checks": checks}

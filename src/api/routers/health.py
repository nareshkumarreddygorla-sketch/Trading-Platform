from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: Dict[str, Any]
    message: Optional[str] = None


class SelfTestResponse(BaseModel):
    status: str
    checks: Dict[str, Any]


class DeepHealthCheckStatus(BaseModel):
    status: str
    reason: Optional[str] = None
    joblib_count: Optional[int] = None
    pt_count: Optional[int] = None


class DeepHealthResponse(BaseModel):
    status: str
    checks: Dict[str, Any]


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness: API process is up."""
    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
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


@router.get("/health/self-test", response_model=SelfTestResponse)
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


@router.get("/health/deep", response_model=DeepHealthResponse)
async def deep_health(request: Request):
    """
    Deep health check: Database, Redis, model files, market data feed, broker session.
    Returns 200 if all critical checks pass, 503 if any fail.
    Each component reports its own status in the response JSON.
    """
    import asyncio
    import os
    from pathlib import Path

    checks = {}
    all_ok = True

    # 1. Database: SELECT 1 with 2s timeout
    try:
        from src.core.config import get_settings
        _settings = get_settings()
        db_url = _settings.database_url or os.environ.get("DATABASE_URL")
        if db_url:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url, pool_pre_ping=True)
            loop = asyncio.get_running_loop()

            def _db_check():
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                engine.dispose()

            await asyncio.wait_for(loop.run_in_executor(None, _db_check), timeout=2.0)
            checks["database"] = {"status": "ok"}
        else:
            # Fallback: try persistence service
            persistence = getattr(request.app.state, "persistence_service", None)
            if persistence is not None:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, persistence.list_positions_sync),
                    timeout=2.0,
                )
                checks["database"] = {"status": "ok"}
            else:
                checks["database"] = {"status": "skipped", "reason": "no database configured"}
    except asyncio.TimeoutError:
        checks["database"] = {"status": "fail", "reason": "timeout (2s)"}
        all_ok = False
    except Exception as e:
        checks["database"] = {"status": "fail", "reason": str(e)}
        all_ok = False

    # 2. Redis: PING with 1s timeout
    try:
        import redis.asyncio as aioredis
        try:
            from src.core.config import get_settings
            redis_url = get_settings().market_data.redis_url
        except Exception:
            redis_url = "redis://localhost:6379/0"
        r = aioredis.from_url(redis_url, decode_responses=True)
        await asyncio.wait_for(r.ping(), timeout=1.0)
        await r.aclose()
        checks["redis"] = {"status": "ok"}
    except asyncio.TimeoutError:
        checks["redis"] = {"status": "fail", "reason": "timeout (1s)"}
        all_ok = False
    except ImportError:
        checks["redis"] = {"status": "skipped", "reason": "redis package not installed"}
    except Exception as e:
        checks["redis"] = {"status": "fail", "reason": str(e)}
        all_ok = False

    # 3. Model files: check models/*.joblib and models/*.pt exist
    try:
        project_root = Path(__file__).resolve().parents[3]
        models_dir = project_root / "models"
        joblib_files = list(models_dir.glob("*.joblib"))
        pt_files = list(models_dir.glob("*.pt"))
        if joblib_files or pt_files:
            checks["model_files"] = {
                "status": "ok",
                "joblib_count": len(joblib_files),
                "pt_count": len(pt_files),
            }
        else:
            checks["model_files"] = {"status": "fail", "reason": "no .joblib or .pt files found"}
            all_ok = False
    except Exception as e:
        checks["model_files"] = {"status": "fail", "reason": str(e)}
        all_ok = False

    # 4. Market data: check is_healthy() on WS connector or feed manager
    try:
        mds = getattr(request.app.state, "market_data_service", None)
        if mds is not None:
            healthy = mds.is_healthy()
            checks["market_data"] = {"status": "ok" if healthy else "degraded"}
            if not healthy:
                all_ok = False
        else:
            # Check YFinance feeder fallback
            yff = getattr(request.app.state, "yf_feeder", None)
            if yff is not None:
                running = getattr(yff, "_running", False)
                checks["market_data"] = {"status": "ok" if running else "degraded"}
                if not running:
                    all_ok = False
            else:
                checks["market_data"] = {"status": "skipped", "reason": "no feed configured"}
    except Exception as e:
        checks["market_data"] = {"status": "fail", "reason": str(e)}
        all_ok = False

    # 5. Broker: check gateway session validity
    try:
        gateway = getattr(request.app.state, "gateway", None)
        if gateway is not None and not getattr(gateway, "paper", True):
            if hasattr(gateway, "auth_failed") and gateway.auth_failed:
                checks["broker"] = {"status": "fail", "reason": "auth_failed flag set"}
                all_ok = False
            elif hasattr(gateway, "get_orders"):
                await asyncio.wait_for(gateway.get_orders(limit=1), timeout=5.0)
                checks["broker"] = {"status": "ok"}
            else:
                checks["broker"] = {"status": "ok", "reason": "no order API to probe"}
        elif gateway is not None:
            checks["broker"] = {"status": "ok", "reason": "paper mode"}
        else:
            checks["broker"] = {"status": "skipped", "reason": "no gateway configured"}
    except asyncio.TimeoutError:
        checks["broker"] = {"status": "fail", "reason": "timeout (5s)"}
        all_ok = False
    except Exception as e:
        checks["broker"] = {"status": "fail", "reason": str(e)}
        all_ok = False

    if not all_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "checks": checks},
        )
    return {"status": "healthy", "checks": checks}

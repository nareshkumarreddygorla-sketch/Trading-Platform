"""
FastAPI application: health, market, strategies, risk, orders, backtest, metrics.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Prevent OMP abort on macOS

import logging

# Load .env before anything reads os.environ
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402

# Import the modular lifespan
from .lifespan import lifespan  # noqa: E402
from .routers import (  # noqa: E402
    agents,
    alpha_research,
    alt_data,
    attribution,
    audit,
    auth,
    backtest,
    broker,
    capital,
    data_pipeline,
    health,
    llm_strategy,
    market,
    marketplace,
    options,
    orders,
    performance,
    reconciliation,
    risk,
    self_learning,
    simulation,
    strategies,
    trading,
    training,
)

logger = logging.getLogger(__name__)


def _prometheus_metrics() -> Response:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Gauge, generate_latest

        # Register trading gauges (idempotent via try/except)
        def _get_or_create_gauge(name, doc):
            try:
                return Gauge(name, doc)
            except ValueError:
                # Already registered
                c = REGISTRY._names_to_collectors.get(name)
                if c is not None:
                    return c
                return Gauge(name, doc)

        try:
            # Get app state from module-level app
            _app = globals().get("app")
            if _app:
                rm = getattr(_app.state, "risk_manager", None)
                if rm:
                    _get_or_create_gauge("trading_equity", "Current portfolio equity").set(rm.equity)
                    _get_or_create_gauge("trading_daily_pnl", "Daily PnL").set(rm.daily_pnl)
                    _get_or_create_gauge("trading_open_positions", "Number of open positions").set(len(rm.positions))
                    _get_or_create_gauge("trading_circuit_open", "Circuit breaker state (1=open)").set(
                        1 if rm.is_circuit_open() else 0
                    )

                cb = getattr(_app.state, "circuit_breaker", None)
                if cb:
                    peak = getattr(cb, "_peak_equity", 0)
                    _get_or_create_gauge("trading_peak_equity", "Peak equity").set(peak)

                ks = getattr(_app.state, "kill_switch", None)
                if ks:
                    _get_or_create_gauge("trading_kill_switch_armed", "Kill switch armed (1=yes)").set(
                        1 if getattr(ks, "_armed", False) else 0
                    )

                sm = getattr(_app.state, "safe_mode", False)
                _get_or_create_gauge("trading_safe_mode", "Safe mode active (1=yes)").set(1 if sm else 0)

                al = getattr(_app.state, "autonomous_loop", None)
                if al:
                    _get_or_create_gauge("trading_autonomous_ticks", "Autonomous loop tick count").set(
                        getattr(al, "_tick_count", 0)
                    )
                    _get_or_create_gauge("trading_open_trades", "Autonomous loop tracked trades").set(
                        len(getattr(al, "_open_trades", {}))
                    )

                # Sprint 10.10: Additional Prometheus metrics
                mr = getattr(_app.state, "model_registry", None)
                if mr:
                    _get_or_create_gauge("trading_models_loaded", "Number of loaded ML models").set(
                        sum(1 for m in mr.list_all() if getattr(m, "_loaded", True))
                    )
                ws_mgr = getattr(_app.state, "ws_manager", None)
                if ws_mgr:
                    _get_or_create_gauge("trading_ws_connections", "Active WebSocket connections").set(
                        len(getattr(ws_mgr, "_active_connections", []))
                    )
        except Exception:
            pass

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(
            content="prometheus_client not installed",
            media_type="text/plain",
            status_code=503,
        )


def _get_allowed_origins() -> list[str]:
    """Return CORS allowed origins from env or sensible defaults."""
    import os

    custom = os.environ.get("CORS_ORIGINS")
    if custom:
        origins = [o.strip() for o in custom.split(",") if o.strip()]
        # Reject wildcard "*" when credentials are enabled (browsers block it anyway)
        if "*" in origins:
            logger.error(
                "CORS_ORIGINS contains '*' which is incompatible with allow_credentials=True. "
                "Falling back to no origins. Set explicit origins."
            )
            return []
        return origins
    env = os.environ.get("ENV", "development").lower()
    if env == "production":
        frontend_url = os.environ.get("FRONTEND_URL")
        if not frontend_url:
            logger.error(
                "PRODUCTION: Neither CORS_ORIGINS nor FRONTEND_URL is set. "
                "CORS will reject all cross-origin requests. "
                "Set CORS_ORIGINS or FRONTEND_URL to your frontend domain."
            )
            return []
        return [frontend_url]
    # Development: allow local origins only
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


# Initialize structured logging before app creation (replaces inline _setup_structured_logging)
from .logging_config import configure_logging  # noqa: E402

configure_logging()


def create_app() -> FastAPI:
    _env = os.environ.get("ENV", "development").lower()
    # Disable interactive API docs in production to reduce attack surface
    docs_kwargs = {} if _env != "production" else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(
        title="Autonomous Trading Platform API",
        version="1.0.0",
        description="Institutional-grade multi-market trading engine",
        lifespan=lifespan,
        **docs_kwargs,
    )

    # Security & observability middleware (order matters: outermost first)
    from .middleware import RateLimitMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=120, auth_requests_per_minute=10)
    allowed_origins = _get_allowed_origins()
    if not allowed_origins:
        logger.warning("CORS: No allowed origins configured — all cross-origin requests will be rejected.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
        max_age=600,  # Cache preflight for 10 minutes to reduce OPTIONS requests
    )
    # ── Global exception handlers (Sprint 9.1) ──
    import uuid

    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        error_id = str(uuid.uuid4())[:8]
        logger.exception("Unhandled error [%s]: %s", error_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "error_id": error_id,
                "message": "An unexpected error occurred",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "details": [
                    {"field": str(e.get("loc", ["unknown"])[-1]), "message": e.get("msg", "")} for e in exc.errors()
                ],
            },
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root():
        """Root: welcome page with links to docs and health."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Trading Platform API</title></head>
        <body style="font-family: system-ui; max-width: 600px; margin: 2rem auto; padding: 0 1rem;">
            <h1>Autonomous Trading Platform API</h1>
            <p>Version 1.0.0</p>
            <ul>
                <li><a href="/docs">API docs (Swagger)</a></li>
                <li><a href="/redoc">ReDoc</a></li>
                <li><a href="/health">Health</a></li>
                <li><a href="/ready">Ready</a></li>
            </ul>
        </body>
        </html>
        """

    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1", tags=["Auth"])
    app.include_router(market.router, prefix="/api/v1/market", tags=["Market"])
    app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
    app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk"])
    app.include_router(orders.router, prefix="/api/v1", tags=["Orders"])
    app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtest"])
    app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])
    app.include_router(alpha_research.router, prefix="/api/v1", tags=["Alpha Research"])
    app.include_router(reconciliation.router, prefix="/api/v1", tags=["Reconciliation"])
    app.include_router(capital.router, prefix="/api/v1", tags=["Capital"])
    app.include_router(audit.router, prefix="/api/v1", tags=["Audit"])
    app.include_router(performance.router, prefix="/api/v1/performance", tags=["Performance"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])
    app.include_router(broker.router, prefix="/api/v1", tags=["Broker"])
    app.include_router(attribution.router, tags=["Attribution"])
    app.include_router(simulation.router, tags=["Simulation"])
    app.include_router(data_pipeline.router, tags=["Data Pipeline"])
    app.include_router(self_learning.router, tags=["Self-Learning"])
    app.include_router(marketplace.router, tags=["Marketplace"])
    app.include_router(options.router, tags=["Options"])
    app.include_router(llm_strategy.router, tags=["Strategy Builder"])
    app.include_router(alt_data.router, tags=["Alternative Data"])

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Accept WebSocket for live updates. Supports two auth modes:
        1. Sec-WebSocket-Protocol header with JWT (preferred, avoids token in URL)
        2. Query param ?token=<jwt> (legacy fallback)
        """
        import os
        from urllib.parse import parse_qs

        from .auth import _decode_token
        from .ws_manager import get_ws_manager

        user_id = "anonymous"
        selected_subprotocol = None

        # Try auth via subprotocol header first (preferred - avoids token in URL)
        subprotocols = websocket.headers.get("sec-websocket-protocol", "")
        token = None
        if subprotocols:
            for proto in subprotocols.split(","):
                proto = proto.strip()
                if proto.startswith("access_token."):
                    token = proto[len("access_token.") :]
                    selected_subprotocol = proto
                    break

        # Fallback: query param (legacy)
        if not token:
            query = parse_qs(websocket.scope.get("query_string", b"").decode())
            token = query.get("token", [None])[0]

        if os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET"):
            if not token:
                # Must accept before closing with custom code; raw close sends 403
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.close(code=4001, reason="missing_token")
                return
            payload = _decode_token(token)
            if not payload:
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.close(code=4001, reason="invalid_token")
                return
            user_id = payload.get("sub") or payload.get("user_id") or "unknown"

        mgr = get_ws_manager()
        if mgr:
            await mgr.connect(websocket, user_id=user_id, subprotocol=selected_subprotocol)
        else:
            await websocket.accept(subprotocol=selected_subprotocol)
        try:
            await websocket.send_json({"type": "connected", "message": "Live", "user_id": user_id})
            while True:
                try:
                    data = await websocket.receive_text()
                    # Handle ping/pong heartbeat from client
                    if data == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif data == "pong":
                        # Client responding to server-initiated heartbeat ping
                        if mgr:
                            mgr.record_pong(websocket)
                    else:
                        # Try parsing JSON messages (e.g. {"type": "pong"})
                        try:
                            import json as _json

                            msg = _json.loads(data)
                            if isinstance(msg, dict) and msg.get("type") == "pong":
                                if mgr:
                                    mgr.record_pong(websocket)
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    break
        except Exception:
            pass
        finally:
            if mgr:
                await mgr.disconnect(websocket)
            try:
                await websocket.close()
            except Exception:
                pass

    app.get("/metrics", include_in_schema=False)(_prometheus_metrics)

    # Debug endpoint: only available in development (never expose internal state in production)
    if os.environ.get("ENV", "development").lower() != "production":

        @app.get("/api/v1/debug/bar-cache", include_in_schema=False)
        def debug_bar_cache():
            bc = getattr(app.state, "bar_cache", None)
            if not bc:
                return {"error": "no bar_cache"}
            keys = list(bc._bars.keys())
            counts = {k: len(bc._bars[k]) for k in keys[:30]}
            from src.core.events import Exchange

            symbols = bc.symbols_with_bars(Exchange.NSE, "1m", min_bars=20)
            return {"total_keys": len(keys), "sample_counts": counts, "symbols_with_20_bars": symbols}

    return app


app = create_app()

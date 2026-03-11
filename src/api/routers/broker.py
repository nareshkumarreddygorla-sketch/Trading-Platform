"""
Broker credential management: configure, validate, disconnect, and check Angel One connection.
Enables zero-intervention flow: user enters creds -> system validates -> auto-switches to live.
Credentials are encrypted at rest using Fernet (AES-128-CBC) derived from JWT_SECRET via PBKDF2.
"""

import asyncio
import base64
import hashlib
import logging
import os
import secrets
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..auth import get_current_user

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential encryption helpers
# ---------------------------------------------------------------------------
_fernet_instance = None
_encryption_available = False


def _derive_fernet_key() -> bytes | None:
    """Derive a Fernet-compatible key from JWT_SECRET using PBKDF2."""
    secret = os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET")
    if not secret:
        return None
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        b"trading-platform-broker-cred-salt",
        iterations=100_000,
        dklen=32,
    )
    # Fernet needs a url-safe base64 encoded 32-byte key
    return base64.urlsafe_b64encode(dk)


def _get_fernet():
    """Lazily initialize Fernet cipher. Returns None if unavailable."""
    global _fernet_instance, _encryption_available
    if _fernet_instance is not None:
        return _fernet_instance
    try:
        from cryptography.fernet import Fernet

        key = _derive_fernet_key()
        if key is None:
            logger.warning("Broker cred encryption: no JWT_SECRET set, falling back to base64 encoding")
            _encryption_available = False
            return None
        _fernet_instance = Fernet(key)
        _encryption_available = True
        return _fernet_instance
    except ImportError:
        logger.warning(
            "Broker cred encryption: 'cryptography' package not installed. "
            "Falling back to base64 encoding. Install with: pip install cryptography"
        )
        _encryption_available = False
        return None


def encrypt_credential(value: str) -> str:
    """Encrypt a credential string. Returns Fernet ciphertext or base64-encoded fallback."""
    f = _get_fernet()
    if f is not None:
        return f.encrypt(value.encode("utf-8")).decode("utf-8")
    # Fallback: base64 encode (obfuscation only, NOT secure)
    return "b64:" + base64.b64encode(value.encode("utf-8")).decode("utf-8")


def decrypt_credential(value: str) -> str:
    """Decrypt a credential string. Handles Fernet ciphertext or base64 fallback."""
    if value.startswith("b64:"):
        return base64.b64decode(value[4:]).decode("utf-8")
    f = _get_fernet()
    if f is not None:
        try:
            return f.decrypt(value.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.warning(
                "Fernet decryption failed (value may be unencrypted/legacy): %s",
                type(e).__name__,
            )
            return value
    return value


router = APIRouter()


class BrokerCredentials(BaseModel):
    api_key: str = Field(..., min_length=1, description="Angel One API key")
    client_id: str = Field(..., min_length=1, description="Angel One client ID")
    password: str = Field(..., min_length=1, description="Angel One password")
    totp_secret: str = Field(..., min_length=1, description="Base32 TOTP secret for auto-login")
    mode: str = Field("live", description="Trading mode: 'paper' or 'live'")


class BrokerCredentialResponse(BaseModel):
    status: str
    message: str
    mode: str | None = None
    connected: bool = False
    auto_started: bool = False
    confirm_token: str | None = None


class BrokerStatusResponse(BaseModel):
    connected: bool
    mode: str
    healthy: bool
    safe_mode: bool | None = None
    has_credentials: bool | None = None
    client_id: str | None = None
    last_connected: str | None = None
    autonomous_running: bool | None = None
    tick_count: int | None = None
    open_trades: int | None = None
    message: str | None = None


class BrokerDisconnectResponse(BaseModel):
    status: str
    message: str
    mode: str
    connected: bool


class BrokerValidateResponse(BaseModel):
    valid: bool
    message: str


class LiveModeConfirmRequest(BaseModel):
    confirm_token: str = Field(..., min_length=1, description="Confirmation token from configure step")


# Paper-to-live confirmation tokens (token -> {creds, ts, user_id})
# Redis-backed for multi-worker deployments, with in-memory fallback.
_pending_live_switch: dict = {}
_CONFIRM_TOKEN_TTL_SECONDS = 300  # 5 minutes
_REDIS_KEY_PREFIX = "broker:confirm:"

_redis_client = None
_redis_available = False


def _get_redis():
    """Lazily initialize Redis client for pending live switch tokens."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None
    try:
        import redis

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, socket_timeout=2, decode_responses=True)
        _redis_client.ping()
        _redis_available = True
        logger.info("Broker confirm tokens: using Redis store (%s)", redis_url)
        return _redis_client
    except Exception as e:
        _redis_available = False
        logger.info("Broker confirm tokens: Redis unavailable (%s), using in-memory fallback", e)
        return None


def _store_pending_token(token: str, data: dict) -> None:
    """Store a pending confirmation token in Redis (with TTL) and in-memory fallback."""
    import json as _json

    _pending_live_switch[token] = data
    rc = _get_redis()
    if rc is not None:
        try:
            rc.setex(
                f"{_REDIS_KEY_PREFIX}{token}",
                _CONFIRM_TOKEN_TTL_SECONDS,
                _json.dumps(data, default=str),
            )
        except Exception as e:
            logger.warning("Redis SET for confirm token failed (in-memory still valid): %s", e)


def _pop_pending_token(token: str) -> dict | None:
    """Retrieve and delete a pending confirmation token. Tries Redis first, falls back to in-memory."""
    import json as _json

    rc = _get_redis()
    if rc is not None:
        try:
            raw = rc.getdel(f"{_REDIS_KEY_PREFIX}{token}")
            if raw is not None:
                # Also remove from in-memory dict to keep them in sync
                _pending_live_switch.pop(token, None)
                return _json.loads(raw)
        except Exception as e:
            logger.warning("Redis GETDEL for confirm token failed, falling back to in-memory: %s", e)
    return _pending_live_switch.pop(token, None)


def _mask_client_id(client_id: str | None) -> str | None:
    """Mask client ID for safe display, e.g. 'A12345' -> 'A1***5'."""
    if not client_id or len(client_id) < 3:
        return client_id
    return client_id[:2] + "*" * (len(client_id) - 3) + client_id[-1]


def _validate_credentials_via_login(api_key: str, client_id: str, password: str, totp_secret: str):
    """
    Validate Angel One credentials by generating a TOTP and performing login.
    Returns (access_token, refresh_token) on success, raises on failure.
    """
    from src.execution.broker.angel_one_http_client import AngelOneHttpClient

    try:
        import pyotp
    except ImportError:
        raise RuntimeError("pyotp package is required. Install with: pip install pyotp") from None

    totp_code = pyotp.TOTP(totp_secret).now()

    test_client = AngelOneHttpClient(
        api_key=api_key,
        client_code=client_id,
        password=password,
        totp=totp_code,
    )
    test_client.login()

    access_token = test_client._access_token
    refresh_token = test_client._refresh_token
    if not access_token:
        raise RuntimeError("Login succeeded but no access token was returned")
    return access_token, refresh_token or ""


@router.post("/broker/configure", response_model=BrokerCredentialResponse)
async def configure_broker(
    request: Request,
    creds: BrokerCredentials,
    current_user: dict = Depends(get_current_user),
):
    """
    Step 1 of paper-to-live transition: validate credentials and return a confirmation token.
    The user must call POST /broker/confirm-live with the token to actually switch to live mode.
    This two-step flow prevents accidental live trading activation.
    """
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Gateway not initialized", "connected": False},
        )

    # Validate credentials by attempting Angel One login
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        access_token, refresh_token = await loop.run_in_executor(
            None,
            _validate_credentials_via_login,
            creds.api_key,
            creds.client_id,
            creds.password,
            creds.totp_secret,
        )
    except Exception:
        logger.exception("Broker credential validation failed")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Credential validation failed",
                "connected": False,
            },
        )

    # ── Paper mode: connect with real credentials for market data, but simulated orders ──
    if creds.mode == "paper":
        try:
            from src.execution.broker.angel_one_http_client import AngelOneHttpClient

            gateway.api_key = creds.api_key
            gateway.access_token = access_token
            gateway._client_code = creds.client_id
            gateway._password = creds.password
            gateway._totp_secret = creds.totp_secret
            gateway.paper = True  # Stay in PAPER mode

            gateway._client = AngelOneHttpClient(
                api_key=creds.api_key,
                client_code=creds.client_id,
                password=creds.password,
                access_token=access_token,
                refresh_token=refresh_token,
            )

            gateway._token_acquired_at = time.monotonic()
            gateway._refresh_failures = 0
            gateway._auth_failed = False

            os.environ["ANGEL_ONE_API_KEY"] = creds.api_key
            os.environ["ANGEL_ONE_CLIENT_ID"] = creds.client_id
            os.environ["ANGEL_ONE_PASSWORD"] = creds.password
            os.environ["ANGEL_ONE_TOTP_SECRET"] = creds.totp_secret
            os.environ["ANGEL_ONE_TOKEN"] = access_token
            os.environ["EXEC_PAPER"] = "true"

            request.app.state.safe_mode = False
            request.app.state.broker_last_connected = datetime.now(UTC).isoformat()

            logger.info(
                "Broker credentials validated -- PAPER mode active with real market data (client=%s, actor=%s)",
                creds.client_id,
                current_user.get("user_id", "unknown"),
            )

            # Audit log
            audit_repo = getattr(request.app.state, "audit_repo", None)
            if audit_repo:
                try:
                    audit_repo.append_sync(
                        "broker_configured",
                        current_user.get("user_id", "api"),
                        {"client_id": creds.client_id, "mode": "paper"},
                    )
                except Exception:
                    pass

            # Auto-start Angel One WebSocket market data feed
            auto_started = False
            try:
                existing_mds = getattr(request.app.state, "market_data_service", None)
                if existing_mds is None:
                    yf_feeder = getattr(request.app.state, "yf_feeder", None)
                    if yf_feeder:
                        try:
                            await yf_feeder.stop()
                            request.app.state.yf_feeder = None
                            logger.info("YFinance fallback feeder stopped (switching to Angel One WS)")
                        except Exception:
                            pass

                    bar_cache = getattr(request.app.state, "bar_cache", None)
                    bar_aggregator = getattr(request.app.state, "bar_aggregator", None)
                    if bar_cache and bar_aggregator:
                        from src.market_data.angel_one_ws_connector import AngelOneWsConnector
                        from src.market_data.market_data_service import MarketDataService
                        from src.scanner.dynamic_universe import get_dynamic_universe

                        feed_symbols = get_dynamic_universe().get_tradeable_stocks(count=20)
                        if not feed_symbols:
                            feed_symbols = []

                        ws_connector = AngelOneWsConnector(
                            api_key=creds.api_key,
                            api_secret=creds.client_id,
                            token=access_token,
                            exchange="NSE",
                        )
                        mds = MarketDataService(
                            ws_connector,
                            bar_cache,
                            bar_aggregator,
                            feed_symbols,
                        )
                        mds.start()
                        request.app.state.market_data_service = mds
                        request.app.state.angel_one_marketdata_enabled = True
                        logger.info(
                            "Angel One MarketDataService started in PAPER mode (symbols=%d)",
                            len(feed_symbols),
                        )
            except Exception as mds_err:
                logger.warning("MarketDataService auto-start failed (non-fatal): %s", mds_err)

            # Auto-start autonomous loop in paper mode
            autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
            if autonomous_loop is not None:
                is_running = getattr(autonomous_loop, "_running", False)
                if not is_running:
                    try:
                        await autonomous_loop.start()
                        auto_started = True
                        logger.info("Autonomous loop auto-started in PAPER mode (client=%s)", creds.client_id)
                    except Exception as loop_err:
                        logger.warning("Failed to auto-start autonomous loop: %s", loop_err)

            return {
                "status": "ok",
                "message": f"Broker connected in PAPER mode (client: {creds.client_id}). Real market data, simulated orders.",
                "mode": "paper",
                "connected": True,
                "auto_started": auto_started,
            }

        except Exception:
            logger.exception("Failed to configure gateway in paper mode")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Gateway configuration failed", "connected": False},
            )

    # ── Live mode: two-step confirmation flow ──
    # Cleanup expired in-memory tokens
    now = time.time()
    expired = [k for k, v in _pending_live_switch.items() if now - v["ts"] > _CONFIRM_TOKEN_TTL_SECONDS]
    for k in expired:
        del _pending_live_switch[k]

    # Credentials validated — issue a confirmation token (valid for 5 minutes)
    # Store only encrypted versions of sensitive fields to avoid plaintext in memory
    token = secrets.token_urlsafe(32)
    token_data = {
        "creds_enc": {
            "api_key": encrypt_credential(creds.api_key),
            "client_id": encrypt_credential(creds.client_id),
            "password": encrypt_credential(creds.password),
            "totp_secret": encrypt_credential(creds.totp_secret),
        },
        "access_token": encrypt_credential(access_token),
        "refresh_token": encrypt_credential(refresh_token),
        "ts": now,
        "user_id": current_user.get("user_id", "unknown"),
    }
    _store_pending_token(token, token_data)

    logger.info(
        "Broker credentials validated -- confirmation token issued (client=%s, actor=%s)",
        creds.client_id,
        current_user.get("user_id", "unknown"),
    )

    return {
        "status": "confirm_required",
        "message": f"Credentials validated for client {creds.client_id}. "
        f"Call POST /broker/confirm-live with the confirm_token to switch to LIVE mode. "
        f"Token expires in {_CONFIRM_TOKEN_TTL_SECONDS // 60} minutes.",
        "mode": "paper",
        "connected": False,
        "auto_started": False,
        "confirm_token": token,
    }


@router.post("/broker/confirm-live", response_model=BrokerCredentialResponse)
async def confirm_live_mode(
    request: Request,
    body: LiveModeConfirmRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Step 2 of paper-to-live transition: confirm the switch to live trading.
    Requires a valid confirm_token from the /broker/configure step.
    """
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Gateway not initialized", "connected": False},
        )

    pending = _pop_pending_token(body.confirm_token)
    if not pending:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid or expired confirmation token", "connected": False},
        )
    if time.time() - pending["ts"] > _CONFIRM_TOKEN_TTL_SECONDS:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Confirmation token expired (5 min TTL)", "connected": False},
        )

    # Decrypt the stored encrypted credentials
    creds_enc = pending["creds_enc"]
    creds = BrokerCredentials(
        api_key=decrypt_credential(creds_enc["api_key"]),
        client_id=decrypt_credential(creds_enc["client_id"]),
        password=decrypt_credential(creds_enc["password"]),
        totp_secret=decrypt_credential(creds_enc["totp_secret"]),
    )
    access_token = decrypt_credential(pending["access_token"])
    refresh_token = decrypt_credential(pending["refresh_token"])

    # Reconfigure the live gateway with validated credentials
    try:
        from src.execution.broker.angel_one_http_client import AngelOneHttpClient

        gateway.api_key = creds.api_key
        gateway.access_token = access_token
        gateway._client_code = creds.client_id
        gateway._password = creds.password
        gateway._totp_secret = creds.totp_secret
        gateway.paper = False  # Switch to LIVE mode

        # Create a fresh HTTP client with the validated credentials
        gateway._client = AngelOneHttpClient(
            api_key=creds.api_key,
            client_code=creds.client_id,
            password=creds.password,
            access_token=access_token,
            refresh_token=refresh_token,
        )

        # Reset token refresh state
        gateway._token_acquired_at = time.monotonic()
        gateway._refresh_failures = 0
        gateway._auth_failed = False

        # Store in environment for restart persistence (plaintext — os.environ is
        # already in-process memory; encrypting here just breaks readers on restart
        # that expect plaintext values).
        os.environ["ANGEL_ONE_API_KEY"] = creds.api_key
        os.environ["ANGEL_ONE_CLIENT_ID"] = creds.client_id
        os.environ["ANGEL_ONE_PASSWORD"] = creds.password
        os.environ["ANGEL_ONE_TOTP_SECRET"] = creds.totp_secret
        os.environ["ANGEL_ONE_TOKEN"] = access_token
        os.environ["EXEC_PAPER"] = "false"

        # Clear safe mode since broker is now reachable
        request.app.state.safe_mode = False

        # Store last connected timestamp
        request.app.state.broker_last_connected = datetime.now(UTC).isoformat()

        logger.info(
            "Broker credentials CONFIRMED -- switched to LIVE mode (client=%s, actor=%s)",
            creds.client_id,
            current_user.get("user_id", "unknown"),
        )

        # Audit log
        audit_repo = getattr(request.app.state, "audit_repo", None)
        if audit_repo:
            try:
                audit_repo.append_sync(
                    "broker_configured",
                    current_user.get("user_id", "api"),
                    {"client_id": creds.client_id, "mode": "live"},
                )
            except Exception:
                pass

        # Auto-start Angel One WebSocket market data feed
        try:
            existing_mds = getattr(request.app.state, "market_data_service", None)
            if existing_mds is None:
                # Stop yfinance fallback feeder if running
                yf_feeder = getattr(request.app.state, "yf_feeder", None)
                if yf_feeder:
                    try:
                        await yf_feeder.stop()
                        request.app.state.yf_feeder = None
                        logger.info("YFinance fallback feeder stopped (switching to Angel One WS)")
                    except Exception:
                        pass

                bar_cache = getattr(request.app.state, "bar_cache", None)
                bar_aggregator = getattr(request.app.state, "bar_aggregator", None)
                if bar_cache and bar_aggregator:
                    from src.market_data.angel_one_ws_connector import AngelOneWsConnector
                    from src.market_data.market_data_service import MarketDataService
                    from src.scanner.dynamic_universe import get_dynamic_universe

                    # Discover symbols dynamically
                    feed_symbols = get_dynamic_universe().get_tradeable_stocks(count=20)
                    if not feed_symbols:
                        feed_symbols = []

                    ws_connector = AngelOneWsConnector(
                        api_key=creds.api_key,
                        api_secret=creds.client_id,
                        token=access_token,
                        exchange="NSE",
                    )
                    mds = MarketDataService(
                        ws_connector,
                        bar_cache,
                        bar_aggregator,
                        feed_symbols,
                    )
                    mds.start()
                    request.app.state.market_data_service = mds
                    request.app.state.angel_one_marketdata_enabled = True
                    logger.info(
                        "Angel One MarketDataService started from UI (symbols=%d)",
                        len(feed_symbols),
                    )
        except Exception as mds_err:
            logger.warning("MarketDataService auto-start failed (non-fatal): %s", mds_err)

        # Auto-start autonomous loop if it exists and is not already running
        auto_started = False
        autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
        if autonomous_loop is not None:
            is_running = getattr(autonomous_loop, "_running", False)
            if not is_running:
                try:
                    await autonomous_loop.start()
                    auto_started = True
                    logger.info(
                        "Autonomous loop auto-started after broker connection (client=%s)",
                        creds.client_id,
                    )
                except Exception as loop_err:
                    logger.warning("Failed to auto-start autonomous loop: %s", loop_err)

        return {
            "status": "ok",
            "message": f"Broker configured -- LIVE mode active (client: {creds.client_id})",
            "mode": "live",
            "connected": True,
            "auto_started": auto_started,
        }

    except Exception:
        logger.exception("Failed to reconfigure gateway")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Gateway reconfiguration failed",
                "connected": False,
            },
        )


@router.get("/broker/status", response_model=BrokerStatusResponse)
async def broker_status(request: Request, current_user: dict = Depends(get_current_user)):
    """Return current broker connection status, mode, and health."""
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        return {
            "connected": False,
            "mode": "paper",
            "healthy": False,
            "client_id": None,
            "last_connected": None,
            "message": "Gateway not initialized",
        }

    is_paper = getattr(gateway, "paper", True)
    safe_mode = getattr(request.app.state, "safe_mode", False)

    # Check if Angel One credentials are configured
    client_code = getattr(gateway, "_client_code", None)
    has_creds = bool(getattr(gateway, "api_key", None) and client_code)

    # Check autonomous loop status
    autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
    autonomous_running = False
    tick_count = 0
    open_trades = 0
    if autonomous_loop is not None:
        autonomous_running = getattr(autonomous_loop, "_running", False)
        tick_count = getattr(autonomous_loop, "_tick_count", 0)
        open_trades = len(getattr(autonomous_loop, "_open_trades", {}))

    return {
        "connected": has_creds,
        "mode": "paper" if is_paper else "live",
        "healthy": not safe_mode,
        "safe_mode": safe_mode,
        "has_credentials": has_creds,
        "client_id": _mask_client_id(client_code) if has_creds else None,
        "last_connected": getattr(request.app.state, "broker_last_connected", None),
        "autonomous_running": autonomous_running,
        "tick_count": tick_count,
        "open_trades": open_trades,
    }


@router.post("/broker/disconnect", response_model=BrokerDisconnectResponse)
async def disconnect_broker(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """
    Disconnect from Angel One broker and switch back to paper trading mode.
    Clears credentials from the gateway and environment. Does NOT affect open positions.
    """
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Gateway not initialized"},
        )

    was_live = not getattr(gateway, "paper", True)
    client_id = getattr(gateway, "_client_code", None)

    # Stop autonomous loop if running
    autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
    if autonomous_loop is not None and getattr(autonomous_loop, "_running", False):
        try:
            await autonomous_loop.stop()
            logger.info("Autonomous loop stopped during broker disconnect")
        except Exception as e:
            logger.warning("Failed to stop autonomous loop during disconnect: %s", e)

    # Disconnect the gateway client (handle both sync and async disconnect methods)
    try:
        if asyncio.iscoroutinefunction(gateway.disconnect):
            await gateway.disconnect()
        else:
            gateway.disconnect()
    except Exception as e:
        logger.warning("Gateway disconnect error (non-fatal): %s", e)

    # Reset gateway to paper mode
    gateway.paper = True
    gateway.api_key = ""
    gateway.access_token = ""
    gateway._client_code = None
    gateway._password = None
    gateway._totp_secret = None
    gateway._client = None
    gateway._token_acquired_at = 0.0
    gateway._refresh_failures = 0
    gateway._auth_failed = False

    # Clear environment variables
    for env_key in [
        "ANGEL_ONE_API_KEY",
        "ANGEL_ONE_CLIENT_ID",
        "ANGEL_ONE_PASSWORD",
        "ANGEL_ONE_TOTP_SECRET",
        "ANGEL_ONE_TOKEN",
    ]:
        os.environ.pop(env_key, None)
    os.environ["EXEC_PAPER"] = "true"

    logger.info(
        "Broker disconnected -- switched to PAPER mode (was_live=%s, client=%s, actor=%s)",
        was_live,
        client_id,
        current_user.get("user_id", "unknown"),
    )

    # Audit log
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            audit_repo.append_sync(
                "broker_disconnected",
                current_user.get("user_id", "api"),
                {"client_id": client_id, "was_live": was_live},
            )
        except Exception:
            pass

    return {
        "status": "ok",
        "message": "Broker disconnected -- switched to PAPER mode",
        "mode": "paper",
        "connected": False,
    }


@router.post("/broker/validate", response_model=BrokerValidateResponse)
async def validate_broker_credentials(creds: BrokerCredentials, current_user: dict = Depends(get_current_user)):
    """Test Angel One credentials without saving. Returns success/failure."""
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _validate_credentials_via_login,
            creds.api_key,
            creds.client_id,
            creds.password,
            creds.totp_secret,
        )
        return {"valid": True, "message": "Credentials validated successfully"}
    except Exception:
        logger.exception("Broker credential validation failed")
        return JSONResponse(
            status_code=400,
            content={"valid": False, "message": "Credential validation failed"},
        )

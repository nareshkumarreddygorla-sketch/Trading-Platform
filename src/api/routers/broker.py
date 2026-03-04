"""
Broker credential management: configure, validate, disconnect, and check Angel One connection.
Enables zero-intervention flow: user enters creds -> system validates -> auto-switches to live.
Credentials are encrypted at rest using Fernet (AES-128-CBC) derived from JWT_SECRET via PBKDF2.
"""
import base64
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

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


def _derive_fernet_key() -> Optional[bytes]:
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
        except Exception:
            # Value may be stored unencrypted (legacy) — return as-is
            return value
    return value

router = APIRouter()


class BrokerCredentials(BaseModel):
    api_key: str = Field(..., min_length=1, description="Angel One API key")
    client_id: str = Field(..., min_length=1, description="Angel One client ID")
    password: str = Field(..., min_length=1, description="Angel One password")
    totp_secret: str = Field(..., min_length=1, description="Base32 TOTP secret for auto-login")


class BrokerCredentialResponse(BaseModel):
    status: str
    message: str
    mode: Optional[str] = None
    connected: bool = False
    auto_started: bool = False


def _mask_client_id(client_id: Optional[str]) -> Optional[str]:
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
        raise RuntimeError("pyotp package is required. Install with: pip install pyotp")

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
    Configure Angel One broker credentials and switch from paper to live mode.
    Validates credentials by attempting authentication, then reconfigures the gateway.
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
    except Exception as e:
        logger.warning("Broker credential validation failed: %s", e)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Credential validation failed: {str(e)}",
                "connected": False,
            },
        )

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

        # Store in environment for restart persistence (encrypted at rest)
        os.environ["ANGEL_ONE_API_KEY"] = encrypt_credential(creds.api_key)
        os.environ["ANGEL_ONE_CLIENT_ID"] = encrypt_credential(creds.client_id)
        os.environ["ANGEL_ONE_PASSWORD"] = encrypt_credential(creds.password)
        os.environ["ANGEL_ONE_TOTP_SECRET"] = encrypt_credential(creds.totp_secret)
        os.environ["ANGEL_ONE_TOKEN"] = encrypt_credential(access_token)
        os.environ["EXEC_PAPER"] = "false"

        # Clear safe mode since broker is now reachable
        request.app.state.safe_mode = False

        # Store last connected timestamp
        request.app.state.broker_last_connected = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Broker credentials configured -- switched to LIVE mode (client=%s, actor=%s)",
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

    except Exception as e:
        logger.error("Failed to reconfigure gateway: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Gateway reconfiguration failed: {str(e)}",
                "connected": False,
            },
        )


@router.get("/broker/status")
async def broker_status(request: Request):
    """Return current broker connection status, mode, and health."""
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        return {
            "connected": False, "mode": "paper", "healthy": False,
            "client_id": None, "last_connected": None,
            "message": "Gateway not initialized",
        }

    is_paper = getattr(gateway, "paper", True)
    safe_mode = getattr(request.app.state, "safe_mode", False)

    # Check if Angel One credentials are configured
    client_code = getattr(gateway, "_client_code", None)
    has_creds = bool(
        getattr(gateway, "api_key", None)
        and client_code
    )

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
        "connected": has_creds and not is_paper,
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


@router.post("/broker/disconnect")
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

    # Disconnect the gateway client
    try:
        await gateway.disconnect()
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
        "ANGEL_ONE_API_KEY", "ANGEL_ONE_CLIENT_ID", "ANGEL_ONE_PASSWORD",
        "ANGEL_ONE_TOTP_SECRET", "ANGEL_ONE_TOKEN",
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


@router.post("/broker/validate")
async def validate_broker_credentials(creds: BrokerCredentials):
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
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"valid": False, "message": str(e)},
        )

"""
Broker credential management: configure, validate, and check Angel One connection.
Enables zero-intervention flow: user enters creds → system validates → auto-switches to live.
"""
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


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
        from src.execution.broker.angel_one_http_client import AngelOneHttpClient

        test_client = AngelOneHttpClient(
            api_key=creds.api_key,
            api_secret="",  # not needed for TOTP auth
            access_token="",
        )
        # Attempt TOTP-based authentication
        auth_result = test_client.authenticate_totp(
            client_code=creds.client_id,
            password=creds.password,
            totp_secret=creds.totp_secret,
        )
        if not auth_result or not auth_result.get("jwtToken"):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Authentication failed — check credentials and TOTP secret",
                    "connected": False,
                },
            )

        access_token = auth_result["jwtToken"]
        refresh_token = auth_result.get("refreshToken", "")

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
        gateway._client = None  # Force new client creation
        gateway.api_key = creds.api_key
        gateway._access_token = access_token
        gateway._refresh_token = refresh_token
        gateway._client_code = creds.client_id
        gateway._password = creds.password
        gateway._totp_secret = creds.totp_secret
        gateway.paper = False  # Switch to LIVE mode

        # Store in environment for restart persistence
        os.environ["ANGEL_ONE_API_KEY"] = creds.api_key
        os.environ["ANGEL_ONE_CLIENT_ID"] = creds.client_id
        os.environ["ANGEL_ONE_PASSWORD"] = creds.password
        os.environ["ANGEL_ONE_TOTP_SECRET"] = creds.totp_secret
        os.environ["ANGEL_ONE_TOKEN"] = access_token
        os.environ["EXEC_PAPER"] = "false"

        # Clear safe mode since broker is now reachable
        request.app.state.safe_mode = False

        logger.info(
            "Broker credentials configured — switched to LIVE mode (client=%s, actor=%s)",
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

        return {
            "status": "ok",
            "message": f"Broker configured — LIVE mode active (client: {creds.client_id})",
            "mode": "live",
            "connected": True,
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
        return {"connected": False, "mode": "paper", "healthy": False, "message": "Gateway not initialized"}

    is_paper = getattr(gateway, "paper", True)
    safe_mode = getattr(request.app.state, "safe_mode", False)

    # Check if Angel One credentials are configured
    has_creds = bool(
        getattr(gateway, "api_key", None)
        and getattr(gateway, "_client_code", None)
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
        "autonomous_running": autonomous_running,
        "tick_count": tick_count,
        "open_trades": open_trades,
    }


@router.post("/broker/validate")
async def validate_broker_credentials(creds: BrokerCredentials):
    """Test Angel One credentials without saving. Returns success/failure."""
    try:
        from src.execution.broker.angel_one_http_client import AngelOneHttpClient

        test_client = AngelOneHttpClient(
            api_key=creds.api_key,
            api_secret="",
            access_token="",
        )
        auth_result = test_client.authenticate_totp(
            client_code=creds.client_id,
            password=creds.password,
            totp_secret=creds.totp_secret,
        )
        if auth_result and auth_result.get("jwtToken"):
            return {"valid": True, "message": "Credentials validated successfully"}
        return JSONResponse(
            status_code=400,
            content={"valid": False, "message": "Authentication failed"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"valid": False, "message": str(e)},
        )

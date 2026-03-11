"""
Angel One SmartAPI REST client: session lifecycle, place/cancel/status, positions.
Sync HTTP with retries (max 3, exponential backoff), timeouts, session refresh on expiry.
Base URL: https://apiconnect.angelone.in
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://apiconnect.angelone.in"
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # 1s, 2s, 4s

# Error codes that indicate token expiry (trigger refresh)
TOKEN_EXPIRED_CODES = {"AG8001", "AG8002", "AB8051"}


class BrokerClientError(Exception):
    """Broker API error (response status=false or HTTP error)."""

    def __init__(self, message: str, errorcode: str = "", response: requests.Response | None = None):
        self.message = message
        self.errorcode = errorcode
        self.response = response
        super().__init__(message)


class AngelOneHttpClient:
    """
    Sync HTTP client for Angel One SmartAPI.
    Maintains JWT and refresh token; refreshes on expiry (AG8002 etc.).
    """

    def __init__(
        self,
        api_key: str,
        *,
        client_code: str | None = None,
        password: str | None = None,
        totp: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        local_ip: str = "127.0.0.1",
        public_ip: str = "127.0.0.1",
        mac_address: str = "00:00:00:00:00:00",
    ):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp = totp
        self._access_token = access_token
        self._refresh_token = refresh_token
        self.timeout = timeout
        self.local_ip = local_ip
        self.public_ip = public_ip
        self.mac_address = mac_address
        self._session = requests.Session()

    def _headers(self, include_auth: bool = True) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": self.local_ip,
            "X-ClientPublicIP": self.public_ip,
            "X-MACAddress": self.mac_address,
            "X-PrivateKey": self.api_key,
        }
        if include_auth and self._access_token:
            h["Authorization"] = f"Bearer {self._access_token}"
        return h

    def _parse_response(self, r: requests.Response) -> dict[str, Any]:
        try:
            data = r.json()
        except Exception as e:
            raise BrokerClientError(f"Invalid JSON: {e}", response=r) from e
        if r.status_code >= 400:
            msg = data.get("message", r.text) or r.reason
            code = data.get("errorcode", "")
            raise BrokerClientError(msg, errorcode=code, response=r)
        if isinstance(data.get("status"), bool) and not data.get("status"):
            msg = data.get("message", "Unknown error")
            code = data.get("errorcode", "")
            raise BrokerClientError(msg, errorcode=code, response=r)
        return data

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        retry_on_expiry: bool = True,
    ) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                r = self._session.request(
                    method,
                    url,
                    json=json,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                data = self._parse_response(r)
                return data
            except BrokerClientError as e:
                last_error = e
                if e.errorcode in TOKEN_EXPIRED_CODES and retry_on_expiry and self._refresh_token:
                    try:
                        self.refresh_session()
                        retry_on_expiry = False
                        continue
                    except Exception as ref:
                        logger.warning("Session refresh failed: %s", ref)
                        raise
                if attempt + 1 < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2**attempt)
                    time.sleep(backoff)
                    continue
                raise
            except requests.RequestException as e:
                last_error = e
                if attempt + 1 < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2**attempt)
                    time.sleep(backoff)
                    continue
                raise BrokerClientError(str(e), response=getattr(e, "response", None)) from e
        raise last_error or BrokerClientError("Request failed")

    def login(self) -> None:
        """Login with client_code, password, totp. Sets JWT and refresh token."""
        if not all([self.client_code, self.password, self.totp]):
            raise BrokerClientError("client_code, password, totp required for login")
        path = "/rest/auth/angelbroking/user/v1/loginByPassword"
        payload = {
            "clientcode": self.client_code,
            "password": self.password,
            "totp": self.totp,
            "state": "live",
        }
        r = self._session.post(
            f"{BASE_URL}{path}",
            json=payload,
            headers=self._headers(include_auth=False),
            timeout=self.timeout,
        )
        data = self._parse_response(r)
        payload_data = data.get("data") or {}
        self._access_token = payload_data.get("jwtToken") or payload_data.get("jwt")
        self._refresh_token = payload_data.get("refreshToken") or payload_data.get("refresh_token")
        if not self._access_token:
            raise BrokerClientError("Login response missing jwtToken")

    def refresh_session(self) -> None:
        """Refresh JWT using refresh_token."""
        if not self._refresh_token:
            raise BrokerClientError("No refresh_token available")
        path = "/rest/auth/angelbroking/jwt/v1/generateTokens"
        payload = {"refreshToken": self._refresh_token}
        r = self._session.post(
            f"{BASE_URL}{path}",
            json=payload,
            headers=self._headers(include_auth=False),
            timeout=self.timeout,
        )
        data = self._parse_response(r)
        payload_data = data.get("data") or {}
        self._access_token = payload_data.get("jwtToken") or payload_data.get("jwt")
        new_refresh = payload_data.get("refreshToken") or payload_data.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
        if not self._access_token:
            raise BrokerClientError("Refresh response missing jwtToken")

    def ensure_session(self) -> None:
        """Ensure we have a valid session: login or refresh."""
        if self._access_token:
            return
        if self._refresh_token:
            self.refresh_session()
            return
        if self.client_code and self.password and self.totp:
            self.login()
            return
        raise BrokerClientError("No session: provide access_token+refresh_token or client_code+password+totp")

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Place order. Params: exchange, tradingsymbol, quantity, transactiontype, ordertype, variety, producttype, etc."""
        self.ensure_session()
        path = "/rest/secure/angelbroking/order/v1/placeOrder"
        # Ensure required fields
        params.setdefault("variety", "NORMAL")
        params.setdefault("producttype", "INTRADAY")
        params.setdefault("duration", "DAY")
        params.setdefault("scripconsent", "yes")
        data = self._request("POST", path, json=params)
        return data.get("data") or data

    def cancel_order(self, variety: str, order_id: str) -> dict[str, Any]:
        """Cancel by broker order id. variety typically NORMAL."""
        self.ensure_session()
        path = "/rest/secure/angelbroking/order/v1/cancelOrder"
        data = self._request("POST", path, json={"variety": variety, "orderid": order_id})
        return data.get("data") or data

    def get_order_details(self, uniqueorderid: str) -> dict[str, Any]:
        """Get single order by uniqueorderid (UUID from place/cancel response)."""
        self.ensure_session()
        path = f"/rest/secure/angelbroking/order/v1/details/{uniqueorderid}"
        data = self._request("GET", path)
        return data.get("data") or data

    def get_order_book(self) -> list[dict[str, Any]]:
        """Get order book (list of orders)."""
        self.ensure_session()
        path = "/rest/secure/angelbroking/order/v1/getOrderBook"
        data = self._request("GET", path)
        raw = data.get("data")
        if raw is None:
            return []
        return raw if isinstance(raw, list) else [raw]

    def get_position(self) -> list[dict[str, Any]]:
        """Get positions (day positions)."""
        self.ensure_session()
        path = "/rest/secure/angelbroking/order/v1/getPosition"
        data = self._request("GET", path)
        raw = data.get("data")
        if raw is None:
            return []
        return raw if isinstance(raw, list) else [raw]

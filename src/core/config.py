"""Central configuration (env + YAML). Pydantic settings for validation."""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_config_logger = logging.getLogger(__name__)

_KNOWN_DEFAULT_PASSWORDS = frozenset({
    "Admin@Trade2026!",
    "admin",
    "password",
    "changeme",
    "secret",
})


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _parse_symbols(v: object) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []  # empty = discover dynamically at runtime


class AngelOneFeedConfig(BaseSettings):
    """Angel One live market data feed. Env: ANGEL_ONE_MARKETDATA_ENABLED, ANGEL_ONE_SYMBOLS, ANGEL_ONE_EXCHANGE."""

    model_config = SettingsConfigDict(env_prefix="ANGEL_ONE_")
    marketdata_enabled: bool = False
    symbols: List[str] = Field(default_factory=list)  # empty = discover dynamically
    exchange: str = "NSE"

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: object) -> List[str]:
        return _parse_symbols(v)


class MarketDataConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MD_")
    redis_url: str = "redis://localhost:6379/0"
    kafka_brokers: str = "localhost:9092"
    kafka_ticks_topic: str = "market.ticks"
    kafka_bars_topic: str = "market.bars"
    timescale_url: str = "postgresql://localhost:5432/tsdb"
    marketdata_reconnect_backoff_seconds: int = 5


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")
    max_position_pct: float = 5.0
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 10
    var_confidence: float = 0.95
    circuit_breaker_drawdown_pct: float = 5.0
    max_sector_concentration_pct: float = 25.0
    var_limit_pct: float = 5.0
    max_consecutive_losses: int = 5
    max_per_symbol_pct: float = 10.0


class ExecutionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXEC_")
    paper: bool = True  # Single toggle: True = paper trading, False = LIVE orders
    angel_one_api_key: Optional[str] = None
    angel_one_api_secret: Optional[str] = None
    angel_one_token: Optional[str] = None
    angel_one_refresh_token: Optional[str] = None
    angel_one_client_code: Optional[str] = None
    angel_one_password: Optional[str] = None
    angel_one_totp: Optional[str] = None
    angel_one_totp_secret: Optional[str] = None  # Base32 TOTP secret for auto token refresh
    angel_one_request_timeout: float = 15.0


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class SecurityConfig(BaseSettings):
    """Security-related settings. All secrets MUST come from environment variables."""
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    cors_origins: str = ""  # Comma-separated origins; set via CORS_ORIGINS or SECURITY_CORS_ORIGINS
    hsts_max_age: int = 31536000  # 1 year
    csp_report_uri: str = ""  # CSP violation report endpoint
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "Lax"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )
    environment: str = "development"
    database_url: Optional[str] = None
    jwt_secret: str = ""
    auth_password: str = ""
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    angel_one_feed: AngelOneFeedConfig = Field(default_factory=AngelOneFeedConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @model_validator(mode="after")
    def _validate_production_database(self) -> "Settings":
        """In production, DATABASE_URL must be explicitly set (no SQLite fallback)."""
        db_url = self.database_url or os.environ.get("DATABASE_URL")
        if self.environment.lower() == "production" and not db_url:
            raise ValueError(
                "DATABASE_URL is required when environment is 'production'. "
                "Set DATABASE_URL to a PostgreSQL connection string, "
                "or set ENVIRONMENT=development to use SQLite."
            )
        return self

    @model_validator(mode="after")
    def _validate_jwt_secret(self) -> "Settings":
        """Validate JWT_SECRET: required and >= 32 chars in production, warned in development."""
        secret = self.jwt_secret or os.environ.get("JWT_SECRET", "")
        env = self.environment.lower()
        if env == "production":
            if not secret:
                raise ValueError(
                    "JWT_SECRET is REQUIRED in production. "
                    "Generate a strong secret with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if len(secret) < 32:
                raise ValueError(
                    "JWT_SECRET must be at least 32 characters in production. "
                    "Generate a strong secret with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
        elif not secret:
            _config_logger.warning(
                "JWT_SECRET is not set. A random secret will be generated at runtime. "
                "Tokens will NOT survive restarts. Set JWT_SECRET for persistent sessions."
            )
        elif len(secret) < 16:
            _config_logger.warning(
                "JWT_SECRET is shorter than 16 characters. Use a stronger secret."
            )
        return self

    @model_validator(mode="after")
    def _validate_production_password(self) -> "Settings":
        """In production, reject known default passwords."""
        if self.environment.lower() == "production":
            password = self.auth_password or os.environ.get("AUTH_PASSWORD", "")
            if password in _KNOWN_DEFAULT_PASSWORDS:
                raise ValueError(
                    f"AUTH_PASSWORD must not be a default password in production. "
                    f"Change AUTH_PASSWORD to a strong, unique value."
                )
        return self

    @model_validator(mode="after")
    def _warn_live_trading(self) -> "Settings":
        """In production with paper=False, log a CRITICAL warning about live trading."""
        if self.environment.lower() == "production" and not self.execution.paper:
            _config_logger.critical(
                "LIVE TRADING ENABLED — paper=False in production environment. "
                "Real broker orders WILL be placed."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache():
    """Clear cached settings (for testing)."""
    get_settings.cache_clear()

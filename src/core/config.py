"""Central configuration (env + YAML). Pydantic settings for validation."""
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _parse_symbols(v: object) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return ["RELIANCE", "TCS", "INFY"]


class AngelOneFeedConfig(BaseSettings):
    """Angel One live market data feed. Env: ANGEL_ONE_MARKETDATA_ENABLED, ANGEL_ONE_SYMBOLS, ANGEL_ONE_EXCHANGE."""

    model_config = SettingsConfigDict(env_prefix="ANGEL_ONE_")
    marketdata_enabled: bool = False
    symbols: List[str] = Field(default_factory=lambda: ["RELIANCE", "TCS", "INFY"])
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
    paper: bool = True
    paper_execution_mode: bool = True  # When True, gateway never places real broker orders
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )
    environment: str = "development"
    # Top-level so PAPER_EXECUTION_MODE=true works (otherwise use EXEC_PAPER_EXECUTION_MODE)
    paper_execution_mode: Optional[bool] = None
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    angel_one_feed: AngelOneFeedConfig = Field(default_factory=AngelOneFeedConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def get_settings() -> Settings:
    return Settings()

"""
Market data persistence models: OHLCV bars and trade outcomes.
Used for: historical data storage, model training, performance attribution.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from src.persistence.models import Base


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OHLCVBarModel(Base):
    """
    Historical OHLCV bar storage. One row per (symbol, exchange, interval, timestamp).
    Designed for TimescaleDB hypertable partitioning on timestamp.
    """

    __tablename__ = "ohlcv_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    exchange = Column(String(16), nullable=False, default="NSE")
    interval = Column(String(8), nullable=False)  # 1m, 5m, 15m, 1h, 1d
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "interval", "timestamp", name="uq_ohlcv_symbol_exchange_interval_ts"),
        Index("ix_ohlcv_symbol_interval_ts", "symbol", "interval", "timestamp"),
        Index("ix_ohlcv_timestamp", "timestamp"),
    )


class TradeOutcomeModel(Base):
    """
    Closed trade outcomes for self-learning and performance attribution.
    Records the full context at trade entry for feature importance analysis.
    """

    __tablename__ = "trade_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_key = Column(String(256), nullable=False, index=True)  # symbol:strategy_id
    symbol = Column(String(32), nullable=False, index=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)  # percentage return
    holding_bars = Column(Integer, nullable=False, default=0)
    strategy_id = Column(String(128), nullable=True)
    model_id = Column(String(128), nullable=True)  # which model generated the signal
    signal_confidence = Column(Float, nullable=True)
    signal_score = Column(Float, nullable=True)
    regime_at_entry = Column(String(32), nullable=True)  # market regime when trade opened
    features_at_entry = Column(Text, nullable=True)  # JSON: feature dict at entry
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    exit_reason = Column(String(32), nullable=True)  # stop_loss, take_profit, trailing, signal, eod
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)

    __table_args__ = (
        Index("ix_trade_outcomes_symbol_time", "symbol", "exit_time"),
        Index("ix_trade_outcomes_strategy", "strategy_id", "exit_time"),
        Index("ix_trade_outcomes_model", "model_id", "exit_time"),
    )


class SimulationResultModel(Base):
    """
    Nightly simulation results: strategy permutation performance metrics.
    Used by the Holly-style strategy selector.
    """

    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime(timezone=True), nullable=False, index=True)
    strategy_id = Column(String(128), nullable=False)
    strategy_params = Column(Text, nullable=True)  # JSON: parameter set
    symbols = Column(Text, nullable=True)  # JSON: list of symbols tested
    interval = Column(String(8), nullable=False, default="15m")
    lookback_days = Column(Integer, nullable=False, default=30)
    # Performance metrics
    total_return_pct = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=False, default=0.0)
    sortino_ratio = Column(Float, nullable=False, default=0.0)
    max_drawdown_pct = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    # Selection
    rank = Column(Integer, nullable=True)  # rank within this run
    selected = Column(Integer, nullable=False, default=0)  # 1 if selected for next day
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)

    __table_args__ = (Index("ix_sim_results_run_date", "run_date", "rank"),)

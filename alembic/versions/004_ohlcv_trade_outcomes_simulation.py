"""OHLCV bars, trade outcomes, simulation results tables.

Revision ID: 004
Revises: 003
Create Date: 2026-03-09

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # OHLCV bar storage (designed for TimescaleDB hypertable)
    op.create_table(
        "ohlcv_bars",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("exchange", sa.String(16), nullable=False, server_default="NSE"),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "exchange", "interval", "timestamp", name="uq_ohlcv_symbol_exchange_interval_ts"),
    )
    op.create_index("ix_ohlcv_symbol_interval_ts", "ohlcv_bars", ["symbol", "interval", "timestamp"])
    op.create_index("ix_ohlcv_timestamp", "ohlcv_bars", ["timestamp"])

    # Trade outcomes for self-learning and attribution
    op.create_table(
        "trade_outcomes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_key", sa.String(256), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("exchange", sa.String(16), nullable=False, server_default="NSE"),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=False),
        sa.Column("pnl_pct", sa.Float(), nullable=False),
        sa.Column("holding_bars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("strategy_id", sa.String(128), nullable=True),
        sa.Column("model_id", sa.String(128), nullable=True),
        sa.Column("signal_confidence", sa.Float(), nullable=True),
        sa.Column("signal_score", sa.Float(), nullable=True),
        sa.Column("regime_at_entry", sa.String(32), nullable=True),
        sa.Column("features_at_entry", sa.Text(), nullable=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("exit_reason", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trade_outcomes_trade_key", "trade_outcomes", ["trade_key"])
    op.create_index("ix_trade_outcomes_symbol", "trade_outcomes", ["symbol"])
    op.create_index("ix_trade_outcomes_symbol_time", "trade_outcomes", ["symbol", "exit_time"])
    op.create_index("ix_trade_outcomes_strategy", "trade_outcomes", ["strategy_id", "exit_time"])
    op.create_index("ix_trade_outcomes_model", "trade_outcomes", ["model_id", "exit_time"])

    # Nightly simulation results
    op.create_table(
        "simulation_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("strategy_id", sa.String(128), nullable=False),
        sa.Column("strategy_params", sa.Text(), nullable=True),
        sa.Column("symbols", sa.Text(), nullable=True),
        sa.Column("interval", sa.String(8), nullable=False, server_default="15m"),
        sa.Column("lookback_days", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("total_return_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sharpe_ratio", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sortino_ratio", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_drawdown_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("profit_factor", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("selected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sim_results_run_date", "simulation_results", ["run_date", "rank"])


def downgrade() -> None:
    op.drop_index("ix_sim_results_run_date", "simulation_results")
    op.drop_table("simulation_results")

    op.drop_index("ix_trade_outcomes_model", "trade_outcomes")
    op.drop_index("ix_trade_outcomes_strategy", "trade_outcomes")
    op.drop_index("ix_trade_outcomes_symbol_time", "trade_outcomes")
    op.drop_index("ix_trade_outcomes_symbol", "trade_outcomes")
    op.drop_index("ix_trade_outcomes_trade_key", "trade_outcomes")
    op.drop_table("trade_outcomes")

    op.drop_index("ix_ohlcv_timestamp", "ohlcv_bars")
    op.drop_index("ix_ohlcv_symbol_interval_ts", "ohlcv_bars")
    op.drop_table("ohlcv_bars")

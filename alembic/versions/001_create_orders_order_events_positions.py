"""Create orders, order_events, positions tables.

Revision ID: 001
Revises:
Create Date: 2025-02-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("order_id", sa.String(64), nullable=False),
        sa.Column("broker_order_id", sa.String(128), nullable=True),
        sa.Column("idempotency_key", sa.String(256), nullable=True),
        sa.Column("strategy_id", sa.String(128), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("exchange", sa.String(16), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("order_type", sa.String(16), nullable=False),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("filled_qty", sa.Float(), nullable=False),
        sa.Column("avg_price", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("order_id", name="uq_orders_order_id"),
    )
    op.create_index("ix_orders_order_id", "orders", ["order_id"], unique=True)
    op.create_index("ix_orders_broker_order_id", "orders", ["broker_order_id"], unique=False)
    op.create_index("ix_orders_status", "orders", ["status"], unique=False)
    op.create_index("ix_orders_symbol", "orders", ["symbol"], unique=False)

    op.create_table(
        "order_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("order_id_ref", sa.String(64), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("from_status", sa.String(32), nullable=True),
        sa.Column("to_status", sa.String(32), nullable=False),
        sa.Column("filled_qty", sa.Float(), nullable=False),
        sa.Column("avg_price", sa.Float(), nullable=True),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["order_id_ref"], ["orders.order_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_order_events_order_id_ref", "order_events", ["order_id_ref"], unique=False)

    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("exchange", sa.String(16), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("avg_price", sa.Float(), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=False),
        sa.Column("strategy_id", sa.String(128), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "exchange", "side", name="uq_position_symbol_exchange_side"),
    )
    op.create_index("ix_positions_symbol", "positions", ["symbol"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_positions_symbol", table_name="positions")
    op.drop_table("positions")
    op.drop_index("ix_order_events_order_id_ref", table_name="order_events")
    op.drop_table("order_events")
    op.drop_index("ix_orders_symbol", table_name="orders")
    op.drop_index("ix_orders_status", table_name="orders")
    op.drop_index("ix_orders_broker_order_id", table_name="orders")
    op.drop_index("ix_orders_order_id", table_name="orders")
    op.drop_table("orders")

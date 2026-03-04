"""Institutional: unique idempotency_key, order/position check constraints, position version (OCC).

Revision ID: 002
Revises: 001
Create Date: 2025-02-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Orders: unique on idempotency_key (PostgreSQL allows multiple NULLs)
    op.create_unique_constraint("uq_orders_idempotency_key", "orders", ["idempotency_key"])
    op.create_check_constraint("ck_orders_quantity_non_negative", "orders", "quantity >= 0")
    op.create_check_constraint("ck_orders_filled_qty_non_negative", "orders", "filled_qty >= 0")
    op.create_check_constraint(
        "ck_orders_status_valid",
        "orders",
        "status IN ('NEW', 'ACK', 'PARTIAL', 'FILLED', 'REJECTED', 'CANCELLED')",
    )

    # Positions: version column for OCC, check quantity >= 0
    op.add_column("positions", sa.Column("version", sa.Integer(), nullable=False, server_default="0"))
    op.create_check_constraint("ck_positions_quantity_non_negative", "positions", "quantity >= 0")


def downgrade() -> None:
    op.drop_constraint("ck_positions_quantity_non_negative", "positions", type_="check")
    op.drop_column("positions", "version")
    op.drop_constraint("ck_orders_status_valid", "orders", type_="check")
    op.drop_constraint("ck_orders_filled_qty_non_negative", "orders", type_="check")
    op.drop_constraint("ck_orders_quantity_non_negative", "orders", type_="check")
    op.drop_constraint("uq_orders_idempotency_key", "orders", type_="unique")

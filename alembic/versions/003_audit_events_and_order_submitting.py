"""Audit events table (immutable). Order status SUBMITTING for write-ahead.

Revision ID: 003
Revises: 002
Create Date: 2025-02-25

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Immutable audit log
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(256), nullable=False, server_default="system"),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.Column("tenant_id", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_tenant_id", "audit_events", ["tenant_id"])

    # Write-ahead: allow SUBMITTING (order persisted before broker call)
    op.drop_constraint("ck_orders_status_valid", "orders", type_="check")
    op.create_check_constraint(
        "ck_orders_status_valid",
        "orders",
        "status IN ('SUBMITTING', 'NEW', 'ACK', 'PARTIAL', 'FILLED', 'REJECTED', 'CANCELLED')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_orders_status_valid", "orders", type_="check")
    op.create_check_constraint(
        "ck_orders_status_valid",
        "orders",
        "status IN ('NEW', 'ACK', 'PARTIAL', 'FILLED', 'REJECTED', 'CANCELLED')",
    )
    op.drop_index("ix_audit_events_tenant_id", "audit_events")
    op.drop_index("ix_audit_events_event_type", "audit_events")
    op.drop_table("audit_events")

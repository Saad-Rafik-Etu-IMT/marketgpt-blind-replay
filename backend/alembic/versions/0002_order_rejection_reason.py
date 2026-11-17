"""Add orders.rejection_reason for fill-time rejections

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-17 12:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.add_column(sa.Column('rejection_reason', sa.String(length=32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.drop_column('rejection_reason')

"""Add confidence column to diagnosis_predictions (ORM alignment).

Revision ID: 003_diagnosis_confidence
Revises: 002_auth_sessions
Create Date: 2026-10-23
"""

from alembic import op
import sqlalchemy as sa

revision = "003_diagnosis_confidence"
down_revision = "002_auth_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "diagnosis_predictions",
        sa.Column("confidence", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("diagnosis_predictions", "confidence")

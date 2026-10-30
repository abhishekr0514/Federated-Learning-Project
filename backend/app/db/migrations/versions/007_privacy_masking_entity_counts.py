"""Create privacy_masking_entity_counts table (aggregate PII counts per masking run).

Revision ID: 007_mask_entity_counts
Revises: 006_prediction_class_index
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "007_mask_entity_counts"
down_revision = "006_prediction_class_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "privacy_masking_entity_counts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "masking_result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("privacy_masking_results.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity_category", sa.String(length=50), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint(
            "masking_result_id",
            "entity_category",
            name="uq_masking_entity_category",
        ),
    )
    op.create_index(
        op.f("ix_privacy_masking_entity_counts_masking_result_id"),
        "privacy_masking_entity_counts",
        ["masking_result_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_privacy_masking_entity_counts_masking_result_id"),
        table_name="privacy_masking_entity_counts",
    )
    op.drop_table("privacy_masking_entity_counts")

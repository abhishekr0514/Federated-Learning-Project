"""Patient clinician questions table.

Revision ID: 005_patient_clinician_questions
Revises: 004_alerts_dq
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "005_patient_questions"
down_revision = "004_alerts_dq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("patient_clinician_questions"):
        return

    op.create_table(
        "patient_clinician_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "hospital_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hospitals.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "author_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "assigned_clinician_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("clinician_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="submitted"),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_patient_clinician_questions_patient_id",
        "patient_clinician_questions",
        ["patient_id"],
    )
    op.create_index(
        "ix_patient_clinician_questions_hospital_id",
        "patient_clinician_questions",
        ["hospital_id"],
    )
    op.create_index(
        op.f("ix_patient_clinician_questions_author_user_id"),
        "patient_clinician_questions",
        ["author_user_id"],
    )
    op.create_index(
        op.f("ix_patient_clinician_questions_assigned_clinician_id"),
        "patient_clinician_questions",
        ["assigned_clinician_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_patient_clinician_questions_assigned_clinician_id"),
        table_name="patient_clinician_questions",
    )
    op.drop_index(
        op.f("ix_patient_clinician_questions_author_user_id"),
        table_name="patient_clinician_questions",
    )
    op.drop_index("ix_patient_clinician_questions_hospital_id", table_name="patient_clinician_questions")
    op.drop_index("ix_patient_clinician_questions_patient_id", table_name="patient_clinician_questions")
    op.drop_table("patient_clinician_questions")

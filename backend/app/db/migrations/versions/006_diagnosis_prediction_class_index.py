"""Add predicted_class_index to diagnosis_predictions.

Revision ID: 006_prediction_class_index
Revises: 005_patient_questions
"""

from alembic import op
import sqlalchemy as sa

revision = "006_prediction_class_index"
down_revision = "005_patient_questions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "diagnosis_predictions",
        sa.Column("predicted_class_index", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("diagnosis_predictions", "predicted_class_index")

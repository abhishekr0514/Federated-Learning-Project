"""Add User Sessions and Patient Clinician Assignments

Revision ID: 002_auth_sessions_and_assignments
Revises: 001_initial_domain_schema
Create Date: 2026-10-09 20:48:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_auth_sessions_and_assignments'
down_revision = '001_initial_domain_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_jti', sa.String(length=255), nullable=False, unique=True),
        sa.Column('refresh_token_hash', sa.String(length=255), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=255), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_user_sessions_user_id'), 'user_sessions', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_sessions_token_jti'), 'user_sessions', ['token_jti'], unique=True)

    # 2. Add user_id column to patients table
    op.add_column('patients', sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True, unique=True))
    op.create_index(op.f('ix_patients_user_id'), 'patients', ['user_id'], unique=True)

    # 3. Create patient_clinician_assignments table
    op.create_table(
        'patient_clinician_assignments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('clinician_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('clinician_profiles.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('patient_id', 'clinician_id', name='uq_patient_clinician_assignment')
    )
    op.create_index(op.f('ix_patient_clinician_assignments_patient_id'), 'patient_clinician_assignments', ['patient_id'], unique=False)
    op.create_index(op.f('ix_patient_clinician_assignments_clinician_id'), 'patient_clinician_assignments', ['clinician_id'], unique=False)
    op.create_index(op.f('ix_patient_clinician_assignments_hospital_id'), 'patient_clinician_assignments', ['hospital_id'], unique=False)


def downgrade() -> None:
    op.drop_table('patient_clinician_assignments')
    op.drop_constraint('fk_patients_user_id_users', 'patients', type_='foreignkey')
    op.drop_column('patients', 'user_id')
    op.drop_table('user_sessions')

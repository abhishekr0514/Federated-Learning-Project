"""Initial Domain Schema Migration

Revision ID: 001_initial_domain_schema
Revises: 
Create Date: 2026-10-09 20:45:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_domain_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Identity & Access Control Tables
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('is_super_admin', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    op.create_table(
        'roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=100), nullable=False, unique=True),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_roles_name'), 'roles', ['name'], unique=True)

    op.create_table(
        'permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('code', sa.String(length=100), nullable=False, unique=True),
        sa.Column('description', sa.String(length=255), nullable=True),
    )
    op.create_index(op.f('ix_permissions_code'), 'permissions', ['code'], unique=True)

    op.create_table(
        'user_roles',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    )

    op.create_table(
        'role_permissions',
        sa.Column('role_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('permission_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('permissions.id', ondelete='CASCADE'), primary_key=True),
    )

    op.create_table(
        'hospitals',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('code', sa.String(length=50), nullable=False, unique=True),
        sa.Column('name', sa.String(length=255), nullable=False, unique=True),
        sa.Column('region', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_hospitals_code'), 'hospitals', ['code'], unique=True)

    op.create_table(
        'hospital_memberships',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role_in_hospital', sa.String(length=50), nullable=False, server_default='Clinician'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('user_id', 'hospital_id', name='uq_user_hospital_membership')
    )
    op.create_index(op.f('ix_hospital_memberships_user_id'), 'hospital_memberships', ['user_id'], unique=False)
    op.create_index(op.f('ix_hospital_memberships_hospital_id'), 'hospital_memberships', ['hospital_id'], unique=False)

    op.create_table(
        'clinician_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('license_number_hash', sa.String(length=255), nullable=True),
        sa.Column('department', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    # 2. Privacy-Safe Patient Entities
    op.create_table(
        'patients',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('opaque_patient_id', sa.String(length=255), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_patients_opaque_patient_id'), 'patients', ['opaque_patient_id'], unique=True)

    op.create_table(
        'patient_hospital_associations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_patient_ref_hash', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('patient_id', 'hospital_id', name='uq_patient_hospital_assoc')
    )

    # 3. Clinical Metadata References
    op.create_table(
        'medical_documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('local_document_uri', sa.String(length=512), nullable=False),
        sa.Column('document_type', sa.String(length=100), nullable=False, server_default='clinical_note'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'document_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('medical_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('checksum_sha256', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('document_id', 'version_number', name='uq_doc_version')
    )

    op.create_table(
        'document_processing_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('medical_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.String(length=512), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'privacy_masking_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('medical_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('masking_method', sa.String(length=100), nullable=False, server_default='BioClinicalBERT-PII-Masker'),
        sa.Column('entities_detected_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('sanitization_status', sa.String(length=50), nullable=False, server_default='sanitized'),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'ecg_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('local_storage_ref', sa.String(length=512), nullable=False),
        sa.Column('sampling_rate_hz', sa.Integer(), nullable=False, server_default='500'),
        sa.Column('duration_seconds', sa.Float(), nullable=False, server_default='10.0'),
        sa.Column('lead_count', sa.Integer(), nullable=False, server_default='12'),
        sa.Column('dataset_source', sa.String(length=100), nullable=False, server_default='PTB-XL-Proxy'),
        sa.Column('acquired_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'clinical_observations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('observation_type', sa.String(length=100), nullable=False, server_default='tabular_features'),
        sa.Column('feature_schema_version', sa.String(length=50), nullable=False, server_default='v1.0'),
        sa.Column('local_ref', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    # 4. Model Registry, Evaluation Metrics & Predictions
    op.create_table(
        'model_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_family', sa.String(length=100), nullable=False, server_default='MedShield-GAT-Multimodal'),
        sa.Column('version_tag', sa.String(length=50), nullable=False, unique=True),
        sa.Column('architecture_config', sa.String(length=1024), nullable=False),
        sa.Column('target_definition', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='experimental'),
        sa.Column('artifact_uri', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_model_versions_version_tag'), 'model_versions', ['version_tag'], unique=True)

    op.create_table(
        'global_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=100), nullable=False, unique=True),
        sa.Column('active_version_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('model_versions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'model_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_version_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('model_versions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('evaluation_split', sa.String(length=50), nullable=False, server_default='test_split'),
        sa.Column('dataset_version', sa.String(length=100), nullable=False, server_default='PTB-XL-v1.0.3'),
        sa.Column('task_target', sa.String(length=255), nullable=False),
        sa.Column('provenance', sa.String(length=255), nullable=False, server_default='empirical_offline_eval'),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'hospital_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_version_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('model_versions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'diagnosis_predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('model_version_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('model_versions.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('target_definition', sa.String(length=255), nullable=False),
        sa.Column('prediction_type', sa.String(length=50), nullable=False, server_default='proxy_evaluation'),
        sa.Column('provenance', sa.String(length=255), nullable=False, server_default='gat_inference_pipeline'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='disabled'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'prediction_explanations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('diagnosis_predictions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('method', sa.String(length=100), nullable=False, server_default='DiCE-Counterfactual'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'counterfactual_explanations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('explanation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('prediction_explanations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('feature_perturbation_summary', sa.String(length=1024), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'causal_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('diagnosis_predictions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('method', sa.String(length=100), nullable=False, server_default='DoWhy-DAG-Intervention'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    # 5. Federated Learning Entities
    op.create_table(
        'federated_rounds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('round_number', sa.Integer(), nullable=False, unique=True),
        sa.Column('model_version_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('model_versions.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('min_clients_required', sa.Integer(), nullable=False, server_default='2'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_federated_rounds_round_number'), 'federated_rounds', ['round_number'], unique=True)

    op.create_table(
        'federated_clients',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='CASCADE'), nullable=False),
        sa.Column('client_identifier', sa.String(length=100), nullable=False, unique=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='active'),
        sa.Column('registered_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'federated_client_updates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('round_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('federated_rounds.id', ondelete='CASCADE'), nullable=False),
        sa.Column('client_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('federated_clients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('update_status', sa.String(length=50), nullable=False, server_default='submitted'),
        sa.Column('local_sample_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('training_duration_seconds', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('weights_reference_uri', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('round_id', 'client_id', name='uq_round_client_update')
    )

    # 6. Audit & System Safety Entities
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('actor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id', ondelete='SET NULL'), nullable=True),
        sa.Column('resource_type', sa.String(length=100), nullable=False),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('request_id', sa.String(length=100), nullable=False),
        sa.Column('outcome', sa.String(length=50), nullable=False, server_default='SUCCESS'),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_audit_logs_event_type'), 'audit_logs', ['event_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_request_id'), 'audit_logs', ['request_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_created_at'), 'audit_logs', ['created_at'], unique=False)

    op.create_table(
        'security_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('event_name', sa.String(length=100), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False, server_default='INFO'),
        sa.Column('request_id', sa.String(length=100), nullable=False),
        sa.Column('actor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('details_json', sa.String(length=1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        'system_health_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    # Drop in reverse order of creation
    op.drop_table('system_health_snapshots')
    op.drop_table('security_events')
    op.drop_table('audit_logs')
    op.drop_table('federated_client_updates')
    op.drop_table('federated_clients')
    op.drop_table('federated_rounds')
    op.drop_table('causal_analyses')
    op.drop_table('counterfactual_explanations')
    op.drop_table('prediction_explanations')
    op.drop_table('diagnosis_predictions')
    op.drop_table('hospital_metrics')
    op.drop_table('model_metrics')
    op.drop_table('global_models')
    op.drop_table('model_versions')
    op.drop_table('clinical_observations')
    op.drop_table('ecg_records')
    op.drop_table('privacy_masking_results')
    op.drop_table('document_processing_jobs')
    op.drop_table('document_versions')
    op.drop_table('medical_documents')
    op.drop_table('patient_hospital_associations')
    op.drop_table('patients')
    op.drop_table('clinician_profiles')
    op.drop_table('hospital_memberships')
    op.drop_table('hospitals')
    op.drop_table('role_permissions')
    op.drop_table('user_roles')
    op.drop_table('permissions')
    op.drop_table('roles')
    op.drop_table('users')

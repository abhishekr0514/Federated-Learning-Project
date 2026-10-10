"""Test suite for domain model definitions, constraints, and relationships."""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models import (
    Base,
    User,
    Role,
    Permission,
    Hospital,
    HospitalMembership,
    ClinicianProfile,
    Patient,
    PatientHospitalAssociation,
    MedicalDocument,
    DocumentVersion,
    ECGRecord,
    ModelVersion,
    DiagnosisPrediction,
    FederatedRound,
    FederatedClient,
    FederatedClientUpdate,
    AuditLog,
)


@pytest.fixture(scope="module")
def db_session():
    """Create in-memory SQLite session for testing model schemas and constraints."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_user_and_hospital_creation(db_session: Session):
    """Verify User and Hospital creation with UUID primary keys and timestamps."""
    user = User(
        email="test_user@medshield.org",
        hashed_password="$2b$12$TestHashedPasswordValue",
        full_name="Dr. Jane Doe",
    )
    hospital = Hospital(
        code="HOSP-TEST-01",
        name="General Hospital",
        region="North",
    )
    db_session.add(user)
    db_session.add(hospital)
    db_session.commit()

    assert isinstance(user.id, uuid.UUID)
    assert isinstance(hospital.id, uuid.UUID)
    assert user.email == "test_user@medshield.org"
    assert hospital.code == "HOSP-TEST-01"


def test_hospital_membership_relationship(db_session: Session):
    """Verify HospitalMembership relationship linking User and Hospital."""
    user = db_session.query(User).filter_by(email="test_user@medshield.org").first()
    hospital = db_session.query(Hospital).filter_by(code="HOSP-TEST-01").first()

    membership = HospitalMembership(
        user_id=user.id,
        hospital_id=hospital.id,
        role_in_hospital="Clinician",
    )
    db_session.add(membership)
    db_session.commit()

    fetched = db_session.query(HospitalMembership).filter_by(user_id=user.id).first()
    assert fetched is not None
    assert fetched.hospital_id == hospital.id
    assert fetched.user.email == "test_user@medshield.org"
    assert fetched.hospital.code == "HOSP-TEST-01"


def test_opaque_patient_creation(db_session: Session):
    """Verify Patient model uses opaque identifier hash without storing raw PII."""
    patient = Patient(
        opaque_patient_id="OPAQUE_HASH_99481A",
    )
    db_session.add(patient)
    db_session.commit()

    assert isinstance(patient.id, uuid.UUID)
    assert patient.opaque_patient_id == "OPAQUE_HASH_99481A"


def test_medical_document_metadata_relationship(db_session: Session):
    """Verify MedicalDocument stores URI metadata and versions without raw document text."""
    patient = db_session.query(Patient).first()
    hospital = db_session.query(Hospital).first()

    doc = MedicalDocument(
        patient_id=patient.id,
        hospital_id=hospital.id,
        local_document_uri="file:///hospital_local_storage/docs/doc_001.pdf",
        document_type="discharge_summary",
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    version = DocumentVersion(
        document_id=doc.id,
        version_number=1,
        checksum_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    db_session.add(version)
    db_session.commit()

    fetched_doc = db_session.query(MedicalDocument).filter_by(id=doc.id).first()
    assert len(fetched_doc.versions) == 1
    assert fetched_doc.versions[0].checksum_sha256.startswith("e3b0c442")


def test_ecg_record_metadata(db_session: Session):
    """Verify ECGRecord stores metadata and local reference without raw waveform signals."""
    patient = db_session.query(Patient).first()
    hospital = db_session.query(Hospital).first()

    ecg = ECGRecord(
        patient_id=patient.id,
        hospital_id=hospital.id,
        local_storage_ref="file:///hospital_local_storage/ecg/rec_001.mat",
        sampling_rate_hz=500,
        duration_seconds=10.0,
        lead_count=12,
        dataset_source="PTB-XL-Proxy",
    )
    db_session.add(ecg)
    db_session.commit()

    assert ecg.lead_count == 12
    assert ecg.dataset_source == "PTB-XL-Proxy"


def test_model_version_and_predictions(db_session: Session):
    """Verify ModelVersion and DiagnosisPrediction enforce explicit proxy target definitions."""
    model_ver = ModelVersion(
        version_tag="v1.0.0-gat-proxy",
        architecture_config='{"ecg_dim": 128, "text_dim": 128, "tab_dim": 64, "fusion": "GAT"}',
        target_definition="PTB-XL Normal Sinus Rhythm vs Pathological Superclasses Proxy Task",
        status="experimental",
    )
    db_session.add(model_ver)
    db_session.commit()

    patient = db_session.query(Patient).first()
    hospital = db_session.query(Hospital).first()

    pred = DiagnosisPrediction(
        patient_id=patient.id,
        hospital_id=hospital.id,
        model_version_id=model_ver.id,
        target_definition=model_ver.target_definition,
        prediction_type="proxy_evaluation",
        status="disabled",
    )
    db_session.add(pred)
    db_session.commit()

    assert pred.status == "disabled"
    assert "Proxy Task" in pred.target_definition


def test_federated_round_and_client_updates(db_session: Session):
    """Verify FederatedRound and FederatedClientUpdate relationship."""
    model_ver = db_session.query(ModelVersion).first()
    hospital = db_session.query(Hospital).first()

    fl_round = FederatedRound(
        round_number=1,
        model_version_id=model_ver.id,
        status="active",
    )
    fl_client = FederatedClient(
        hospital_id=hospital.id,
        client_identifier="FL-CLIENT-HOSP-01",
        status="active",
    )
    db_session.add(fl_round)
    db_session.add(fl_client)
    db_session.commit()

    update = FederatedClientUpdate(
        round_id=fl_round.id,
        client_id=fl_client.id,
        update_status="accepted",
        local_sample_count=150,
        training_duration_seconds=45.2,
        weights_reference_uri="s3://medshield-fl-artifacts/rounds/1/hosp_01.pt",
    )
    db_session.add(update)
    db_session.commit()

    fetched_update = db_session.query(FederatedClientUpdate).filter_by(round_id=fl_round.id).first()
    assert fetched_update.local_sample_count == 150
    assert fetched_update.client.client_identifier == "FL-CLIENT-HOSP-01"


def test_audit_log_persistence(db_session: Session):
    """Verify AuditLog persistence with request tracing metadata."""
    audit = AuditLog(
        event_type="USER_LOGIN_ATTEMPT",
        resource_type="auth",
        request_id="req-uuid-998811",
        outcome="SUCCESS",
        ip_address="127.0.0.1",
    )
    db_session.add(audit)
    db_session.commit()

    fetched = db_session.query(AuditLog).filter_by(request_id="req-uuid-998811").first()
    assert fetched is not None
    assert fetched.event_type == "USER_LOGIN_ATTEMPT"

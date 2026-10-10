"""Development seeding script for Clinician Workspace smoke testing."""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.app.core.security import hash_password
from backend.app.models import (
    Base,
    User,
    Role,
    Hospital,
    HospitalMembership,
    ClinicianProfile,
    Patient,
    PatientHospitalAssociation,
    PatientClinicianAssignment,
    MedicalDocument,
    DocumentVersion,
    DocumentProcessingJob,
    PrivacyMaskingResult,
    ECGRecord,
    ClinicalObservation,
)

DEV_DB_URL = "sqlite+aiosqlite:///medshield_dev.db"

async def seed():
    engine = create_async_engine(DEV_DB_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        # 1. Roles
        role_clinician = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        role_admin = Role(id=uuid.uuid4(), name="HOSPITAL_ADMIN", description="Hospital Administrator")
        role_patient = Role(id=uuid.uuid4(), name="PATIENT", description="Patient")
        session.add_all([role_clinician, role_admin, role_patient])
        await session.commit()

        # 2. Hospital
        hospital = Hospital(id=uuid.uuid4(), code="HOSP-METRO", name="Metro General Hospital")
        session.add(hospital)
        await session.commit()

        # 3. Clinician User
        clin_user = User(
            id=uuid.uuid4(),
            email="dr_metro@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            is_super_admin=False,
        )
        clin_user.roles.append(role_clinician)
        session.add(clin_user)
        await session.commit()

        # 4. Membership & Profile
        membership = HospitalMembership(
            id=uuid.uuid4(),
            user_id=clin_user.id,
            hospital_id=hospital.id,
            role_in_hospital="CLINICIAN",
        )
        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=clin_user.id,
            hospital_id=hospital.id,
            department="Cardiology & Electrophysiology",
            license_number_hash="HASH_LIC_METRO_001",
        )
        session.add_all([membership, profile])
        await session.commit()

        # 5. Assigned Patient
        pat_1 = Patient(
            id=uuid.uuid4(),
            opaque_patient_id="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            user_id=None,
        )
        # Unassigned Patient
        pat_2 = Patient(
            id=uuid.uuid4(),
            opaque_patient_id="a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8",
            user_id=None,
        )
        session.add_all([pat_1, pat_2])
        await session.commit()

        assoc_1 = PatientHospitalAssociation(
            id=uuid.uuid4(),
            patient_id=pat_1.id,
            hospital_id=hospital.id,
            hospital_patient_ref_hash="REF_METRO_001",
        )
        assoc_2 = PatientHospitalAssociation(
            id=uuid.uuid4(),
            patient_id=pat_2.id,
            hospital_id=hospital.id,
            hospital_patient_ref_hash="REF_METRO_002",
        )
        session.add_all([assoc_1, assoc_2])
        await session.commit()

        # Explicit Clinician Assignment for Patient 1 ONLY
        assign_1 = PatientClinicianAssignment(
            id=uuid.uuid4(),
            patient_id=pat_1.id,
            clinician_id=profile.id,
            hospital_id=hospital.id,
        )
        session.add(assign_1)
        await session.commit()

        # 6. Documents and ECG for Patient 1
        doc_1 = MedicalDocument(
            id=uuid.uuid4(),
            patient_id=pat_1.id,
            hospital_id=hospital.id,
            local_document_uri="local://metro/storage/notes/doc_001.bin",
            document_type="clinical_note",
            status="completed",
        )
        session.add(doc_1)
        await session.commit()

        ver_1 = DocumentVersion(
            id=uuid.uuid4(),
            document_id=doc_1.id,
            version_number=1,
            checksum_sha256="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        )
        job_1 = DocumentProcessingJob(
            id=uuid.uuid4(),
            document_id=doc_1.id,
            status="completed",
        )
        mask_1 = PrivacyMaskingResult(
            id=uuid.uuid4(),
            document_id=doc_1.id,
            masking_method="BioClinicalBERT-Sanitizer",
            entities_detected_count=4,
            sanitization_status="sanitized",
        )
        session.add_all([ver_1, job_1, mask_1])
        await session.commit()

        ecg_1 = ECGRecord(
            id=uuid.uuid4(),
            patient_id=pat_1.id,
            hospital_id=hospital.id,
            local_storage_ref="local://metro/ecg/record_001.mat",
            sampling_rate_hz=500,
            duration_seconds=10.0,
            lead_count=12,
            dataset_source="PTB-XL-Proxy",
        )
        obs_1 = ClinicalObservation(
            id=uuid.uuid4(),
            patient_id=pat_1.id,
            hospital_id=hospital.id,
            observation_type="tabular_vitals",
            feature_schema_version="v1.0",
            local_ref="local://metro/obs/vitals_001.parquet",
        )
        session.add_all([ecg_1, obs_1])
        await session.commit()

    print("Dev database seeded successfully with clinician: dr_metro@metro.org / Password123!")

if __name__ == "__main__":
    asyncio.run(seed())

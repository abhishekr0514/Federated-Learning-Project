"""Development seeding script for Patient Portal testing and verification."""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.app.core.security import hash_password
from backend.app.models import (
    Base,
    User,
    Role,
    Hospital,
    Patient,
    PatientHospitalAssociation,
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
        # 1. Ensure Roles
        role_pat = (await session.execute(select(Role).where(Role.name == "PATIENT"))).scalar_one_or_none()
        if not role_pat:
            role_pat = Role(id=uuid.uuid4(), name="PATIENT", description="Patient Portal User")
            session.add(role_pat)
            await session.commit()

        # 2. Ensure Hospital
        hospital = (await session.execute(select(Hospital).where(Hospital.code == "HOSP-METRO"))).scalar_one_or_none()
        if not hospital:
            hospital = Hospital(id=uuid.uuid4(), code="HOSP-METRO", name="Metro General Hospital")
            session.add(hospital)
            await session.commit()

        # 3. Create or Update Patient User
        pat_user = (await session.execute(select(User).where(User.email == "patient_alice@example.com"))).scalar_one_or_none()
        if not pat_user:
            pat_user = User(
                id=uuid.uuid4(),
                email="patient_alice@example.com",
                hashed_password=hash_password("Password123!"),
                full_name="Alice Smith",
                is_active=True,
                is_super_admin=False,
            )
            pat_user.roles.append(role_pat)
            session.add(pat_user)
            await session.commit()

        # 4. Link Patient Entity to pat_user.id
        pat_entity = (await session.execute(select(Patient).where(Patient.user_id == pat_user.id))).scalar_one_or_none()
        if not pat_entity:
            pat_entity = Patient(
                id=uuid.uuid4(),
                opaque_patient_id="c89a71239bf049b1a520fa981e7d34107298319f030f81d86d5268c07e594d21",
                user_id=pat_user.id,
            )
            session.add(pat_entity)
            await session.commit()

            assoc = PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=pat_entity.id,
                hospital_id=hospital.id,
                hospital_patient_ref_hash="REF_METRO_ALICE_001",
            )
            session.add(assoc)
            await session.commit()

            # 5. Documents, ECG, and Observations for Alice
            doc = MedicalDocument(
                id=uuid.uuid4(),
                patient_id=pat_entity.id,
                hospital_id=hospital.id,
                local_document_uri="local://metro/storage/notes/doc_alice_001.bin",
                document_type="cardiology_consult",
                status="completed",
            )
            session.add(doc)
            await session.commit()

            mask = PrivacyMaskingResult(
                id=uuid.uuid4(),
                document_id=doc.id,
                masking_method="BioClinicalBERT-PII-Masker",
                entities_detected_count=2,
                sanitization_status="sanitized",
            )
            ecg = ECGRecord(
                id=uuid.uuid4(),
                patient_id=pat_entity.id,
                hospital_id=hospital.id,
                local_storage_ref="local://metro/ecg/alice_ecg_001.mat",
                sampling_rate_hz=500,
                duration_seconds=10.0,
                lead_count=12,
                dataset_source="PTB-XL-Proxy",
            )
            obs = ClinicalObservation(
                id=uuid.uuid4(),
                patient_id=pat_entity.id,
                hospital_id=hospital.id,
                observation_type="vital_signs",
                feature_schema_version="v1.0",
                local_ref="local://metro/obs/alice_vitals_001.parquet",
            )
            session.add_all([mask, ecg, obs])
            await session.commit()

    print("Dev database seeded successfully with patient: patient_alice@example.com / Password123!")

if __name__ == "__main__":
    asyncio.run(seed())

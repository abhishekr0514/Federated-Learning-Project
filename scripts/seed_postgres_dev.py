"""Idempotently seed MedShield-FL development users in PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from getpass import getpass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402
from backend.app.core.security import hash_password  # noqa: E402
from backend.app.db.connectivity import (  # noqa: E402
    assert_postgres_configuration,
    database_target_summary,
    verify_async_connection,
)
from backend.app.models.identity import (  # noqa: E402
    ClinicianProfile,
    Hospital,
    HospitalMembership,
    Role,
    User,
)
from backend.app.models.clinical import (  # noqa: E402
    ClinicalObservation,
    ECGRecord,
    MedicalDocument,
    PrivacyMaskingResult,
)
from backend.app.models.patient import Patient, PatientHospitalAssociation  # noqa: E402

EXPECTED_DATABASE = "medshield_fl_db"
DEV_HOSPITAL_CODE = "HOSP-METRO"
DEV_HOSPITAL_NAME = "Metro General Hospital"
DEV_PATIENT_OPAQUE_ID = (
    "c89a71239bf049b1a520fa981e7d34107298319f030f81d86d5268c07e594d21"
)
DEV_PATIENT_REF_HASH = "REF_METRO_ALICE_001"
PATIENT_DEV_EMAIL = "patient_alice@example.com"
MIN_PASSWORD_LEN = 12

ACCOUNTS = [
    {
        "email": "dr_metro@metro.org",
        "full_name": "Dr Metro",
        "role": "CLINICIAN",
    },
    {
        "email": "patient_alice@example.com",
        "full_name": "Alice Smith",
        "role": "PATIENT",
    },
]


def _prompt_new_password(email: str) -> str:
    while True:
        password = getpass(f"Set a development password for {email} (min {MIN_PASSWORD_LEN} chars): ")
        if len(password) < MIN_PASSWORD_LEN:
            print(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
            continue
        confirm = getpass("Confirm password: ")
        if password != confirm:
            print("Passwords did not match.")
            continue
        return password


async def _ensure_clinician_workspace(
    session,
    user: User,
) -> None:
    """Minimum records for GET /api/v1/clinician/overview (profile guard + hospital context)."""
    hospital_result = await session.execute(
        select(Hospital).where(Hospital.code == DEV_HOSPITAL_CODE)
    )
    hospital = hospital_result.scalar_one_or_none()
    if hospital is None:
        hospital = Hospital(
            id=uuid.uuid4(),
            code=DEV_HOSPITAL_CODE,
            name=DEV_HOSPITAL_NAME,
            is_active=True,
        )
        session.add(hospital)
        await session.flush()
        print(f"Created hospital: {DEV_HOSPITAL_CODE}")
    else:
        print(f"Reused hospital: {DEV_HOSPITAL_CODE}")

    membership_result = await session.execute(
        select(HospitalMembership).where(
            HospitalMembership.user_id == user.id,
            HospitalMembership.hospital_id == hospital.id,
        )
    )
    membership = membership_result.scalar_one_or_none()
    if membership is None:
        session.add(
            HospitalMembership(
                id=uuid.uuid4(),
                user_id=user.id,
                hospital_id=hospital.id,
                role_in_hospital="CLINICIAN",
            )
        )
        print(f"Created hospital membership for {user.email}")
    else:
        print(f"Reused hospital membership for {user.email}")

    profile_result = await session.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == user.id)
    )
    profile = profile_result.scalar_one_or_none()
    if profile is None:
        session.add(
            ClinicianProfile(
                id=uuid.uuid4(),
                user_id=user.id,
                hospital_id=hospital.id,
                department="Cardiology & Electrophysiology",
                license_number_hash="DEV_HASH_LIC_METRO_001",
            )
        )
        print(f"Created clinician profile for {user.email}")
    elif profile.hospital_id != hospital.id:
        profile.hospital_id = hospital.id
        print(f"Aligned clinician profile hospital for {user.email}")
    else:
        print(f"Reused clinician profile for {user.email}")


async def _ensure_patient_portal_link(
    session,
    hospital: Hospital,
    patient_user: User,
) -> None:
    """Opaque Patient + hospital association for portal user (no clinician assignment)."""
    patient_result = await session.execute(
        select(Patient).where(Patient.user_id == patient_user.id)
    )
    patient = patient_result.scalar_one_or_none()
    if patient is None:
        opaque_result = await session.execute(
            select(Patient).where(Patient.opaque_patient_id == DEV_PATIENT_OPAQUE_ID)
        )
        patient = opaque_result.scalar_one_or_none()
        if patient is None:
            patient = Patient(
                id=uuid.uuid4(),
                opaque_patient_id=DEV_PATIENT_OPAQUE_ID,
                user_id=patient_user.id,
            )
            session.add(patient)
            await session.flush()
            print(f"Created opaque patient record for {patient_user.email}")
        elif patient.user_id is None:
            patient.user_id = patient_user.id
            print(f"Linked existing opaque patient to {patient_user.email}")
        else:
            print(f"Reused opaque patient for {patient_user.email}")
    else:
        print(f"Reused patient portal link for {patient_user.email}")

    assoc_result = await session.execute(
        select(PatientHospitalAssociation).where(
            PatientHospitalAssociation.patient_id == patient.id,
            PatientHospitalAssociation.hospital_id == hospital.id,
        )
    )
    if assoc_result.scalar_one_or_none() is None:
        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient.id,
                hospital_id=hospital.id,
                hospital_patient_ref_hash=DEV_PATIENT_REF_HASH,
            )
        )
        print(f"Created hospital association for {patient_user.email} at {hospital.code}")
    else:
        print(f"Reused hospital association for {patient_user.email} at {hospital.code}")


async def _ensure_dev_clinical_metadata(
    session,
    hospital: Hospital,
    patient: Patient,
) -> None:
    """Optional dev-only metadata rows (local refs, no raw PHI, no AI predictions)."""
    doc_result = await session.execute(
        select(MedicalDocument.id).where(
            MedicalDocument.patient_id == patient.id,
            MedicalDocument.local_document_uri == "local://metro/storage/notes/doc_alice_001.bin",
        )
    )
    if doc_result.scalar_one_or_none() is None:
        doc = MedicalDocument(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=hospital.id,
            local_document_uri="local://metro/storage/notes/doc_alice_001.bin",
            document_type="cardiology_consult",
            status="completed",
        )
        session.add(doc)
        await session.flush()
        session.add(
            PrivacyMaskingResult(
                id=uuid.uuid4(),
                document_id=doc.id,
                masking_method="BioClinicalBERT-PII-Masker",
                entities_detected_count=2,
                sanitization_status="sanitized",
            )
        )
        print("Created dev medical document metadata for Alice")
    else:
        print("Reused dev medical document metadata for Alice")

    ecg_result = await session.execute(
        select(ECGRecord.id).where(
            ECGRecord.patient_id == patient.id,
            ECGRecord.local_storage_ref == "local://metro/ecg/alice_ecg_001.mat",
        )
    )
    if ecg_result.scalar_one_or_none() is None:
        session.add(
            ECGRecord(
                id=uuid.uuid4(),
                patient_id=patient.id,
                hospital_id=hospital.id,
                local_storage_ref="local://metro/ecg/alice_ecg_001.mat",
                sampling_rate_hz=500,
                duration_seconds=10.0,
                lead_count=12,
                dataset_source="PTB-XL-Proxy",
            )
        )
        print("Created dev ECG metadata for Alice")
    else:
        print("Reused dev ECG metadata for Alice")

    obs_result = await session.execute(
        select(ClinicalObservation.id).where(
            ClinicalObservation.patient_id == patient.id,
            ClinicalObservation.local_ref == "local://metro/obs/alice_vitals_001.parquet",
        )
    )
    if obs_result.scalar_one_or_none() is None:
        session.add(
            ClinicalObservation(
                id=uuid.uuid4(),
                patient_id=patient.id,
                hospital_id=hospital.id,
                observation_type="vital_signs",
                feature_schema_version="v1.0",
                local_ref="local://metro/obs/alice_vitals_001.parquet",
            )
        )
        print("Created dev observation metadata for Alice")
    else:
        print("Reused dev observation metadata for Alice")


async def seed(*, with_dev_clinical_metadata: bool = False) -> None:
    assert_postgres_configuration(expected_database=EXPECTED_DATABASE)
    target = database_target_summary()
    print(
        f"Target database: {target['username']}@{target['host']}:{target['port']}/{target['database']}"
    )

    connectivity = await verify_async_connection()
    if not connectivity.ok:
        print(f"Aborting: database connection failed ({connectivity.error_type}).")
        print("Fix POSTGRES_* values in the repository .env file, then retry.")
        raise SystemExit(1)

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with Session() as session:
            async with session.begin():
                for account in ACCOUNTS:
                    role_result = await session.execute(
                        select(Role).where(Role.name == account["role"])
                    )
                    role = role_result.scalar_one_or_none()

                    if role is None:
                        role = Role(
                            id=uuid.uuid4(),
                            name=account["role"],
                            description=f"{account['role'].title()} development role",
                        )
                        session.add(role)
                        await session.flush()
                        print(f"Created role: {account['role']}")
                    else:
                        print(f"Reused role: {account['role']}")

                    user_result = await session.execute(
                        select(User)
                        .where(User.email == account["email"])
                        .options(selectinload(User.roles))
                    )
                    user = user_result.scalar_one_or_none()

                    if user is None:
                        password = _prompt_new_password(account["email"])
                        user = User(
                            id=uuid.uuid4(),
                            email=account["email"],
                            full_name=account["full_name"],
                            hashed_password=hash_password(password),
                            is_active=True,
                            is_super_admin=False,
                        )
                        user.roles.append(role)
                        session.add(user)
                        await session.flush()
                        print(f"Created account: {account['email']}")
                    else:
                        if not any(r.name == role.name for r in user.roles):
                            user.roles.append(role)
                            print(f"Updated roles for existing account: {account['email']}")
                        else:
                            print(f"Reused existing account: {account['email']}")

                    if account["role"] == "CLINICIAN":
                        await _ensure_clinician_workspace(session, user)

                hospital_result = await session.execute(
                    select(Hospital).where(Hospital.code == DEV_HOSPITAL_CODE)
                )
                hospital = hospital_result.scalar_one_or_none()
                if hospital is not None:
                    pat_user_result = await session.execute(
                        select(User).where(User.email == PATIENT_DEV_EMAIL)
                    )
                    pat_user = pat_user_result.scalar_one_or_none()
                    if pat_user is not None:
                        await _ensure_patient_portal_link(session, hospital, pat_user)
                        if with_dev_clinical_metadata:
                            patient_result = await session.execute(
                                select(Patient).where(Patient.user_id == pat_user.id)
                            )
                            linked_patient = patient_result.scalar_one_or_none()
                            if linked_patient is not None:
                                await _ensure_dev_clinical_metadata(
                                    session, hospital, linked_patient
                                )

        print("PostgreSQL development-user seeding completed.")
        print(
            "Patient-clinician assignments are not seeded automatically. "
            "Use POST /api/v1/clinician/patients/{patient_id}/assignment as a clinician."
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed PostgreSQL development users.")
    parser.add_argument(
        "--with-dev-clinical-metadata",
        action="store_true",
        help="Insert idempotent metadata-only clinical rows (local URIs, no AI predictions).",
    )
    args = parser.parse_args()
    asyncio.run(seed(with_dev_clinical_metadata=args.with_dev_clinical_metadata))

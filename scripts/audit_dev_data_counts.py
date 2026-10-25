"""Read-only PostgreSQL counts for dev portal audit (no secrets, no PHI)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402
from backend.app.db.connectivity import assert_postgres_configuration, verify_async_connection  # noqa: E402
from backend.app.models.clinical import ClinicalObservation, ECGRecord, MedicalDocument  # noqa: E402
from backend.app.models.identity import User  # noqa: E402
from backend.app.models.ml_models import DiagnosisPrediction  # noqa: E402
from backend.app.models.patient import (  # noqa: E402
    Patient,
    PatientClinicianAssignment,
    PatientHospitalAssociation,
)

PATIENT_EMAIL = "patient_alice@example.com"
CLINICIAN_EMAIL = "dr_metro@metro.org"


async def main() -> int:
    assert_postgres_configuration()
    conn = await verify_async_connection()
    if not conn.ok:
        print("Database unavailable.")
        return 1

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        pat_user = (
            await session.execute(select(User.id).where(User.email == PATIENT_EMAIL))
        ).scalar_one_or_none()
        patient = None
        if pat_user:
            patient = (
                await session.execute(select(Patient).where(Patient.user_id == pat_user))
            ).scalar_one_or_none()

        print(f"patient_user_linked: {pat_user is not None}")
        print(f"patient_entity_linked: {patient is not None}")
        if patient:
            pid = patient.id
            assoc_n = (
                await session.execute(
                    select(func.count(PatientHospitalAssociation.id)).where(
                        PatientHospitalAssociation.patient_id == pid
                    )
                )
            ).scalar_one()
            assign_n = (
                await session.execute(
                    select(func.count(PatientClinicianAssignment.id)).where(
                        PatientClinicianAssignment.patient_id == pid
                    )
                )
            ).scalar_one()
            doc_n = (
                await session.execute(
                    select(func.count(MedicalDocument.id)).where(MedicalDocument.patient_id == pid)
                )
            ).scalar_one()
            ecg_n = (
                await session.execute(
                    select(func.count(ECGRecord.id)).where(ECGRecord.patient_id == pid)
                )
            ).scalar_one()
            obs_n = (
                await session.execute(
                    select(func.count(ClinicalObservation.id)).where(ClinicalObservation.patient_id == pid)
                )
            ).scalar_one()
            pred_n = (
                await session.execute(
                    select(func.count(DiagnosisPrediction.id)).where(DiagnosisPrediction.patient_id == pid)
                )
            ).scalar_one()
            print(f"hospital_associations: {assoc_n}")
            print(f"clinician_assignments: {assign_n}")
            print(f"medical_documents: {doc_n}")
            print(f"ecg_records: {ecg_n}")
            print(f"clinical_observations: {obs_n}")
            print(f"diagnosis_predictions: {pred_n}")

        clin_user = (
            await session.execute(select(User.id).where(User.email == CLINICIAN_EMAIL))
        ).scalar_one_or_none()
        if clin_user:
            from backend.app.models.identity import ClinicianProfile

            profile = (
                await session.execute(
                    select(ClinicianProfile).where(ClinicianProfile.user_id == clin_user)
                )
            ).scalar_one_or_none()
            if profile:
                assigned = (
                    await session.execute(
                        select(func.count(PatientClinicianAssignment.id)).where(
                            PatientClinicianAssignment.clinician_id == profile.id
                        )
                    )
                ).scalar_one()
                print(f"clinician_assigned_patients: {assigned}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

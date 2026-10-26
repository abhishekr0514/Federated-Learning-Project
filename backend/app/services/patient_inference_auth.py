"""Server-side authorization for patient-linked inference (GAT / ECG-report)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.errors import ForbiddenError, NotFoundError
from backend.app.core.security import UserRole
from backend.app.models.identity import ClinicianProfile, HospitalMembership, User
from backend.app.models.patient import Patient, PatientClinicianAssignment, PatientHospitalAssociation


@dataclass
class PatientInferenceContext:
    patient: Patient
    profile: Optional[ClinicianProfile]
    hospital_id: uuid.UUID


async def authorize_patient_linked_inference(
    db: AsyncSession,
    current_user: User,
    patient_id: uuid.UUID,
) -> PatientInferenceContext:
    """
    Enforce role-specific access before persisting DiagnosisPrediction.

    - Super Admin: any existing patient reference.
    - Hospital Admin: patient must belong to an authorized hospital membership.
    - Clinician: patient must be explicitly assigned to the clinician profile.
    """
    stmt_patient = select(Patient).where(Patient.id == patient_id)
    patient = (await db.execute(stmt_patient)).scalar_one_or_none()
    if not patient:
        raise NotFoundError("Patient reference resource not accessible")

    if current_user.is_super_admin:
        hospital_id = await _resolve_hospital_id_for_patient(db, patient.id)
        profile_stmt = select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
        profile = (await db.execute(profile_stmt)).scalar_one_or_none()
        return PatientInferenceContext(patient=patient, profile=profile, hospital_id=hospital_id)

    role_names = {r.name for r in current_user.roles}
    is_clinician = UserRole.CLINICIAN.value in role_names
    is_hospital_admin = UserRole.HOSPITAL_ADMIN.value in role_names

    profile: Optional[ClinicianProfile] = None
    if is_clinician:
        profile_stmt = select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
        profile = (await db.execute(profile_stmt)).scalar_one_or_none()
        if not profile:
            raise ForbiddenError("Active Clinician Profile required for patient-associated evaluation.")

    authorized = False
    hospital_id: Optional[uuid.UUID] = None

    if is_clinician and profile:
        stmt_assign = select(PatientClinicianAssignment).where(
            PatientClinicianAssignment.patient_id == patient_id,
            PatientClinicianAssignment.clinician_id == profile.id,
            PatientClinicianAssignment.hospital_id == profile.hospital_id,
        )
        if (await db.execute(stmt_assign)).scalar_one_or_none():
            authorized = True
            hospital_id = profile.hospital_id

    if is_hospital_admin and not authorized:
        mem_stmt = select(HospitalMembership.hospital_id).where(
            HospitalMembership.user_id == current_user.id
        )
        allowed_hospitals = set((await db.execute(mem_stmt)).scalars().all())
        if allowed_hospitals:
            assoc_stmt = select(PatientHospitalAssociation).where(
                PatientHospitalAssociation.patient_id == patient_id,
                PatientHospitalAssociation.hospital_id.in_(allowed_hospitals),
            )
            assoc = (await db.execute(assoc_stmt)).scalar_one_or_none()
            if assoc:
                authorized = True
                hospital_id = assoc.hospital_id

    if not authorized or hospital_id is None:
        raise NotFoundError("Patient reference resource not accessible")

    return PatientInferenceContext(patient=patient, profile=profile, hospital_id=hospital_id)


async def _resolve_hospital_id_for_patient(db: AsyncSession, patient_id: uuid.UUID) -> uuid.UUID:
    stmt = (
        select(PatientHospitalAssociation.hospital_id)
        .where(PatientHospitalAssociation.patient_id == patient_id)
        .limit(1)
    )
    hospital_id = (await db.execute(stmt)).scalar_one_or_none()
    if not hospital_id:
        raise NotFoundError("Patient reference resource not accessible")
    return hospital_id

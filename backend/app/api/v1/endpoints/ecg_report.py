"""FastAPI endpoints for operational two-modality ECG + clinical report inference."""

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import (
    get_current_clinician_context,
    get_current_user,
    get_db,
    require_role,
)
from backend.app.core.errors import ForbiddenError, NotFoundError
from backend.app.core.security import UserRole
from backend.app.models.identity import ClinicianProfile, User
from backend.app.models.patient import Patient, PatientClinicianAssignment
from backend.app.schemas.ecg_report_inference import (
    ECGReportEvaluateRequest,
    ECGReportEvaluateResponse,
    ECGReportReadinessResponse,
)
from backend.app.services.ecg_report_inference_service import ECGReportInferenceService

router = APIRouter(prefix="/ecg-report", tags=["ECG Report Inference"])


@router.get("/readiness", response_model=ECGReportReadinessResponse)
async def get_ecg_report_readiness() -> Dict[str, Any]:
    """
    Safe readiness diagnostics for ECG + clinical report model.
    Does not expose filesystem paths or credentials.
    """
    return ECGReportInferenceService.get_readiness()


@router.post(
    "/evaluate",
    response_model=ECGReportEvaluateResponse,
    status_code=status.HTTP_200_OK,
)
async def evaluate_ecg_report(
    body: ECGReportEvaluateRequest,
    patient_id: Optional[uuid.UUID] = Query(None, description="Optional assigned patient UUID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        require_role(UserRole.CLINICIAN, UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN)
    ),
) -> ECGReportEvaluateResponse:
    """
    Evaluate 12-lead ECG and clinical report text with verified checkpoint.

    - If patient_id is provided, requires calling user to be an assigned clinician
      and persists the prediction with clinical provenance.
    - If patient_id is omitted, performs research-only proxy evaluation without
      persisting patient health records.
    """
    target_patient: Optional[Patient] = None
    target_profile: Optional[ClinicianProfile] = None

    if patient_id is not None:
        # 1. Resolve clinician profile
        stmt_prof = select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
        res_prof = await db.execute(stmt_prof)
        target_profile = res_prof.scalar_one_or_none()

        if not current_user.is_super_admin and not target_profile:
            raise ForbiddenError("Active Clinician Profile required for patient-associated evaluation.")

        # 2. Resolve patient and verify assignment
        stmt_patient = select(Patient).where(Patient.id == patient_id)
        res_patient = await db.execute(stmt_patient)
        target_patient = res_patient.scalar_one_or_none()

        if not target_patient:
            raise NotFoundError("Patient reference resource not accessible")

        if not current_user.is_super_admin and target_profile:
            stmt_assign = select(PatientClinicianAssignment).where(
                PatientClinicianAssignment.patient_id == patient_id,
                PatientClinicianAssignment.clinician_id == target_profile.id,
                PatientClinicianAssignment.hospital_id == target_profile.hospital_id,
            )
            res_assign = await db.execute(stmt_assign)
            if not res_assign.scalar_one_or_none():
                raise NotFoundError("Patient reference resource not accessible")

    result = await ECGReportInferenceService.evaluate(
        db,
        body=body,
        user=current_user,
        profile=target_profile,
        patient=target_patient,
    )
    if target_patient and db:
        await db.commit()
    return result

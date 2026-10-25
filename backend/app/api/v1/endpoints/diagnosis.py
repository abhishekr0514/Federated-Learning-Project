"""Multimodal clinical diagnosis inference (research proxy evaluation)."""

import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import (
    get_authorized_clinician_patient,
    get_current_clinician_context,
    get_db,
)
from backend.app.core.errors import ForbiddenError, NotFoundError
from backend.app.models.identity import ClinicianProfile, User
from backend.app.models.patient import Patient
from backend.app.schemas.diagnosis_inference import DiagnosisEvaluateRequest, DiagnosisEvaluateResponse
from backend.app.services.diagnosis_inference_service import DiagnosisInferenceService

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


@router.get("/readiness")
async def get_diagnosis_inference_readiness() -> Dict[str, Any]:
    """Safe readiness diagnostics (no filesystem paths or secrets)."""
    return DiagnosisInferenceService.get_readiness()


@router.post("/evaluate", response_model=DiagnosisEvaluateResponse)
async def evaluate_multimodal_diagnosis(
    body: DiagnosisEvaluateRequest,
    patient_id: uuid.UUID = Query(..., description="Assigned patient UUID"),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: tuple[User, ClinicianProfile | None] = Depends(get_current_clinician_context),
    patient: Patient = Depends(get_authorized_clinician_patient),
) -> DiagnosisEvaluateResponse:
    """
    Evaluate precomputed modality embeddings with verified GAT fusion checkpoint.

    Requires clinician role, active profile, and explicit patient assignment.
    """
    user, profile = clinician_ctx
    if profile is None:
        raise ForbiddenError("Clinician profile required for inference submission")
    if patient.id != patient_id:
        raise NotFoundError("Patient reference resource not accessible")

    result = await DiagnosisInferenceService.evaluate_for_assigned_patient(
        db,
        user=user,
        profile=profile,
        patient=patient,
        body=body,
        patient_id=patient_id,
    )
    await db.commit()
    return result

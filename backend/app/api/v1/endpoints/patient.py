"""Patient Portal API v1 Endpoints.

Provides privacy-safe, non-technical health overview, authorized records,
reports, and AI assessment states strictly scoped to the authenticated patient.
"""

import uuid
from typing import Any, Dict, Tuple
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import (
    get_authorized_patient_self,
    get_current_patient_context,
    get_db,
)
from backend.app.models.identity import User
from backend.app.models.patient import Patient
from backend.app.services.patient_service import PatientService

router = APIRouter(prefix="/patient", tags=["Patient Portal"])


@router.get("/overview")
async def get_patient_overview(
    db: AsyncSession = Depends(get_db),
    patient_ctx: Tuple[User, Patient] = Depends(get_current_patient_context),
) -> Dict[str, Any]:
    """Retrieve high-level health summary and assessment capability status."""
    _, patient = patient_ctx
    return await PatientService.get_health_overview(db, patient)


@router.get("/records")
async def get_patient_records(
    db: AsyncSession = Depends(get_db),
    patient_ctx: Tuple[User, Patient] = Depends(get_current_patient_context),
) -> Dict[str, Any]:
    """Retrieve personal authorized health records (documents, vitals, ECG test dates)."""
    _, patient = patient_ctx
    return await PatientService.get_records(db, patient)


@router.get("/records/{patient_id}")
async def get_patient_records_scoped(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    patient: Patient = Depends(get_authorized_patient_self),
) -> Dict[str, Any]:
    """Retrieve personal health records with route parameter validation.
    
    Enforces patient_id matches the authenticated patient; returns 404 otherwise.
    """
    return await PatientService.get_records(db, patient)


@router.get("/reports")
async def get_patient_reports(
    db: AsyncSession = Depends(get_db),
    patient_ctx: Tuple[User, Patient] = Depends(get_current_patient_context),
) -> Dict[str, Any]:
    """Retrieve personal diagnostic reports and privacy sanitization summaries."""
    _, patient = patient_ctx
    return await PatientService.get_reports(db, patient)


@router.get("/diagnosis")
async def get_patient_diagnosis(
    db: AsyncSession = Depends(get_db),
    patient_ctx: Tuple[User, Patient] = Depends(get_current_patient_context),
) -> Dict[str, Any]:
    """Retrieve personal AI-assisted assessment results or honest unavailable status."""
    _, patient = patient_ctx
    return await PatientService.get_diagnosis(db, patient)


@router.get("/diagnosis/{patient_id}")
async def get_patient_diagnosis_scoped(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    patient: Patient = Depends(get_authorized_patient_self),
) -> Dict[str, Any]:
    """Retrieve personal assessment results with route parameter validation.
    
    Enforces patient_id matches the authenticated patient; returns 404 otherwise.
    """
    return await PatientService.get_diagnosis(db, patient)

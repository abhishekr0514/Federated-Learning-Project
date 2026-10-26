"""Clinician Workspace API Endpoints.

Enforces:
1. Strict Clinician Role & Active Profile check (403 for non-clinicians).
2. Explicit Patient Assignment Scoping: Clinicians query ONLY assigned patients (404 for unassigned/cross-tenant).
3. Zero Mock Data Contract: Real database records only; canonical honest states for unavailable AI modules.
4. Privacy Boundary: Zero PII, zero raw clinical notes, zero raw ECG waveform arrays.
"""

import uuid
from typing import Any, Dict, Optional, Tuple
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import (
    get_authorized_clinician_patient,
    get_current_clinician_context,
    get_db,
)
from backend.app.models.identity import ClinicianProfile, User
from backend.app.models.patient import Patient
from backend.app.services.clinician_service import ClinicianService
from backend.app.schemas.patient_questions import ClinicianQuestionResponseRequest
from backend.app.services.document_ingestion_service import DocumentIngestionService
from backend.app.services.patient_question_service import PatientQuestionService

router = APIRouter(prefix="/clinician", tags=["Clinician Workspace"])


@router.get("/overview")
async def get_clinician_overview(
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve clinician profile summary, assigned patient counters, and subsystem capability states."""
    user, profile = clinician_ctx
    return await ClinicianService.get_overview(db, user, profile)


@router.get("/patients")
async def get_assigned_patients(
    limit: int = Query(50, ge=1, le=100, description="Page size limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve paginated list of authorized assigned patient references."""
    user, profile = clinician_ctx
    return await ClinicianService.get_assigned_patients(
        db,
        profile,
        limit=limit,
        offset=offset,
        super_admin_list_all=user.is_super_admin,
    )


@router.get("/patients/eligible")
async def get_eligible_patients_for_assignment(
    limit: int = Query(50, ge=1, le=100, description="Page size limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Patients at the clinician's hospital who are not yet assigned to this clinician."""
    _, profile = clinician_ctx
    return await ClinicianService.get_eligible_patients_for_assignment(
        db, profile, limit=limit, offset=offset
    )


@router.post("/patients/{patient_id}/assignment")
async def assign_patient_to_clinician(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Explicitly assign an eligible hospital patient to the authenticated clinician."""
    _, profile = clinician_ctx
    result = await ClinicianService.assign_patient(db, profile, patient_id)
    await db.commit()
    return result


@router.get("/patients/{patient_id}")
async def get_patient_detail(
    patient: Patient = Depends(get_authorized_clinician_patient),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve full clinical metadata dossier for an authorized assigned patient reference.
    
    Returns HTTP 404 for unassigned patients or patients from another hospital node.
    """
    _, profile = clinician_ctx
    return await ClinicianService.get_patient_detail(db, profile, patient)


@router.get("/documents")
async def get_clinical_documents(
    patient_id: Optional[uuid.UUID] = Query(None, description="Optional patient UUID filter"),
    limit: int = Query(50, ge=1, le=100, description="Page size limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve paginated medical document metadata for assigned patients."""
    _, profile = clinician_ctx
    return await ClinicianService.get_documents(
        db, profile, patient_id=patient_id, limit=limit, offset=offset
    )


@router.get("/documents/{document_id}")
async def get_document_detail(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve document version and processing job metadata.
    
    Returns HTTP 404 if document belongs to an unassigned patient.
    """
    _, profile = clinician_ctx
    return await ClinicianService.get_document_detail(db, profile, document_id)


@router.post("/documents/upload")
async def upload_clinical_document(
    file: UploadFile = File(...),
    patient_id: Optional[uuid.UUID] = Form(None),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Upload a medical document to protected storage.
    
    Validates file format, magic bytes, MIME type, and clinician patient authorization.
    Returns safe metadata only. Zero raw text or PII returned.
    """
    user, profile = clinician_ctx
    content = await file.read()
    return await DocumentIngestionService.upload_document(
        db=db,
        user=user,
        profile=profile,
        filename=file.filename or "upload.bin",
        content_type=file.content_type or "application/octet-stream",
        content=content,
        patient_id=patient_id,
    )


@router.post("/documents/{document_id}/process")
async def process_clinical_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Execute document security check, text extraction, classification, masking, and downstream AI gating.
    
    Reaches DOCUMENT_READY_FOR_AI upon successful privacy validation.
    Returns safe processing metadata and aggregate entity counts only.
    """
    user, profile = clinician_ctx
    return await DocumentIngestionService.process_document(
        db=db,
        user=user,
        profile=profile,
        document_id=document_id,
    )


@router.get("/questions")
async def get_clinician_question_inbox(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Inbox of patient questions for explicitly assigned patients."""
    _, profile = clinician_ctx
    if not profile:
        return {"status": "NO_DATA", "questions": [], "limit": limit, "offset": offset}
    return await PatientQuestionService.list_clinician_questions(
        db, profile, limit=limit, offset=offset
    )


@router.patch("/questions/{question_id}")
async def update_clinician_question(
    question_id: uuid.UUID,
    body: ClinicianQuestionResponseRequest,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Mark a patient question reviewed and/or submit a clinician response."""
    _, profile = clinician_ctx
    if not profile:
        from backend.app.core.errors import ForbiddenError

        raise ForbiddenError("Active Clinician Profile required.")
    return await PatientQuestionService.clinician_update_question(
        db,
        profile,
        question_id,
        mark_reviewed=body.mark_reviewed,
        response_text=body.response_text,
    )


@router.get("/diagnosis")
async def get_diagnosis_predictions(
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve genuine diagnosis prediction records for assigned patients.
    
    Never fabricates predictions. Returns NOT_IMPLEMENTED capability indicator.
    """
    _, profile = clinician_ctx
    return await ClinicianService.get_diagnosis(db, profile)


@router.get("/diagnosis/{patient_id}")
async def get_patient_diagnosis(
    patient: Patient = Depends(get_authorized_clinician_patient),
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve diagnosis prediction for an authorized patient reference.
    
    Returns HTTP 404 if patient is unassigned. Returns NO_DATA / NOT_IMPLEMENTED if no prediction exists.
    """
    _, profile = clinician_ctx
    return await ClinicianService.get_patient_diagnosis(db, profile, patient)


@router.get("/explanations/{prediction_id}")
async def get_prediction_explanation(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve DiCE counterfactual explanations for a genuine diagnosis prediction.
    
    Returns NOT_IMPLEMENTED because DiCE is offline.
    """
    _, profile = clinician_ctx
    return await ClinicianService.get_prediction_explanation(db, profile, prediction_id)


@router.get("/causal/{prediction_id}")
async def get_prediction_causal(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Dict[str, Any]:
    """Retrieve DoWhy causal analyses for a genuine diagnosis prediction.
    
    Returns NOT_IMPLEMENTED because DoWhy is offline.
    """
    _, profile = clinician_ctx
    return await ClinicianService.get_prediction_causal(db, profile, prediction_id)

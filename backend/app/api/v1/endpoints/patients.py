"""Patients API v1 Endpoints."""

import uuid
from fastapi import APIRouter, Depends, status
from backend.app.api.dependencies import get_current_user, verify_patient_access
from backend.app.core.errors import FeatureNotImplementedError
from backend.app.models.identity import User

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def list_patients(current_user: User = Depends(get_current_user)):
    """List opaque patient references. Skeleton returns 501 Not Implemented."""
    raise FeatureNotImplementedError("Patient directory API not implemented")


@router.get("/{patient_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_patient(patient_id: uuid.UUID, current_user: User = Depends(verify_patient_access)):
    """Get patient reference details. Skeleton returns 501 Not Implemented."""
    raise FeatureNotImplementedError("Patient details API not implemented")

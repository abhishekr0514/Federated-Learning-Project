"""Hospitals API v1 Endpoints."""

import uuid
from fastapi import APIRouter, Depends, status
from backend.app.api.dependencies import get_current_user, verify_hospital_access
from backend.app.core.errors import FeatureNotImplementedError
from backend.app.models.identity import User

router = APIRouter(prefix="/hospitals", tags=["Hospitals"])


@router.get("/", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def list_hospitals(current_user: User = Depends(get_current_user)):
    """List hospitals. Skeleton returns 501 Not Implemented."""
    raise FeatureNotImplementedError("Hospital management API not implemented")


@router.get("/{hospital_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_hospital(hospital_id: uuid.UUID, current_user: User = Depends(verify_hospital_access)):
    """Get hospital node details. Skeleton returns 501 Not Implemented."""
    raise FeatureNotImplementedError("Hospital details API not implemented")

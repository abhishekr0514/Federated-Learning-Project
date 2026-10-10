"""Audit log API v1 Endpoints."""

from fastapi import APIRouter, Depends, status
from backend.app.api.dependencies import require_role
from backend.app.core.errors import FeatureNotImplementedError
from backend.app.core.security import UserRole
from backend.app.models.identity import User

router = APIRouter(prefix="/audit", tags=["Audit Logs"])


@router.get("/", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def list_audit_events(current_user: User = Depends(require_role(UserRole.SUPER_ADMIN, UserRole.HOSPITAL_ADMIN))):
    """Retrieve security audit events log. Skeleton returns 501 Not Implemented."""
    raise FeatureNotImplementedError("Audit log retrieval API not implemented")

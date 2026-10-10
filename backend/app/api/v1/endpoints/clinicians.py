"""Clinician workflow router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/clinicians", tags=["Clinicians"])


@router.get("/dashboard")
async def clinician_dashboard():
    """Retrieve clinician dashboard data."""
    raise FeatureNotImplementedError("Clinician portal workflow pending implementation.")

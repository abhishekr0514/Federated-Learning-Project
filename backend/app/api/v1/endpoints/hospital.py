"""Hospital Admin Command Center and Node Observability Endpoints.

All endpoints strictly enforce server-side hospital scoping via authorized memberships.
Unauthorized cross-tenant access returns HTTP 404 to prevent resource enumeration.
Zero Mock Data contract enforced: missing records return explicit NO_DATA / NOT_ENABLED states.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.api.dependencies import (
    get_db,
    get_current_user,
    require_role,
    get_authorized_hospital_context,
)
from backend.app.core.security import UserRole
from backend.app.models.identity import Hospital, HospitalMembership, User
from backend.app.services.hospital_service import HospitalService

router = APIRouter(
    prefix="/hospital",
    tags=["Hospital Admin Command Center"],
    dependencies=[Depends(require_role(UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN))],
)


@router.get(
    "/accessible-nodes",
    status_code=status.HTTP_200_OK,
    summary="List Accessible Hospital Nodes for Authenticated User",
)
async def get_accessible_nodes(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all hospital nodes the authenticated administrator has membership in."""
    if current_user.is_super_admin:
        stmt = select(Hospital).where(Hospital.is_active.isnot(False))
        res = await db.execute(stmt)
        hospitals = res.scalars().all()
    else:
        stmt = (
            select(Hospital)
            .join(HospitalMembership, Hospital.id == HospitalMembership.hospital_id)
            .where(HospitalMembership.user_id == current_user.id, Hospital.is_active.isnot(False))
        )
        res = await db.execute(stmt)
        hospitals = res.scalars().all()

    nodes_list = [
        {
            "id": str(h.id),
            "name": h.name,
            "code": h.code,
            "is_active": h.is_active,
        }
        for h in hospitals
    ]

    return {
        "status": "AVAILABLE" if nodes_list else "NO_DATA",
        "count": len(nodes_list),
        "hospitals": nodes_list,
    }


@router.get(
    "/overview",
    status_code=status.HTTP_200_OK,
    summary="Hospital Node Overview Telemetry",
)
async def get_hospital_overview(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve operational overview metrics for the authorized hospital node."""
    return await HospitalService.get_overview(db, hospital)


@router.get(
    "/system-health",
    status_code=status.HTTP_200_OK,
    summary="Hospital Node System Health & Connectivity",
)
async def get_hospital_system_health(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve system health and database connectivity for the authorized hospital."""
    return await HospitalService.get_system_health(db, hospital)


@router.get(
    "/clinicians",
    status_code=status.HTTP_200_OK,
    summary="Hospital-Scoped Clinician Directory",
)
async def get_hospital_clinicians(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve paginated clinician profiles authorized for this hospital node."""
    return await HospitalService.get_clinicians(db, hospital, limit=limit, offset=offset)


@router.get(
    "/patients",
    status_code=status.HTTP_200_OK,
    summary="Opaque Patient Reference Index",
)
async def get_hospital_patients(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve paginated opaque patient reference hashes. Zero PII/PHI exposed."""
    return await HospitalService.get_patients(db, hospital, limit=limit, offset=offset)


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
    summary="Local Model Architectures and Checkpoint Visibility",
)
async def get_hospital_models(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve registered models and local training status for the hospital node."""
    return await HospitalService.get_models(db, hospital)


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Local Performance Evaluation Metrics",
)
async def get_hospital_metrics(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve local performance evaluation metrics with declared task targets."""
    return await HospitalService.get_metrics(db, hospital)


@router.get(
    "/federated",
    status_code=status.HTTP_200_OK,
    summary="Hospital Federated Node Participation Status",
)
async def get_hospital_federated(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve federated client registration and historical round participation."""
    return await HospitalService.get_federated(db, hospital)


@router.get(
    "/privacy",
    status_code=status.HTTP_200_OK,
    summary="Local Privacy Sanitization Metrics",
)
async def get_hospital_privacy(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve privacy sanitization and entity redaction telemetry."""
    return await HospitalService.get_privacy(db, hospital)


@router.get(
    "/data-quality",
    status_code=status.HTTP_200_OK,
    summary="Hospital Data Quality Telemetry",
)
async def get_hospital_data_quality(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve hospital-scoped data quality events (zero clinical text)."""
    return await HospitalService.get_data_quality(db, hospital)


@router.get(
    "/audit",
    status_code=status.HTTP_200_OK,
    summary="Hospital-Scoped Append-Only Audit Log",
)
async def get_hospital_audit(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve paginated immutable audit log events scoped strictly to this hospital."""
    return await HospitalService.get_audit(db, hospital, limit=limit, offset=offset)


@router.get(
    "/alerts",
    status_code=status.HTTP_200_OK,
    summary="Hospital-Scoped Operational Alerts",
)
async def get_hospital_alerts(
    db: AsyncSession = Depends(get_db),
    hospital: Hospital = Depends(get_authorized_hospital_context),
) -> Dict[str, Any]:
    """Retrieve active and historical operational alerts for this hospital node."""
    return await HospitalService.get_alerts(db, hospital)

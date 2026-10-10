from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.api.dependencies import get_db, require_role, require_permission
from backend.app.core.security import UserRole, SystemPermission
from backend.app.schemas.common import GenericSuccessEnvelope, RequestMetadata
from backend.app.services.admin_service import AdminService

router = APIRouter(
    prefix="/admin",
    tags=["Admin Command Center"],
    dependencies=[
        Depends(require_role(UserRole.SUPER_ADMIN)),
        Depends(require_permission(SystemPermission.GLOBAL_ADMIN)),
    ],
)

@router.get("/overview", response_model=GenericSuccessEnvelope)
async def get_admin_overview(db: AsyncSession = Depends(get_db)):
    """Retrieve global Super Admin Command Center overview metrics."""
    data = await AdminService.get_overview_summary(db)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Super Admin overview retrieved successfully.")
    )

@router.get("/system-health", response_model=GenericSuccessEnvelope)
async def get_system_health(db: AsyncSession = Depends(get_db)):
    """Retrieve API latency, DB connectivity status, and hardware monitoring flags."""
    data = await AdminService.get_system_health(db)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="System health metrics retrieved successfully.")
    )

@router.get("/hospitals", response_model=GenericSuccessEnvelope)
async def get_admin_hospitals(db: AsyncSession = Depends(get_db)):
    """Retrieve hospital nodes registry overview."""
    data = await AdminService.get_hospitals_overview(db)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Hospital registry overview retrieved successfully.")
    )

@router.get("/federated", response_model=GenericSuccessEnvelope)
async def get_admin_federated():
    """Retrieve federated Flower training state."""
    data = await AdminService.get_federated_status()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Federated status retrieved successfully.")
    )

@router.get("/models", response_model=GenericSuccessEnvelope)
async def get_admin_models():
    """Retrieve global model version artifacts."""
    data = await AdminService.get_models_overview()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Model artifacts overview retrieved successfully.")
    )

@router.get("/metrics", response_model=GenericSuccessEnvelope)
async def get_admin_metrics():
    """Retrieve model performance metrics."""
    data = await AdminService.get_metrics_overview()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Model performance metrics retrieved successfully.")
    )

@router.get("/privacy", response_model=GenericSuccessEnvelope)
async def get_admin_privacy():
    """Retrieve privacy masking statistics."""
    data = await AdminService.get_privacy_metrics()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Privacy metrics retrieved successfully.")
    )

@router.get("/security", response_model=GenericSuccessEnvelope)
async def get_admin_security(db: AsyncSession = Depends(get_db)):
    """Retrieve aggregated security event telemetry."""
    data = await AdminService.get_security_metrics(db)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Security telemetry retrieved successfully.")
    )

@router.get("/audit", response_model=GenericSuccessEnvelope)
async def get_admin_audit(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve paginated append-only security audit log entries."""
    data = await AdminService.get_audit_ledger(db, limit=limit, offset=offset)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Audit log ledger retrieved successfully.")
    )

@router.get("/alerts", response_model=GenericSuccessEnvelope)
async def get_admin_alerts():
    """Retrieve active system alerts."""
    data = await AdminService.get_alerts_overview()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="System alerts retrieved successfully.")
    )

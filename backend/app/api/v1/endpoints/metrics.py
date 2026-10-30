"""System performance and evaluation metrics."""

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import require_permission
from backend.app.core.security import SystemPermission
from backend.app.schemas.common import GenericSuccessEnvelope, RequestMetadata
from backend.app.services.offline_evaluation_metrics_service import (
    load_offline_gat_evaluation_summary,
)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/evaluation", response_model=GenericSuccessEnvelope)
async def get_evaluation_metrics(
    _perm=Depends(require_permission(SystemPermission.GLOBAL_METRICS_READ)),
):
    """Empirical offline GAT evaluation from reports/model_evaluation/ (no fabricated scores)."""
    data = load_offline_gat_evaluation_summary()
    msg = (
        "Offline evaluation summary loaded from disk."
        if data.get("status") == "AVAILABLE"
        else data.get("message", "Evaluation report unavailable.")
    )
    return GenericSuccessEnvelope(data=data, metadata=RequestMetadata(message=msg))

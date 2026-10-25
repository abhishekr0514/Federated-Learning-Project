"""Federated learning read-only experiment status (no campaign orchestration)."""

from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import get_current_user, require_permission
from backend.app.core.errors import FeatureNotImplementedError
from backend.app.core.security import SystemPermission
from backend.app.models.identity import User
from backend.app.schemas.common import GenericSuccessEnvelope, RequestMetadata
from backend.app.services.federated_experiment_service import get_tabular_experiment_status

router = APIRouter(prefix="/federated", tags=["Federated"])


@router.get("/rounds")
async def list_fl_rounds():
    """List historical federated training rounds."""
    raise FeatureNotImplementedError("Flower federated learning orchestration pending FL phase implementation.")


@router.post("/start")
async def start_fl_campaign():
    """Initiate a new federated training campaign."""
    raise FeatureNotImplementedError("FL campaign trigger pending implementation.")


@router.get("/experiments/tabular/latest", response_model=GenericSuccessEnvelope)
async def get_latest_tabular_federated_experiment(
    run_id: str | None = Query(None, description="Optional run id under tabular_cardio_runs/"),
    _user: User = Depends(get_current_user),
    __perm: User = Depends(require_permission(SystemPermission.FEDERATED_MONITOR)),
):
    """
    Read sanitized summary of the latest saved tabular FL experiment.

    Does not expose raw rows, record ids, or per-client hash lists.
    Does not start training or expose a public FL server.
    """
    data = get_tabular_experiment_status(run_id=run_id)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Tabular federated experiment summary retrieved."),
    )

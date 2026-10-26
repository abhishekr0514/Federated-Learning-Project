"""Federated learning status and gated synthetic orchestration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from backend.app.api.dependencies import get_current_user, require_permission
from backend.app.core.errors import ValidationAppError
from backend.app.core.security import SystemPermission
from backend.app.models.identity import User
from backend.app.schemas.common import GenericSuccessEnvelope, RequestMetadata
from backend.app.services.federated_experiment_service import get_tabular_experiment_status
from backend.app.services.gat_federated_experiment_service import get_gat_federated_experiment_status
from backend.app.services.flower_orchestration_service import FlowerOrchestrationService

router = APIRouter(prefix="/federated", tags=["Federated"])


class FederatedStartRequest(BaseModel):
    num_rounds: Optional[int] = Field(default=None, ge=1, le=20)


@router.get("/status", response_model=GenericSuccessEnvelope)
async def get_federated_orchestration_status(
    _user: User = Depends(get_current_user),
    __perm: User = Depends(require_permission(SystemPermission.FEDERATED_MONITOR)),
) -> GenericSuccessEnvelope:
    data = FlowerOrchestrationService.get_status()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Federated orchestration status retrieved."),
    )


@router.get("/rounds")
async def list_fl_rounds(
    _user: User = Depends(get_current_user),
    __perm: User = Depends(require_permission(SystemPermission.FEDERATED_MONITOR)),
) -> Dict[str, Any]:
    status_payload = FlowerOrchestrationService.get_status()
    return {
        "rounds": status_payload.get("history") or [],
        "lifecycle": status_payload.get("lifecycle"),
        "run_id": status_payload.get("run_id"),
    }


@router.post("/start", response_model=GenericSuccessEnvelope, status_code=status.HTTP_202_ACCEPTED)
async def start_fl_campaign(
    body: FederatedStartRequest | None = None,
    _user: User = Depends(get_current_user),
    __perm: User = Depends(require_permission(SystemPermission.FEDERATED_MONITOR)),
) -> GenericSuccessEnvelope:
    try:
        data = FlowerOrchestrationService.start_run(
            num_rounds=body.num_rounds if body else None,
        )
    except RuntimeError as exc:
        raise ValidationAppError(str(exc)) from exc
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Federated GAT research run started."),
    )


@router.post("/stop", response_model=GenericSuccessEnvelope)
async def stop_fl_campaign(
    _user: User = Depends(get_current_user),
    __perm: User = Depends(require_permission(SystemPermission.FEDERATED_MONITOR)),
) -> GenericSuccessEnvelope:
    data = FlowerOrchestrationService.stop_run()
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="Federated run stop requested."),
    )


@router.get("/experiments/gat/latest", response_model=GenericSuccessEnvelope)
async def get_latest_gat_federated_experiment(
    run_id: str | None = Query(None, description="Optional Flower run id under gat_ecg_report_runs/"),
    _user: User = Depends(get_current_user),
    __perm: User = Depends(require_permission(SystemPermission.FEDERATED_MONITOR)),
):
    """Sanitized summary of a completed GAT federated Flower run (held-out test AUROC when available)."""
    data = get_gat_federated_experiment_status(run_id=run_id)
    return GenericSuccessEnvelope(
        data=data,
        metadata=RequestMetadata(message="GAT federated experiment summary retrieved."),
    )


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

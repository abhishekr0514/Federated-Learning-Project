"""Two-modality GAT fusion (ECG + clinical report embeddings). Research proxy only."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, status

from backend.app.api.dependencies import (
    get_current_user,
    require_role,
)
from backend.app.core.security import UserRole
from backend.app.models.identity import User
from backend.app.schemas.gat_fusion import (
    GATFusionEvaluateRequest,
    GATFusionEvaluateResponse,
    GATFusionReadinessResponse,
)
from backend.app.services.gat_fusion_service import GATFusionService

router = APIRouter(prefix="/gat-fusion", tags=["GAT Fusion Inference"])


@router.get(
    "/readiness",
    response_model=GATFusionReadinessResponse,
    summary="GAT fusion pipeline readiness diagnostics",
)
async def get_gat_fusion_readiness() -> Dict[str, Any]:
    """
    Return readiness for two-modality GAT (ECG 128-d + clinical text 128-d).
    - Fusion checkpoint presence and trained status
    - PyTorch and torch-geometric availability

    Does not expose filesystem paths, credentials, or internal model weights.
    """
    return GATFusionService.get_readiness()


@router.post(
    "/evaluate",
    response_model=GATFusionEvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Two-modality GAT fusion proxy evaluation",
)
async def evaluate_gat_fusion(
    body: GATFusionEvaluateRequest,
    patient_id: Optional[uuid.UUID] = Query(
        None,
        description="Optional patient UUID for provenance annotation (no DB write performed by this endpoint).",
    ),
    current_user: User = Depends(
        require_role(UserRole.CLINICIAN, UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN)
    ),
) -> GATFusionEvaluateResponse:
    """
    Requires both 128-d embeddings from the authorized ECG-report encoder stack.
    Enabled only when FUSION_INFERENCE_ENABLED=true and a trained GAT checkpoint is deployed.
    """
    return await GATFusionService.evaluate(
        body=body,
        patient_id=patient_id,
    )

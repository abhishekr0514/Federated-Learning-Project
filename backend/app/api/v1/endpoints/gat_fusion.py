"""Two-modality GAT fusion (ECG + clinical report embeddings). Research proxy only."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import (
    get_db,
    require_role,
)
from backend.app.core.security import UserRole
from backend.app.models.identity import User
from backend.app.services.patient_inference_auth import authorize_patient_linked_inference
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
        description="Optional assigned patient UUID; persists DiagnosisPrediction when authorized.",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(
        require_role(UserRole.CLINICIAN, UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN)
    ),
) -> GATFusionEvaluateResponse:
    """
    Requires both 128-d embeddings from the authorized ECG-report encoder stack.
    Enabled only when FUSION_INFERENCE_ENABLED=true and a trained GAT checkpoint is deployed.

    - With patient_id: role-checked persistence (super admin, hospital admin scope, clinician assignment).
    - Without patient_id: research-only proxy evaluation; no patient health record write.
    """
    target_patient = None
    target_profile = None
    persist_hospital_id = None

    if patient_id is not None:
        ctx = await authorize_patient_linked_inference(db, current_user, patient_id)
        target_patient = ctx.patient
        target_profile = ctx.profile
        persist_hospital_id = ctx.hospital_id

    result = await GATFusionService.evaluate(
        db,
        body=body,
        profile=target_profile,
        patient=target_patient,
        hospital_id=persist_hospital_id,
    )
    if target_patient:
        await db.commit()
    return result

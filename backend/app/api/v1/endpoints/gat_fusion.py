"""FastAPI endpoints for three-modality GAT fusion proxy inference.

Architecture: medshield.diagnostic_net.gat_fusion
Research proxy task only. Not validated for clinical use.
"""

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
    Return safe readiness diagnostics for the three-modality GAT fusion pipeline.

    Reports:
    - Data linkage gate verdict (BLOCKED / CONDITIONAL_GO)
    - Embedding dimension contracts (ECG: 128-d, text: 128-d, tabular: 64-d)
    - Fusion checkpoint presence and trained status
    - PyTorch and torch-geometric availability

    Does not expose filesystem paths, credentials, or internal model weights.
    """
    return GATFusionService.get_readiness()


@router.post(
    "/evaluate",
    response_model=GATFusionEvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Three-modality GAT fusion proxy evaluation",
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
    Run a three-modality GAT (Graph Attention Network) fusion forward pass.

    Inputs are **pre-computed embedding vectors** extracted by the upstream
    single-modality encoders:

    | Modality      | Dimension | Encoder                                |
    |---------------|-----------|----------------------------------------|
    | ECG           | 128       | ECGEmbeddingProvider                   |
    | Clinical text | 128       | ClinicalTextEncodingPipeline (BERT CLS)|
    | Tabular       | 64        | NeuralTabularEncoder                   |

    At least one modality must be provided. Missing modalities are zero-padded
    and masked inside `MedShieldDiagnosticNet`.

    **This endpoint is BLOCKED in production until:**
    1. A verified multimodal dataset with aligned ECG + text + tabular patient
       records is obtained (e.g. MIMIC-IV-ECG + MIMIC-IV).
    2. Supervised three-modality GAT training is executed and verified.
    3. The trained checkpoint is placed in FUSION_MODEL_DIR with a valid manifest.

    Returns an uncalibrated research proxy score. Not for clinical use.
    """
    return await GATFusionService.evaluate(
        body=body,
        patient_id=patient_id,
    )

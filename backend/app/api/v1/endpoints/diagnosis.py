"""Multimodal clinical diagnosis router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


@router.post("/evaluate")
async def evaluate_multimodal_diagnosis():
    """Evaluate multimodal inputs using GAT fusion network.
    
    This endpoint is intentionally disabled until actual GAT neural model training
    and evaluation are completed. Fabricating prediction probabilities or running legacy formulas is strictly forbidden.
    """
    raise FeatureNotImplementedError(
        "Multimodal GAT inference is pending model training and verification. "
        "Legacy prediction formulas are disabled to prevent scientific misrepresentation."
    )

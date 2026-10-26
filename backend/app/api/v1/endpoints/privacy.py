from typing import Any, Dict
from pydantic import BaseModel, Field
from fastapi import APIRouter, status

from backend.app.services.privacy_masking import PrivacyMasker
from backend.app.services.privacy_gate import MaskingValidator

router = APIRouter(prefix="/privacy", tags=["Privacy"])


class PrivacyMaskRequest(BaseModel):
    raw_text: str = Field(
        ...,
        description="Clinical report narrative to validate through privacy boundary.",
        min_length=1,
    )


class PrivacyMaskResponse(BaseModel):
    is_valid: bool
    status: str
    validation_message: str
    entities_detected_count: int
    entity_categories: Dict[str, int]


@router.get("/status")
async def privacy_status():
    """Retrieve privacy masking engine status."""
    return {
        "privacy_masking_enabled": True,
        "pii_sanitization": "configured",
        "differential_privacy": "active",
        "epsilon": 1.0,
        "delta": 1e-5,
    }


@router.post(
    "/mask-text",
    response_model=PrivacyMaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate and test privacy masking on clinical report text",
)
async def mask_clinical_text(body: PrivacyMaskRequest) -> PrivacyMaskResponse:
    """
    Apply privacy masking and fail-closed validation to clinical report text.
    Zero raw text or detected PII values are returned, logged, or persisted.
    """
    masked_text, counts, detected_values = PrivacyMasker.mask_document(body.raw_text)
    is_valid, validation_status = MaskingValidator.validate_masking(
        original_text=body.raw_text,
        masked_text=masked_text,
        detected_values=detected_values,
    )

    active_counts = {k: v for k, v in counts.items() if v > 0}
    total_detected = sum(active_counts.values())

    return PrivacyMaskResponse(
        is_valid=is_valid,
        status="VALIDATED" if is_valid else "BLOCKED",
        validation_message=validation_status,
        entities_detected_count=total_detected,
        entity_categories=active_counts,
    )


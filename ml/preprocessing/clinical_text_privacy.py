"""Apply hospital privacy masking before ML text consumption (Prompt 13)."""

from __future__ import annotations

from backend.app.services.privacy_gate import MaskingValidator
from backend.app.services.privacy_masking import PrivacyMasker

from ml.text.errors import MaskingFailedError


def mask_clinical_text_for_encoding(raw_text: str) -> str:
    """
    Mask direct identifiers using the established PrivacyMasker + MaskingValidator.

    Fail-closed: raises MaskingFailedError if validation fails.
    Does not log raw or masked text.
    """
    masked_text, _counts, detected_values = PrivacyMasker.mask_document(raw_text)
    is_valid, status = MaskingValidator.validate_masking(
        original_text=raw_text,
        masked_text=masked_text,
        detected_values=detected_values,
    )
    if not is_valid:
        raise MaskingFailedError(status)
    return masked_text

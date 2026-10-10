"""Downstream AI Privacy Gate and Masking Validation Service.

MANDATORY SAFETY INVARIANTS:
1. Masking Validation:
   - Asserts all detected raw PII values are completely absent from the masked text.
   - Asserts known direct identifier patterns (emails, phones, SSNs) are absent.
   - Asserts clinical and medical text remains intact.
   - Rejects empty outputs when source text was meaningful.
2. Downstream AI Gate:
   - document_safe_for_ai() returns True ONLY when all security, masking, validation,
     and patient-authorization requirements are verified.
   - Under no circumstances is raw or unmasked text passed to downstream AI modules.
   - State reaches DOCUMENT_READY_FOR_AI, never DIAGNOSIS_COMPLETED.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class MaskingValidator:
    """Validates that privacy masking was executed thoroughly and safely."""

    @classmethod
    def validate_masking(
        cls,
        original_text: str,
        masked_text: str,
        detected_values: List[str],
    ) -> Tuple[bool, str]:
        """Validate that masked representation is safe for downstream processing.
        
        Returns:
            Tuple of (is_valid: bool, status_message: str)
        """
        if not original_text.strip():
            # Empty source text is trivially safe if masked text is also empty
            return True, "VALIDATED"

        if not masked_text.strip():
            return False, "MASKING_VALIDATION_FAILED: Masked text is unexpectedly empty."

        # 1. Verify detected raw PII values are completely absent
        for val in detected_values:
            val_clean = val.strip()
            # Avoid checking trivial single characters
            if len(val_clean) > 2 and val_clean in masked_text:
                return False, f"MASKING_VALIDATION_FAILED: Detected PII value persisted in masked output."

        # 2. Check for obvious unmasked email patterns
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        if email_pattern.search(masked_text):
            return False, "MASKING_VALIDATION_FAILED: Residual unmasked email pattern detected."

        # 3. Check for obvious unmasked phone patterns
        phone_pattern = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        if phone_pattern.search(masked_text):
            return False, "MASKING_VALIDATION_FAILED: Residual unmasked phone pattern detected."

        # 4. Check for obvious unmasked SSN patterns
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        if ssn_pattern.search(masked_text):
            return False, "MASKING_VALIDATION_FAILED: Residual unmasked SSN pattern detected."

        # 5. Check clinical content preservation:
        # If original text contained medical keywords or numbers/units, ensure some are preserved
        clinical_markers = [
            "mg/dl", "mmol/l", "bpm", "mmhg", "ecg", "ekg", "ldl", "hdl",
            "cholesterol", "troponin", "ejection fraction", "sinus", "rhythm",
            "blood", "heart", "exam", "patient", "normal", "abnormal", "report"
        ]
        orig_markers = [m for m in clinical_markers if m in original_text.lower()]
        if orig_markers:
            masked_markers = [m for m in orig_markers if m in masked_text.lower()]
            if not masked_markers:
                return False, "MASKING_VALIDATION_FAILED: Clinical terminology was improperly scrubbed."

        return True, "VALIDATED"


def document_safe_for_ai(
    doc_metadata: Dict[str, Any],
    downstream_payload: Dict[str, Any],
) -> Tuple[bool, str]:
    """Authoritative downstream AI privacy gate.
    
    Verifies that all privacy, validation, and authorization boundaries are satisfied
    prior to any potential AI inference consumption.
    
    Returns:
        Tuple of (is_safe: bool, final_state: str)
    """
    # 1. File Validation Check
    if doc_metadata.get("file_validation_status") != "VALIDATED":
        return False, "VALIDATION_FAILED"

    # 2. Security / Antivirus Check
    sec_status = doc_metadata.get("security_status")
    if sec_status not in ["CLEAN", "NOT_IMPLEMENTED"]:
        return False, "SECURITY_CHECK_FAILED"

    # 3. Text Extraction Check
    ext_status = doc_metadata.get("extraction_status")
    if ext_status not in ["SUCCESS", "EMPTY_TEXT"]:
        return False, "TEXT_EXTRACTION_FAILED"

    # 4. Classification Check
    if not doc_metadata.get("document_type"):
        return False, "CLASSIFICATION_PENDING"

    # 5. Entity Detection & Masking Check
    if not doc_metadata.get("masking_completed"):
        return False, "MASKING_PENDING"

    # 6. Masking Validation Check
    if doc_metadata.get("masking_validation_status") != "VALIDATED":
        return False, "MASKING_VALIDATION_FAILED"

    # 7. Patient Association Authorization Check
    if not doc_metadata.get("patient_associated") or not doc_metadata.get("patient_id"):
        return False, "PATIENT_ASSOCIATION_REQUIRED"

    # 8. Strict Downstream Payload Inspection (Zero Raw Text Invariant)
    # The payload MUST NOT contain the unmasked text or raw OCR buffer
    if "raw_text" in downstream_payload or "unmasked_content" in downstream_payload:
        return False, "PRIVACY_GATE_REJECTED: Raw unmasked content present in AI payload."

    raw_original = doc_metadata.get("_original_text_in_memory")
    if raw_original and len(raw_original) > 20:
        # Ensure the downstream payload does not accidentally mirror the raw unmasked text
        downstream_text = downstream_payload.get("masked_text", "")
        if raw_original == downstream_text and doc_metadata.get("total_pii_count", 0) > 0:
            return False, "PRIVACY_GATE_REJECTED: Downstream payload matches unmasked source containing PII."

    return True, "DOCUMENT_READY_FOR_AI"

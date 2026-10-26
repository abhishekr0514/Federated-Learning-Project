"""Opaque patient reference labels for clinician / diagnostic studio lists (no PII)."""

from __future__ import annotations

from typing import Any, Dict, Optional

DEMO_SYNTHETIC_OPAQUE_PREFIX = "DEMO-SYNTH-"


def is_demo_synthetic_opaque(opaque_patient_id: str) -> bool:
    return opaque_patient_id.startswith(DEMO_SYNTHETIC_OPAQUE_PREFIX)


def format_opaque_display_label(
    opaque_patient_id: str,
    *,
    hospital_code: Optional[str] = None,
) -> str:
    """Human-readable list label; full demo synthetic opaque id, truncated hash otherwise."""
    if is_demo_synthetic_opaque(opaque_patient_id):
        base = opaque_patient_id
    elif len(opaque_patient_id) > 36:
        base = f"{opaque_patient_id[:12]}…{opaque_patient_id[-8:]}"
    else:
        base = opaque_patient_id
    if hospital_code:
        return f"{base} · {hospital_code}"
    return base


def enrich_patient_list_entry(
    *,
    patient_id: str,
    opaque_patient_id: str,
    hospital_code: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Ensure list payloads expose stable UUID + display metadata for UI binding."""
    payload: Dict[str, Any] = {
        "patient_id": patient_id,
        "opaque_patient_id": opaque_patient_id,
        "is_demo_synthetic": is_demo_synthetic_opaque(opaque_patient_id),
        "display_label": format_opaque_display_label(
            opaque_patient_id, hospital_code=hospital_code
        ),
        **extra,
    }
    return payload

"""Privacy boundaries for FL messages (FedAvg is not differential privacy)."""

from __future__ import annotations

FORBIDDEN_MESSAGE_KEYS = frozenset(
    {
        "patient_id",
        "opaque_patient_id",
        "ecg_id",
        "raw_report",
        "clinical_note",
        "raw_ecg",
        "gradients",
        "raw_rows",
        "identifiers",
    }
)

ALLOWED_CLIENT_METRIC_KEYS = frozenset({"loss"})

PRIVACY_LIMITATIONS = (
    "MedShield-FL Prompt 20 infrastructure uses standard FedAvg parameter aggregation. "
    "This is NOT differential privacy, secure aggregation, or protection against model inversion. "
    "Only model parameters and explicitly allowlisted scalar metrics may leave clients. "
    "Raw clinical data must remain local."
)


def assert_safe_metadata_dict(metadata: dict, *, context: str) -> None:
    for key in metadata:
        lower = key.lower()
        if lower in FORBIDDEN_MESSAGE_KEYS:
            raise ValueError(f"{context}: forbidden metadata key {key!r}")
        if "password" in lower or "secret" in lower or "token" in lower or "api_key" in lower:
            raise ValueError(f"{context}: suspicious secret-like key {key!r}")

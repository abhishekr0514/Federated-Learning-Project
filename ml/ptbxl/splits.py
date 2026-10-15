"""Patient-independent split assignment for PTB-XL protocol v1."""

from __future__ import annotations

from typing import Iterable

SPLIT_POLICY_V1 = {
    "protocol_version": "1.0.0",
    "train_folds": list(range(1, 8)),
    "validation_folds": [8],
    "test_folds": [10],
    "unused_folds": [9],
    "selection_metric": "val_auroc",
    "test_fold_never_for_selection": True,
}


def assign_split(strat_fold: int) -> str | None:
    if strat_fold in SPLIT_POLICY_V1["train_folds"]:
        return "train"
    if strat_fold in SPLIT_POLICY_V1["validation_folds"]:
        return "validation"
    if strat_fold in SPLIT_POLICY_V1["test_folds"]:
        return "test"
    if strat_fold in SPLIT_POLICY_V1["unused_folds"]:
        return None
    raise ValueError(f"Unknown strat_fold: {strat_fold}")


def verify_patient_disjoint(records: Iterable[dict]) -> dict[str, bool | int]:
    """Ensure each patient_id maps to at most one split partition."""
    patient_to_split: dict[str, str] = {}
    conflicts = 0
    for rec in records:
        split = assign_split(rec["strat_fold"])
        if split is None:
            continue
        pid = str(rec["patient_id"])
        if pid in patient_to_split and patient_to_split[pid] != split:
            conflicts += 1
        else:
            patient_to_split[pid] = split
    return {
        "patient_disjoint": conflicts == 0,
        "conflict_count": conflicts,
        "patients_assigned": len(patient_to_split),
    }

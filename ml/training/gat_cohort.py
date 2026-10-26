"""Canonical PTB-XL ECG+report cohort splits for GAT vs federated comparisons."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from ml.ptbxl.ecg_report_index import build_ecg_report_index
from ml.training.train_ecg_report_fusion import partition_records


def canonical_ecg_report_partitions(
    ptbxl_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records, index_stats = build_ecg_report_index(ptbxl_root)
    return partition_records(records), index_stats


def sorted_ecg_ids(records: Iterable[dict[str, Any]]) -> list[int]:
    return sorted(int(r["ecg_id"]) for r in records)


def ecg_id_set_fingerprint(ecg_ids: Iterable[int]) -> str:
    payload = ",".join(str(i) for i in sorted(ecg_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_head_record_caps(
    parts: dict[str, list[dict[str, Any]]],
    *,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Apply optional head-of-list caps (legacy GAT used --max-train 1500)."""
    out = {k: list(v) for k, v in parts.items()}
    if max_train is not None:
        out["train"] = out["train"][:max_train]
    if max_val is not None:
        out["validation"] = out["validation"][:max_val]
    if max_test is not None:
        out["test"] = out["test"][:max_test]
    return out


def verify_test_cohort_against_reference(
    test_records: list[dict[str, Any]],
    reference_test_ecg_ids: frozenset[int],
) -> dict[str, Any]:
    actual = frozenset(int(r["ecg_id"]) for r in test_records)
    return {
        "match": actual == reference_test_ecg_ids,
        "test_count": len(actual),
        "reference_count": len(reference_test_ecg_ids),
        "test_ecg_id_fingerprint": ecg_id_set_fingerprint(actual),
        "reference_ecg_id_fingerprint": ecg_id_set_fingerprint(reference_test_ecg_ids),
        "missing_from_actual_count": len(reference_test_ecg_ids - actual),
        "extra_in_actual_count": len(actual - reference_test_ecg_ids),
    }


def fl_canonical_test_ecg_ids(ptbxl_root: Path) -> frozenset[int]:
    parts, _ = canonical_ecg_report_partitions(ptbxl_root)
    return frozenset(int(r["ecg_id"]) for r in parts["test"])

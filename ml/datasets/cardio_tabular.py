"""
cardio_train.csv adapter — tabular research only; no cross-dataset linkage.

Protocol: docs/research/TABULAR_BASELINE_PROTOCOL_v1.md
Schema: medshield.tabular.cardio_survey v1.0.0
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ml.tabular.errors import TabularRowValidationError, TabularSchemaError
from ml.tabular.feature_schema import (
    EXCLUDED_FROM_FEATURES,
    extract_feature_vector_from_row,
    validate_raw_record_columns,
)

CARDIO_COLUMNS = (
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
)

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment,misc]


@dataclass
class CardioSplitSummary:
    seed: int
    total_raw_rows: int
    rejected_invalid_bp: int
    accepted_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_pos: int
    train_neg: int
    val_pos: int
    val_neg: int
    test_pos: int
    test_neg: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_cardio_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise TabularSchemaError("Empty cardio CSV or missing header.")
        validate_raw_record_columns(reader.fieldnames)
        for row in reader:
            yield row


def row_to_features_and_target(row: dict[str, str]) -> tuple[dict[str, float], int, int]:
    record_id = int(row["id"])
    target = int(row["cardio"])
    if target not in (0, 1):
        raise TabularRowValidationError(f"Invalid cardio target value: {row['cardio']}")
    features = extract_feature_vector_from_row({k: row[k] for k in row})
    feature_names = set(features.keys())
    if feature_names & EXCLUDED_FROM_FEATURES:
        raise TabularSchemaError("ID or target leaked into feature dict.")
    return features, target, record_id


def load_and_filter_cardio_cohort(
    path: Path,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """
    Load cardio_train.csv, enforce unique IDs, and filter invalid BP.

    Returns:
        (accepted_records, rejected_bp_ids, total_rows)
        Each accepted record is a dict with keys:
          'record_id': int, 'features': dict[str, float], 'target': int
    """
    seen_ids: set[int] = set()
    accepted: list[dict[str, Any]] = []
    rejected_bp_ids: list[int] = []
    total = 0

    for row in read_cardio_rows(path):
        total += 1
        record_id = int(row["id"])
        if record_id in seen_ids:
            raise TabularSchemaError(f"Duplicate record id detected: {record_id}")
        seen_ids.add(record_id)

        try:
            feats, target, rid = row_to_features_and_target(row)
            accepted.append({"record_id": rid, "features": feats, "target": target})
        except TabularRowValidationError as exc:
            if "ap_lo greater than ap_hi" in str(exc):
                rejected_bp_ids.append(record_id)
            else:
                raise

    return accepted, rejected_bp_ids, total


def stratified_cardio_split(
    records: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[set[int], set[int], set[int]]:
    """
    Prompt 17 Stratified Splitting:
    - Unique record id.
    - Seed: 42.
    - Fractions: 0.70 train, 0.15 val, 0.15 test (remainder of stratum assigned to test).
    - Stratify by cardio within each class, then shuffle with seed 42.
    - Disjoint partitions; every accepted id appears in exactly one partition.
    """
    class0_ids = sorted([r["record_id"] for r in records if r["target"] == 0])
    class1_ids = sorted([r["record_id"] for r in records if r["target"] == 1])

    rng0 = random.Random(seed)
    rng0.shuffle(class0_ids)
    n0 = len(class0_ids)
    n0_train = int(n0 * train_frac)
    n0_val = int(n0 * val_frac)
    c0_train = set(class0_ids[:n0_train])
    c0_val = set(class0_ids[n0_train : n0_train + n0_val])
    c0_test = set(class0_ids[n0_train + n0_val :])

    rng1 = random.Random(seed)
    rng1.shuffle(class1_ids)
    n1 = len(class1_ids)
    n1_train = int(n1 * train_frac)
    n1_val = int(n1 * val_frac)
    c1_train = set(class1_ids[:n1_train])
    c1_val = set(class1_ids[n1_train : n1_train + n1_val])
    c1_test = set(class1_ids[n1_train + n1_val :])

    train_ids = c0_train | c1_train
    val_ids = c0_val | c1_val
    test_ids = c0_test | c1_test

    all_ids = set([r["record_id"] for r in records])
    assert train_ids.isdisjoint(val_ids), "Train and Val partitions must be disjoint"
    assert train_ids.isdisjoint(test_ids), "Train and Test partitions must be disjoint"
    assert val_ids.isdisjoint(test_ids), "Val and Test partitions must be disjoint"
    assert (train_ids | val_ids | test_ids) == all_ids, "All accepted IDs must be partitioned"

    return train_ids, val_ids, test_ids


def deterministic_id_split(
    record_ids: list[int],
    *,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[set[int], set[int], set[int]]:
    """Legacy unstratified ID split (preserved for backward compatibility)."""
    rng = random.Random(seed)
    ids = sorted(set(record_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    return train, val, test


def hash_record_id(record_id: int) -> str:
    """Compute SHA-256 of string record ID to avoid publishing raw IDs."""
    return hashlib.sha256(str(record_id).encode("utf-8")).hexdigest()


def save_split_manifest_hashed(
    path: Path,
    *,
    seed: int,
    total_raw: int,
    rejected_bp_ids: Sequence[int],
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
    train_pos: int,
    train_neg: int,
    val_pos: int,
    val_neg: int,
    test_pos: int,
    test_neg: int,
) -> None:
    """Persist hashed ID manifest under gitignored manifests path."""
    manifest = {
        "protocol_id": "medshield.tabular.cardio_survey.baseline",
        "protocol_version": "1.0.0",
        "seed": seed,
        "counts": {
            "total_raw_rows": total_raw,
            "rejected_invalid_bp_rows": len(rejected_bp_ids),
            "accepted_cohort_rows": len(train_ids) + len(val_ids) + len(test_ids),
            "train_rows": len(train_ids),
            "val_rows": len(val_ids),
            "test_rows": len(test_ids),
            "train_class_distribution": {"class_0": train_neg, "class_1": train_pos},
            "val_class_distribution": {"class_0": val_neg, "class_1": val_pos},
            "test_class_distribution": {"class_0": test_neg, "class_1": test_pos},
        },
        "hashed_partitions": {
            "train_id_hashes": sorted([hash_record_id(rid) for rid in train_ids]),
            "val_id_hashes": sorted([hash_record_id(rid) for rid in val_ids]),
            "test_id_hashes": sorted([hash_record_id(rid) for rid in test_ids]),
            "rejected_bp_id_hashes": sorted([hash_record_id(rid) for rid in rejected_bp_ids]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


class CardioTabularDataset(_TorchDataset):
    """PyTorch Dataset for preprocessed tabular features and binary target."""

    def __init__(self, features: Any, targets: Any) -> None:
        import numpy as np
        import torch

        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = torch.as_tensor(features, dtype=torch.float32)

        if isinstance(targets, np.ndarray):
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = torch.as_tensor(targets, dtype=torch.float32)

        if self.features.dim() != 2 or self.features.size(-1) != 10:
            raise ValueError(f"Expected features shape (N, 10), got {tuple(self.features.shape)}")
        if self.targets.dim() != 1 or self.targets.size(0) != self.features.size(0):
            raise ValueError(
                f"Targets length {self.targets.size(0)} does not match features {self.features.size(0)}"
            )

    def __len__(self) -> int:
        return self.targets.size(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "features": self.features[idx],
            "target": self.targets[idx],
        }

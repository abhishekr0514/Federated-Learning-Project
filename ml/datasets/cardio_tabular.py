"""
cardio_train.csv adapter — tabular research only; no cross-dataset linkage.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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


@dataclass
class CardioSplitManifest:
    seed: int
    train_ids: int
    val_ids: int
    test_ids: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_cardio_pos: int
    val_cardio_pos: int
    test_cardio_pos: int
    rejected_invalid_bp: int


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
    if "id" in EXCLUDED_FROM_FEATURES and "cardio" in EXCLUDED_FROM_FEATURES:
        pass
    features = extract_feature_vector_from_row({k: row[k] for k in row})
    feature_names = set(features.keys())
    if feature_names & EXCLUDED_FROM_FEATURES:
        raise TabularSchemaError("ID or target leaked into feature dict.")
    return features, target, record_id


def deterministic_id_split(
    record_ids: list[int],
    *,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[set[int], set[int], set[int]]:
    import random

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

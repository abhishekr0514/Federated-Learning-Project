"""
Reproducible simulated-client partitions for cardio_train.csv federated experiments.

These are disjoint slices of the protocol train split — NOT independent hospitals.
Val/test splits follow TABULAR_BASELINE_PROTOCOL_v1 unchanged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ml.datasets.cardio_tabular import (
    hash_record_id,
    load_and_filter_cardio_cohort,
    stratified_cardio_split,
)

EXPECTED_DATASET_SHA256 = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"


@dataclass(frozen=True)
class FederatedPartitionPlan:
    seed: int
    num_clients: int
    max_federated_train_rows: int | None
    client_partition_strategy: str
    simulation_disclaimer: str
    client_train_counts: tuple[int, ...]
    val_row_count: int
    test_row_count: int
    train_row_count_total: int
    client_train_id_hashes: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cap_stratified_train(
    train_records: Sequence[dict[str, Any]],
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(train_records) <= max_rows:
        return list(train_records)
    c0 = [r for r in train_records if r["target"] == 0]
    c1 = [r for r in train_records if r["target"] == 1]
    ratio = len(c1) / max(len(train_records), 1)
    n1 = int(round(max_rows * ratio))
    n0 = max_rows - n1
    rng0 = random.Random(seed + 1)
    rng1 = random.Random(seed + 2)
    rng0.shuffle(c0)
    rng1.shuffle(c1)
    capped = c0[:n0] + c1[:n1]
    if len(capped) < max_rows:
        remainder = [r for r in train_records if r not in capped]
        rng = random.Random(seed + 3)
        rng.shuffle(remainder)
        capped.extend(remainder[: max_rows - len(capped)])
    return capped[:max_rows]


def assign_stratified_round_robin_partitions(
    train_records: Sequence[dict[str, Any]],
    *,
    num_clients: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """
    Assign records to clients in round-robin order within each class (after shuffling).
    Improves per-client label balance vs contiguous record-id slices (simulation only).
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    c0 = [r for r in train_records if r["target"] == 0]
    c1 = [r for r in train_records if r["target"] == 1]
    rng0 = random.Random(seed + 10)
    rng1 = random.Random(seed + 11)
    rng0.shuffle(c0)
    rng1.shuffle(c1)
    parts: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for idx, rec in enumerate(c0):
        parts[idx % num_clients].append(rec)
    for idx, rec in enumerate(c1):
        parts[idx % num_clients].append(rec)
    return parts


def assign_client_partitions(
    train_records: Sequence[dict[str, Any]],
    *,
    num_clients: int,
) -> list[list[dict[str, Any]]]:
    """Disjoint contiguous chunks on sorted record_id (simulated silos)."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    ordered = sorted(train_records, key=lambda r: int(r["record_id"]))
    n = len(ordered)
    base = n // num_clients
    parts: list[list[dict[str, Any]]] = []
    start = 0
    for client_idx in range(num_clients):
        end = n if client_idx == num_clients - 1 else start + base
        parts.append(ordered[start:end])
        start = end
    return parts


def build_tabular_federated_partitions(
    data_path: Path,
    *,
    seed: int = 42,
    num_clients: int = 3,
    max_federated_train_rows: int | None = 2400,
    client_partition_strategy: str = "contiguous_record_id",
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], FederatedPartitionPlan]:
    accepted, _rejected, _total = load_and_filter_cardio_cohort(data_path)
    train_ids, val_ids, test_ids = stratified_cardio_split(accepted, seed=seed)

    train_all = [r for r in accepted if r["record_id"] in train_ids]
    val_records = [r for r in accepted if r["record_id"] in val_ids]
    test_records = [r for r in accepted if r["record_id"] in test_ids]

    if max_federated_train_rows is not None:
        train_fed = _cap_stratified_train(train_all, max_federated_train_rows, seed)
    else:
        train_fed = train_all

    if client_partition_strategy == "stratified_round_robin":
        client_parts = assign_stratified_round_robin_partitions(
            train_fed, num_clients=num_clients, seed=seed
        )
    elif client_partition_strategy == "contiguous_record_id":
        client_parts = assign_client_partitions(train_fed, num_clients=num_clients)
    else:
        raise ValueError(f"Unknown client_partition_strategy: {client_partition_strategy}")

    # Verify disjointness
    seen: set[int] = set()
    for part in client_parts:
        for rec in part:
            rid = int(rec["record_id"])
            if rid in seen:
                raise ValueError(f"Duplicate record across clients: {rid}")
            seen.add(rid)

    id_hashes = [
        tuple(sorted(hash_record_id(int(r["record_id"])) for r in part)) for part in client_parts
    ]
    plan = FederatedPartitionPlan(
        seed=seed,
        num_clients=num_clients,
        max_federated_train_rows=max_federated_train_rows,
        client_partition_strategy=client_partition_strategy,
        simulation_disclaimer=(
            "Simulated Flower clients are disjoint record-id slices of the frozen protocol "
            "train split. They do NOT represent verified independent hospitals."
        ),
        client_train_counts=tuple(len(p) for p in client_parts),
        val_row_count=len(val_records),
        test_row_count=len(test_records),
        train_row_count_total=len(train_fed),
        client_train_id_hashes=id_hashes,
    )
    return client_parts, val_records, test_records, plan


def audit_partition_data(
    client_parts: list[list[dict[str, Any]]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Leakage/disjointness and label balance diagnostics (no raw ids in API exports)."""

    def _label_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"n": 0, "positive": 0, "negative": 0, "positive_rate": float("nan")}
        y = [int(r["target"]) for r in records]
        pos = sum(y)
        n = len(y)
        return {
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": float(pos / n) if n else float("nan"),
        }

    train_ids: set[int] = set()
    for idx, part in enumerate(client_parts):
        for rec in part:
            rid = int(rec["record_id"])
            if rid in train_ids:
                raise ValueError(f"Duplicate train record_id across clients: {rid}")
            train_ids.add(rid)

    val_ids = {int(r["record_id"]) for r in val_records}
    test_ids = {int(r["record_id"]) for r in test_records}
    overlap_vt = val_ids & test_ids
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids

    client_stats = []
    for idx, part in enumerate(client_parts):
        st = _label_stats(part)
        st["client_index"] = idx
        client_stats.append(st)

    pos_rates = [c["positive_rate"] for c in client_stats if c["n"]]
    imbalance = float(max(pos_rates) - min(pos_rates)) if len(pos_rates) >= 2 else 0.0

    return {
        "train_disjoint_across_clients": True,
        "train_val_overlap_count": len(overlap_tv),
        "train_test_overlap_count": len(overlap_tt),
        "val_test_overlap_count": len(overlap_vt),
        "split_leakage_detected": bool(overlap_vt or overlap_tv or overlap_tt),
        "train_label_stats": _label_stats([r for p in client_parts for r in p]),
        "val_label_stats": _label_stats(val_records),
        "test_label_stats": _label_stats(test_records),
        "client_label_stats": client_stats,
        "client_positive_rate_spread": imbalance,
    }


def save_partition_plan(path: Path, plan: FederatedPartitionPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)

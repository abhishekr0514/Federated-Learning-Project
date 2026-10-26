"""Simulated-client partitions for PTB-XL ECG + report federated experiments."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ml.ptbxl.ecg_report_index import build_ecg_report_index
from ml.ptbxl.splits import SPLIT_POLICY_V1, verify_patient_disjoint
from ml.training.train_ecg_report_fusion import partition_records

PTBXL_DATASET_SHA256 = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"


@dataclass(frozen=True)
class ECGReportPartitionPlan:
    seed: int
    num_clients: int
    max_federated_train_records: int | None
    client_partition_strategy: str
    split_policy_version: str
    simulation_disclaimer: str
    index_stats: dict[str, Any]
    patient_disjoint_check: dict[str, Any]
    client_train_counts: tuple[int, ...]
    val_row_count: int
    test_row_count: int
    train_row_count_total: int
    client_train_ecg_id_hashes: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_ecg_id(ecg_id: int) -> str:
    return hashlib.sha256(str(ecg_id).encode("utf-8")).hexdigest()


def _cap_train_records(
    train_records: Sequence[dict[str, Any]],
    max_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(train_records) <= max_records:
        return list(train_records)
    c0 = [r for r in train_records if int(r["label"]) == 0]
    c1 = [r for r in train_records if int(r["label"]) == 1]
    ratio = len(c1) / max(len(train_records), 1)
    n1 = int(round(max_records * ratio))
    n0 = max_records - n1
    rng0 = random.Random(seed + 1)
    rng1 = random.Random(seed + 2)
    rng0.shuffle(c0)
    rng1.shuffle(c1)
    capped = c0[:n0] + c1[:n1]
    if len(capped) < max_records:
        remainder = [r for r in train_records if r not in capped]
        rng = random.Random(seed + 3)
        rng.shuffle(remainder)
        capped.extend(remainder[: max_records - len(capped)])
    return capped[:max_records]


def _patient_label(records: list[dict[str, Any]]) -> int:
    labels = [int(r["label"]) for r in records]
    return max(set(labels), key=labels.count)


def assign_patient_disjoint_clients(
    train_records: Sequence[dict[str, Any]],
    *,
    num_clients: int,
    seed: int,
    strategy: str,
) -> list[list[dict[str, Any]]]:
    """Assign whole patients to clients (no patient spans clients)."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for rec in train_records:
        pid = str(rec["patient_id"])
        by_patient.setdefault(pid, []).append(rec)

    patients = list(by_patient.keys())
    if strategy == "stratified_round_robin":
        p0 = [p for p in patients if _patient_label(by_patient[p]) == 0]
        p1 = [p for p in patients if _patient_label(by_patient[p]) == 1]
        rng0 = random.Random(seed + 10)
        rng1 = random.Random(seed + 11)
        rng0.shuffle(p0)
        rng1.shuffle(p1)
        client_patients: list[list[str]] = [[] for _ in range(num_clients)]
        for idx, pid in enumerate(p0):
            client_patients[idx % num_clients].append(pid)
        for idx, pid in enumerate(p1):
            client_patients[idx % num_clients].append(pid)
    elif strategy == "contiguous_patient_id":
        ordered = sorted(patients)
        rng = random.Random(seed)
        rng.shuffle(ordered)
        base = len(ordered) // num_clients
        client_patients = []
        start = 0
        for client_idx in range(num_clients):
            end = len(ordered) if client_idx == num_clients - 1 else start + base
            client_patients.append(ordered[start:end])
            start = end
    else:
        raise ValueError(f"Unknown client_partition_strategy: {strategy}")

    parts: list[list[dict[str, Any]]] = []
    for plist in client_patients:
        recs: list[dict[str, Any]] = []
        for pid in plist:
            recs.extend(by_patient[pid])
        parts.append(recs)
    return parts


def audit_ecg_report_partitions(
    client_parts: list[list[dict[str, Any]]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    def _stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"n": 0, "positive": 0, "negative": 0, "positive_rate": float("nan")}
        y = [int(r["label"]) for r in records]
        pos = sum(y)
        n = len(y)
        return {"n": n, "positive": pos, "negative": n - pos, "positive_rate": float(pos / n)}

    train_ecg_ids: set[int] = set()
    train_patients: set[str] = set()
    for part in client_parts:
        part_patients: set[str] = set()
        for rec in part:
            eid = int(rec["ecg_id"])
            pid = str(rec["patient_id"])
            if eid in train_ecg_ids:
                raise ValueError(f"Duplicate ecg_id across clients: {eid}")
            train_ecg_ids.add(eid)
            part_patients.add(pid)
        for pid in part_patients:
            if pid in train_patients:
                raise ValueError(f"Patient split across clients: {pid}")
            train_patients.add(pid)

    val_ids = {int(r["ecg_id"]) for r in val_records}
    test_ids = {int(r["ecg_id"]) for r in test_records}
    overlap_tv = train_ecg_ids & val_ids
    overlap_tt = train_ecg_ids & test_ids
    overlap_vt = val_ids & test_ids

    client_stats = []
    for idx, part in enumerate(client_parts):
        st = _stats(part)
        st["client_index"] = idx
        st["unique_patients"] = len({str(r["patient_id"]) for r in part})
        client_stats.append(st)

    return {
        "train_disjoint_across_clients": True,
        "train_val_ecg_id_overlap": len(overlap_tv),
        "train_test_ecg_id_overlap": len(overlap_tt),
        "val_test_ecg_id_overlap": len(overlap_vt),
        "split_leakage_detected": bool(overlap_tv or overlap_tt or overlap_vt),
        "train_label_stats": _stats([r for p in client_parts for r in p]),
        "val_label_stats": _stats(val_records),
        "test_label_stats": _stats(test_records),
        "client_label_stats": client_stats,
    }


def build_ecg_report_federated_partitions(
    ptbxl_root: Path,
    *,
    seed: int = 42,
    num_clients: int = 3,
    max_federated_train_records: int | None = None,
    client_partition_strategy: str = "stratified_round_robin",
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], ECGReportPartitionPlan]:
    records, index_stats = build_ecg_report_index(ptbxl_root)
    disjoint = verify_patient_disjoint(records)
    parts = partition_records(records)

    train_all = parts["train"]
    val_records = parts["validation"]
    test_records = parts["test"]

    if max_federated_train_records is not None:
        train_fed = _cap_train_records(train_all, max_federated_train_records, seed)
    else:
        train_fed = train_all

    client_parts = assign_patient_disjoint_clients(
        train_fed,
        num_clients=num_clients,
        seed=seed,
        strategy=client_partition_strategy,
    )

    audit_ecg_report_partitions(client_parts, val_records, test_records)

    id_hashes = [
        tuple(sorted(_hash_ecg_id(int(r["ecg_id"])) for r in part)) for part in client_parts
    ]
    plan = ECGReportPartitionPlan(
        seed=seed,
        num_clients=num_clients,
        max_federated_train_records=max_federated_train_records,
        client_partition_strategy=client_partition_strategy,
        split_policy_version=SPLIT_POLICY_V1["protocol_version"],
        simulation_disclaimer=(
            "Simulated Flower clients are disjoint patient groups within the PTB-XL protocol "
            "train split. They do NOT represent verified independent hospitals."
        ),
        index_stats=index_stats,
        patient_disjoint_check=disjoint,
        client_train_counts=tuple(len(p) for p in client_parts),
        val_row_count=len(val_records),
        test_row_count=len(test_records),
        train_row_count_total=len(train_fed),
        client_train_ecg_id_hashes=id_hashes,
    )
    return client_parts, val_records, test_records, plan


def save_ecg_report_partition_plan(path: Path, plan: ECGReportPartitionPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)

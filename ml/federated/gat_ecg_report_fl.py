"""
Two-client FedAvg for MedShieldGATECGReportNet over frozen PTB-XL ECG+report embeddings.

Scope: GAT fusion head only (projections + GAT + classifier). Upstream encoders stay frozen
locally when embeddings are precomputed; clients never send raw ECG or report text over Flower.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.ecg_metrics import compute_binary_metrics
from ml.models.gat_ecg_report_fusion import (
    ARCHITECTURE_ID,
    ARCHITECTURE_VERSION,
    GATECGReportConfig,
    GATECGReportInput,
    MedShieldGATECGReportNet,
    load_gat_ecg_report_checkpoint,
    save_gat_ecg_report_checkpoint,
)
from ml.ptbxl.splits import SPLIT_POLICY_V1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = REPO_ROOT / "data" / "processed" / "gat_ecg_report" / "federated_fl_bundle.pt"
DEFAULT_DEPLOY_CHECKPOINT = REPO_ROOT / "models" / "gat_ecg_report" / "best_gat_checkpoint.pt"
GAT_FL_RUNS_ROOT = REPO_ROOT / "data" / "processed" / "federated" / "gat_ecg_report_runs"
FEDERATION_ID = "medshield-gat-ecg-report"
TASK_ID = "gat-ecg-report-federated"
FEDERATED_TRAINING_SCOPE = "gat_fusion_head_over_frozen_embeddings"
NUM_CLIENTS = 2
PARTITION_SEED = 42


class GATFLBundleError(RuntimeError):
    pass


def default_bundle_path() -> Path:
    override = __import__("os").environ.get("MEDSHIELD_GAT_FL_BUNDLE")
    if override:
        return Path(override)
    return DEFAULT_BUNDLE_PATH


def assign_client_for_patient(patient_id: str, *, seed: int = PARTITION_SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % NUM_CLIENTS


def label_class_counts(y: torch.Tensor) -> dict[str, int]:
    arr = y.detach().cpu().numpy().astype(int)
    uniq, cnt = np.unique(arr, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq, cnt)}


def state_dict_to_numpy_list(model: MedShieldGATECGReportNet) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().astype(np.float32) for v in model.state_dict().values()]


def numpy_list_to_state_dict(model: MedShieldGATECGReportNet, arrays: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(arrays):
        raise ValueError(f"Parameter count mismatch: model {len(keys)} vs arrays {len(arrays)}")
    state = {}
    for key, arr in zip(keys, arrays):
        state[key] = torch.tensor(np.asarray(arr, dtype=np.float32))
    model.load_state_dict(state, strict=True)


def load_bundle(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_bundle_path()
    if not p.is_file():
        raise GATFLBundleError(
            f"GAT federated embedding bundle missing: {p}. "
            "Run: python scripts/build_gat_fl_embedding_bundle.py"
        )
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    required = ("meta", "clients", "test")
    for key in required:
        if key not in payload:
            raise GATFLBundleError(f"Invalid bundle (missing {key}): {p}")
    if len(payload["clients"]) != NUM_CLIENTS:
        raise GATFLBundleError(f"Expected {NUM_CLIENTS} client partitions in bundle.")
    return payload


def bundle_readiness() -> tuple[bool, list[str]]:
    blockers: list[str] = []
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        blockers.append("torch_geometric is not installed (required for GAT FL clients).")
    p = default_bundle_path()
    if not p.is_file():
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            rel = p
        blockers.append(f"Missing GAT FL embedding bundle: {rel}")
    return len(blockers) == 0, blockers


def create_model(*, init_checkpoint: Path | str | None = None) -> MedShieldGATECGReportNet:
    ckpt = Path(init_checkpoint) if init_checkpoint else DEFAULT_DEPLOY_CHECKPOINT
    if ckpt.is_file():
        model = load_gat_ecg_report_checkpoint(ckpt, map_location="cpu")
        model.config = GATECGReportConfig(trained=False)
        return model
    return MedShieldGATECGReportNet(GATECGReportConfig(trained=False))


class GATFederatedClientAdapter:
    """Local training on precomputed embeddings for one simulated hospital partition."""

    def __init__(self, partition_id: int, bundle: dict[str, Any], *, init_checkpoint: Path | None = None) -> None:
        if partition_id < 0 or partition_id >= NUM_CLIENTS:
            raise ValueError(f"partition_id must be 0..{NUM_CLIENTS - 1}")
        part = bundle["clients"][partition_id]
        self.partition_id = partition_id
        self.ecg = part["ecg"].float()
        self.text = part["clinical_text"].float()
        self.y = part["labels"].long()
        self.model = create_model(init_checkpoint=init_checkpoint)
        self._criterion = nn.CrossEntropyLoss()
        self._device = torch.device("cpu")

    def num_local_examples(self) -> int:
        return int(self.y.size(0))

    def parameter_arrays(self) -> list[np.ndarray]:
        return state_dict_to_numpy_list(self.model)

    def set_parameters(self, arrays: list[np.ndarray]) -> None:
        numpy_list_to_state_dict(self.model, arrays)

    def fit(self, config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        epochs = max(1, int(config.get("local_epochs", 1)))
        lr = float(config.get("learning_rate", 1e-3))
        loader = DataLoader(
            TensorDataset(self.ecg, self.text, self.y),
            batch_size=int(config.get("batch_size", 64)),
            shuffle=True,
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()
        last_loss = 0.0
        for _ in range(epochs):
            for ecg, text, y in loader:
                opt.zero_grad(set_to_none=True)
                logits = self.model(GATECGReportInput(ecg, text))
                loss = self._criterion(logits, y)
                loss.backward()
                opt.step()
                last_loss = float(loss.item())
        return self.parameter_arrays(), self.num_local_examples(), {"train_loss": last_loss}

    def evaluate(self, config: dict[str, Any]) -> tuple[float, int, dict[str, float]]:
        loader = DataLoader(
            TensorDataset(self.ecg, self.text, self.y),
            batch_size=int(config.get("batch_size", 128)),
            shuffle=False,
        )
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for ecg, text, y in loader:
                logits = self.model(GATECGReportInput(ecg, text))
                loss = self._criterion(logits, y)
                total_loss += float(loss.item()) * y.size(0)
                n += y.size(0)
        avg = total_loss / max(n, 1)
        return avg, n, {"eval_loss": avg}


@torch.no_grad()
def evaluate_model_on_split(
    model: MedShieldGATECGReportNet,
    *,
    ecg: torch.Tensor,
    text: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    ys, scores = [], []
    loader = DataLoader(TensorDataset(ecg, text, labels), batch_size=batch_size, shuffle=False)
    for e, t, y in loader:
        logits = model(GATECGReportInput(e, t))
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.append(y.numpy())
        scores.append(prob)
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    metrics = compute_binary_metrics(y_true, y_score)
    metrics["n_samples"] = int(len(y_true))
    metrics["class_counts"] = label_class_counts(labels)
    return metrics


def run_dir_for(run_id: str) -> Path:
    return GAT_FL_RUNS_ROOT / run_id


def save_global_checkpoint(
    run_id: str,
    arrays: list[np.ndarray],
    *,
    round_number: int,
    parameter_hash: str,
) -> Path:
    out_dir = run_dir_for(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(init_checkpoint=None)
    numpy_list_to_state_dict(model, arrays)
    model.config = GATECGReportConfig(trained=True)
    path = out_dir / "global_federated_checkpoint.pt"
    save_gat_ecg_report_checkpoint(
        path,
        model,
        extra={
            "run_id": run_id,
            "round_number": round_number,
            "global_parameter_hash": parameter_hash,
            "federated_training_scope": FEDERATED_TRAINING_SCOPE,
            "architecture_id": ARCHITECTURE_ID,
        },
    )
    return path


def finalize_federated_run(run_id: str, *, live_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Held-out test evaluation for a completed Flower run. Persists report.json under gat_ecg_report_runs/.
    """
    bundle = load_bundle()
    meta = bundle["meta"]
    out_dir = run_dir_for(run_id)
    ckpt_path = out_dir / "global_federated_checkpoint.pt"
    if not ckpt_path.is_file() and live_state:
        history = live_state.get("round_history") or []
        for item in reversed(history):
            alt_id = str(item.get("run_id") or "")
            if not alt_id:
                continue
            alt_path = run_dir_for(alt_id) / "global_federated_checkpoint.pt"
            if alt_path.is_file():
                ckpt_path = alt_path
                break
    if not ckpt_path.is_file():
        raise GATFLBundleError(f"Missing federated checkpoint for run {run_id}: {ckpt_path}")

    model = load_gat_ecg_report_checkpoint(ckpt_path, map_location="cpu")
    test = bundle["test"]
    test_metrics = evaluate_model_on_split(
        model,
        ecg=test["ecg"],
        text=test["clinical_text"],
        labels=test["labels"],
    )

    baseline_metrics: dict[str, Any] | None = None
    baseline_note: str | None = None
    deploy_ckpt = Path(meta.get("init_checkpoint_path") or DEFAULT_DEPLOY_CHECKPOINT)
    if deploy_ckpt.is_file():
        baseline_model = load_gat_ecg_report_checkpoint(deploy_ckpt, map_location="cpu")
        baseline_metrics = evaluate_model_on_split(
            baseline_model,
            ecg=test["ecg"],
            text=test["clinical_text"],
            labels=test["labels"],
        )
        baseline_note = (
            "Centralized GAT checkpoint evaluated on the same held-out test embeddings "
            "(not re-trained during this federated run)."
        )
    else:
        baseline_note = "Centralized baseline checkpoint unavailable."

    init_h = (live_state or {}).get("initial_parameter_hash")
    final_h = (live_state or {}).get("global_parameter_hash")
    round_history = (live_state or {}).get("round_history") or []

    report = {
        "experiment": "gat_ecg_report_federated",
        "federation_id": FEDERATION_ID,
        "task_id": TASK_ID,
        "architecture_id": ARCHITECTURE_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "federated_training_scope": FEDERATED_TRAINING_SCOPE,
        "simulation_disclaimer": (
            "Two simulated hospital clients with disjoint PTB-XL train partitions. "
            "Not independently deployed institutions."
        ),
        "privacy_note": (
            "FedAvg exchanges GAT fusion-head weights and allowlisted scalar metrics only. "
            "Precomputed embeddings are kept local; raw ECG and clinical reports are not sent via Flower."
        ),
        "target_definition": meta.get(
            "target_definition",
            "PTB-XL proxy: NORM (0) vs MI/STTC (1). Research demonstration — not clinical performance.",
        ),
        "config": {
            "federation_id": FEDERATION_ID,
            "task_id": TASK_ID,
            "architecture_id": ARCHITECTURE_ID,
            "seed": meta.get("partition_seed", PARTITION_SEED),
            "num_rounds": (live_state or {}).get("total_rounds"),
            "participation": {"num_clients": NUM_CLIENTS},
            "local_training": {"local_epochs": (live_state or {}).get("local_epochs", 1)},
            "aggregation_strategy": "FedAvg",
            "split_policy": meta.get("split_policy", SPLIT_POLICY_V1),
            "bundle_path_relative": meta.get("bundle_relative_path"),
        },
        "partition_plan": {
            "seed": meta.get("partition_seed", PARTITION_SEED),
            "num_clients": NUM_CLIENTS,
            "client_train_counts": meta.get("client_train_counts"),
            "test_row_count": int(test["labels"].size(0)),
            "train_row_count_total": meta.get("train_row_count_total"),
            "simulation_disclaimer": (
                "Client partitions are disjoint by patient_id assignment on the PTB-XL train split only."
            ),
        },
        "round_history": round_history,
        "parameter_l2_initial": init_h,
        "parameter_l2_final": final_h,
        "evaluation": {
            "test": test_metrics,
            "held_out_split": "test",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "centralized_baseline_same_test_embeddings": {
            "note": baseline_note,
            "checkpoint_relative": str(deploy_ckpt.relative_to(REPO_ROOT)) if deploy_ckpt.is_file() else None,
            "test": baseline_metrics,
        },
        "checkpoint_summary": {
            "global_parameter_hash": final_h,
            "initial_parameter_hash": init_h,
            "checkpoint_hash_sha256_prefix": final_h,
        },
        "flower_run_id": run_id,
        "completed": True,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_bundle_from_tensors(
    *,
    train_ecg: torch.Tensor,
    train_text: torch.Tensor,
    train_y: torch.Tensor,
    train_patient_ids: list[str],
    test_ecg: torch.Tensor,
    test_text: torch.Tensor,
    test_y: torch.Tensor,
    test_patient_ids: list[str],
    meta: dict[str, Any],
    out_path: Path,
) -> None:
    if len(train_patient_ids) != train_y.size(0):
        raise ValueError("train patient_ids length mismatch")
    client_buckets: list[list[int]] = [[] for _ in range(NUM_CLIENTS)]
    seen_patients: dict[str, int] = {}
    for idx, pid in enumerate(train_patient_ids):
        pid_s = str(pid)
        if pid_s in seen_patients and seen_patients[pid_s] != assign_client_for_patient(pid_s):
            raise ValueError(f"Patient {pid_s} would span multiple clients")
        client = assign_client_for_patient(pid_s)
        seen_patients[pid_s] = client
        client_buckets[client].append(idx)

    clients = []
    client_counts = []
    for cid in range(NUM_CLIENTS):
        idxs = client_buckets[cid]
        if not idxs:
            raise ValueError(f"Client {cid} has zero training examples")
        idx_t = torch.tensor(idxs, dtype=torch.long)
        clients.append(
            {
                "ecg": train_ecg.index_select(0, idx_t),
                "clinical_text": train_text.index_select(0, idx_t),
                "labels": train_y.index_select(0, idx_t),
                "patient_ids": [train_patient_ids[i] for i in idxs],
            }
        )
        client_counts.append(len(idxs))

    overlap = set(train_patient_ids) & set(test_patient_ids)
    if overlap:
        raise ValueError(f"Train/test patient leakage: {len(overlap)} overlapping patient_ids")

    meta = {
        **meta,
        "client_train_counts": client_counts,
        "train_row_count_total": int(train_y.size(0)),
        "partition_seed": PARTITION_SEED,
        "num_clients": NUM_CLIENTS,
        "federated_training_scope": FEDERATED_TRAINING_SCOPE,
    }
    payload = {
        "meta": meta,
        "clients": clients,
        "test": {
            "ecg": test_ecg,
            "clinical_text": test_text,
            "labels": test_y,
            "patient_ids": test_patient_ids,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

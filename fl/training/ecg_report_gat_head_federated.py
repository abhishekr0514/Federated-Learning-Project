"""Federated GAT-head training on precomputed ECG+report embeddings (not encoder FL)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl.adapters.ecg_report_federation import load_encoder_bundle
from fl.aggregation.fedavg import weighted_fedavg
from fl.config import FederationConfig
from fl.ecg_report.device import resolve_fl_device
from fl.ecg_report.partitions import (
    audit_ecg_report_partitions,
    build_ecg_report_federated_partitions,
    save_ecg_report_partition_plan,
)
from fl.parameters.schema import try_schema_from_torch_state_dict
from fl.parameters.serialize import ndarrays_to_state_dict, state_dict_to_ndarrays
from fl.training.ecg_report_federated import (
    _cap_train_records,
    _enrich_metrics,
    _verify_ptbxl_fingerprint,
)
from ml.evaluation.tabular_metrics import find_best_youden_threshold
from ml.models.gat_ecg_report_fusion import GATECGReportConfig, MedShieldGATECGReportNet
from ml.training.train_ecg_report_fusion import fit_train_normalization
from ml.training.train_gat_ecg_report import evaluate_gat, extract_embeddings, train_gat


def run_ecg_report_gat_head_federated_experiment(
    cfg: FederationConfig,
    *,
    output_root: Path | None = None,
    experiment_label: str | None = None,
) -> dict[str, Any]:
    if cfg.federated_trainable_scope != "gat_head":
        raise ValueError("federated_trainable_scope must be gat_head")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    t0 = time.time()
    ptbxl_root = REPO_ROOT / cfg.ptbxl_root
    _verify_ptbxl_fingerprint(ptbxl_root, cfg.expected_ptbxl_database_sha256)
    device = resolve_fl_device()

    client_parts, val_records, test_records, plan = build_ecg_report_federated_partitions(
        ptbxl_root,
        seed=cfg.seed,
        num_clients=cfg.participation.num_clients,
        max_federated_train_records=cfg.max_federated_train_records,
        client_partition_strategy=cfg.client_partition_strategy,
    )
    if cfg.eval_max_val_records is not None:
        val_records = _cap_train_records(val_records, cfg.eval_max_val_records, cfg.seed + 100)
    if cfg.eval_max_test_records is not None:
        test_records = _cap_train_records(test_records, cfg.eval_max_test_records, cfg.seed + 101)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root or (
        REPO_ROOT / cfg.output_dir / cfg.ecg_report_experiment_subdir / (experiment_label or f"gat_head_{run_id}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_ecg_report_partition_plan(run_dir / "partition_plan.json", plan)
    audit_ecg_report_partitions(client_parts, val_records, test_records)

    manifest_path = REPO_ROOT / cfg.encoder_manifest_path
    encoder_model, pipe, _deploy_norm, _enc_manifest, enc_fp = load_encoder_bundle(
        REPO_ROOT, manifest_path, device
    )
    all_train = [rec for part in client_parts for rec in part]
    norm_stats = fit_train_normalization(ptbxl_root, all_train, max_records=None)

    def _embed(records: list[dict[str, Any]]):
        return extract_embeddings(
            encoder_model,
            pipe,
            norm_stats,
            records,
            ptbxl_root,
            device,
            cfg.local_training.batch_size,
        )

    client_tensors = [_embed(part) for part in client_parts]
    val_ecg, val_text, val_y = _embed(val_records)
    test_ecg, test_text, test_y = _embed(test_records)

    template = MedShieldGATECGReportNet(GATECGReportConfig(trained=False)).to(device)
    schema = try_schema_from_torch_state_dict(
        template.state_dict(),
        architecture_id="medshield.diagnostic_net.gat_ecg_report",
        architecture_version="1.0.0",
    )
    global_arrays, names = state_dict_to_ndarrays(template.state_dict())

    def _local_train(state_arrays, ecg, text, y, epochs: int):
        model = MedShieldGATECGReportNet(GATECGReportConfig(trained=False)).to(device)
        model.load_state_dict(ndarrays_to_state_dict(state_arrays, names, template.state_dict()))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.local_training.learning_rate)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(ecg, text, y.long()), batch_size=cfg.local_training.batch_size, shuffle=True
        )
        for _ in range(epochs):
            train_gat(model, loader, opt, crit, device)
        return state_dict_to_ndarrays(model.state_dict())[0]

    for _round in range(1, cfg.num_rounds + 1):
        updates = []
        for ecg, text, y in client_tensors:
            out = _local_train(global_arrays, ecg, text, y, cfg.local_training.local_epochs)
            updates.append((out, int(y.size(0))))
        global_arrays = weighted_fedavg(updates, schema=schema)

    gat = MedShieldGATECGReportNet(GATECGReportConfig(trained=False)).to(device)
    gat.load_state_dict(ndarrays_to_state_dict(global_arrays, names, template.state_dict()))
    val_loader = DataLoader(
        TensorDataset(val_ecg, val_text, val_y.long()), batch_size=cfg.local_training.batch_size
    )
    test_loader = DataLoader(
        TensorDataset(test_ecg, test_text, test_y.long()), batch_size=cfg.local_training.batch_size
    )
    y_v, s_v = evaluate_gat(gat, val_loader, device)
    thr, youden_j, _ = find_best_youden_threshold(y_v, s_v)
    y_t, s_t = evaluate_gat(gat, test_loader, device)
    test_metrics = _enrich_metrics(y_t, s_t, thr)

    report = {
        "experiment": "federated_gat_head_pretrained_embeddings",
        "training_mode": {
            "description": "GAT head only; embeddings from frozen ECG-report encoder.",
            "not_federated_encoder_training": True,
        },
        "encoder_checkpoint_sha256": enc_fp,
        "evaluation": {"test": test_metrics, "youden_threshold": thr, "youden_j": youden_j},
        "test_n": int(len(test_y)),
        "val_n": int(len(val_y)),
        "eval_caps_applied": {
            "eval_max_val_records": cfg.eval_max_val_records,
            "eval_max_test_records": cfg.eval_max_test_records,
        },
        "elapsed_seconds": time.time() - t0,
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {
        "run_dir": str(run_dir),
        "test_auroc": test_metrics.get("auroc"),
        "test_accuracy": test_metrics.get("accuracy"),
    }

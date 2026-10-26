"""Federated ECG + clinical report experiment (PTB-XL proxy, frozen encoders + head FedAvg)."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl.adapters.ecg_report_federation import (
    ECGReportFederationAdapter,
    build_client_model_template,
    load_encoder_bundle,
)
from fl.aggregation.fedavg import weighted_fedavg
from fl.checkpoints import FederatedCheckpointMetadata, save_federated_checkpoint
from fl.client.medshield_numpy_client import MedShieldNumPyClient
from fl.config import FederationConfig
from fl.ecg_report.partitions import (
    PTBXL_DATASET_SHA256,
    _cap_train_records,
    audit_ecg_report_partitions,
    build_ecg_report_federated_partitions,
    save_ecg_report_partition_plan,
)
from fl.ecg_report.device import assert_model_on_device, resolve_fl_device
from fl.ecg_report.trainable import merge_trainable_into_model, trainable_state_dict
from fl.parameters.serialize import ndarrays_to_state_dict, state_dict_to_ndarrays
from fl.server.strategy_factory import build_fedavg_strategy, require_flwr_strategy
from ml.datasets.ptbxl_ecg_report import PTBXLECGReportDataset
from ml.evaluation.calibration import CalibrationInputError, compute_calibration_metrics
from ml.evaluation.ecg_metrics import compute_binary_metrics
from ml.evaluation.tabular_metrics import find_best_youden_threshold
from ml.models.ecg_report_fusion import ECGReportFusionClassifier
from ml.training.train_ecg_report_fusion import make_collate_fn, train_one_epoch, fit_train_normalization

TEXT_MAX_LEN = 256


def _verify_ptbxl_fingerprint(ptbxl_root: Path, expected: str) -> None:
    db = ptbxl_root / "ptbxl_database.csv"
    if not db.is_file():
        raise FileNotFoundError(f"Missing {db}")
    h = hashlib.sha256()
    with db.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError(f"PTB-XL database CSV SHA256 mismatch: {h.hexdigest()} != {expected}")


def _enrich_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    base = compute_binary_metrics(y_true, y_score, threshold=threshold)
    cm = base["confusion_matrix"]
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    base["n_samples"] = int(n)
    base["accuracy"] = float((tp + tn) / n) if n else float("nan")
    sens = base.get("sensitivity", float("nan"))
    spec = base.get("specificity", float("nan"))
    if not math.isnan(sens) and not math.isnan(spec):
        base["balanced_accuracy"] = float((sens + spec) / 2.0)
    else:
        base["balanced_accuracy"] = float("nan")
    base["applied_threshold"] = float(threshold)
    try:
        cal = compute_calibration_metrics(y_true, y_score)
        base["brier_score"] = cal.get("brier_score")
        base["ece"] = cal.get("ece")
    except CalibrationInputError as exc:
        base["calibration_note"] = str(exc)
    return base


def _evaluate_model_on_split(
    model: ECGReportFusionClassifier,
    records: list[dict[str, Any]],
    *,
    ptbxl_root: Path,
    normalization,
    text_pipeline,
    device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    if not records:
        return np.array([]), np.array([])
    ds = PTBXLECGReportDataset(records, ptbxl_root, normalization)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(text_pipeline.tokenizer, TEXT_MAX_LEN),
    )
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].cpu().numpy()
            logit, _, _ = model(x, ids, mask)
            prob = torch.sigmoid(logit).cpu().numpy()
            ys.append(y)
            scores.append(prob)
    return np.concatenate(ys), np.concatenate(scores)


def _evaluate_global(
    model: ECGReportFusionClassifier,
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    ptbxl_root: Path,
    normalization,
    text_pipeline,
    device,
    batch_size: int,
) -> dict[str, Any]:
    val_y, val_probs = _evaluate_model_on_split(
        model, val_records, ptbxl_root=ptbxl_root, normalization=normalization,
        text_pipeline=text_pipeline, device=device, batch_size=batch_size,
    )
    test_y, test_probs = _evaluate_model_on_split(
        model, test_records, ptbxl_root=ptbxl_root, normalization=normalization,
        text_pipeline=text_pipeline, device=device, batch_size=batch_size,
    )
    best_threshold, youden_j, grid = find_best_youden_threshold(val_y, val_probs)
    return {
        "validation": _enrich_metrics(val_y, val_probs, best_threshold),
        "test": _enrich_metrics(test_y, test_probs, best_threshold),
        "youden_threshold": best_threshold,
        "youden_j": youden_j,
        "threshold_grid_size": len(grid),
    }


def _centralized_fusion_head_baseline(
    template_model: ECGReportFusionClassifier,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    trainable_scope: str,
    ptbxl_root: Path,
    normalization,
    text_pipeline,
    device,
    total_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    import copy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from fl.ecg_report.trainable import apply_trainable_scope

    torch.manual_seed(seed)
    model = copy.deepcopy(template_model)
    apply_trainable_scope(model, trainable_scope)
    model.to(device)
    ds = PTBXLECGReportDataset(train_records, ptbxl_root, normalization)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(text_pipeline.tokenizer, TEXT_MAX_LEN),
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(total_epochs):
        train_one_epoch(model, loader, optimizer, criterion, device)
    metrics = _evaluate_global(
        model, val_records, test_records,
        ptbxl_root=ptbxl_root, normalization=normalization, text_pipeline=text_pipeline,
        device=device, batch_size=batch_size,
    )
    return {
        "experiment_type": "centralized_fusion_head_same_train_rows",
        "total_epochs": total_epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        **metrics,
    }


def run_ecg_report_federated_experiment(
    cfg: FederationConfig,
    *,
    output_root: Path | None = None,
    experiment_label: str | None = None,
) -> dict[str, Any]:
    cfg.validate()
    if not cfg.real_ecg_report_experiment:
        raise ValueError("Config must set real_ecg_report_experiment=true")
    if cfg.federated_trainable_scope == "gat_head":
        raise ValueError("Use run-ecg-report-gat-head-experiment for gat_head scope")

    require_flwr_strategy()
    import flwr
    import torch

    t0 = time.time()
    ptbxl_root = REPO_ROOT / cfg.ptbxl_root
    _verify_ptbxl_fingerprint(ptbxl_root, cfg.expected_ptbxl_database_sha256)

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
    if output_root is not None:
        run_dir = Path(output_root)
    else:
        sub = experiment_label or run_id
        run_dir = REPO_ROOT / cfg.output_dir / cfg.ecg_report_experiment_subdir / sub
    run_dir.mkdir(parents=True, exist_ok=True)
    save_ecg_report_partition_plan(run_dir / "partition_plan.json", plan)

    all_train = [rec for part in client_parts for rec in part]
    data_audit = audit_ecg_report_partitions(client_parts, val_records, test_records)
    with (run_dir / "data_audit.json").open("w", encoding="utf-8") as f:
        json.dump(data_audit, f, indent=2)

    device = resolve_fl_device()
    manifest_path = REPO_ROOT / cfg.encoder_manifest_path
    encoder_model, text_pipeline, _deploy_norm, enc_manifest, enc_fp = load_encoder_bundle(
        REPO_ROOT, manifest_path, device
    )
    assert_model_on_device(encoder_model, device)

    # Normalization fit on federated train records only (protocol train subset used for FL).
    norm_stats = fit_train_normalization(ptbxl_root, all_train, max_records=None)

    norm_path = run_dir / "normalization_federated_train.json"
    norm_stats.save(norm_path)

    scope = cfg.federated_trainable_scope
    template, trainable_names, frozen_names = build_client_model_template(encoder_model, scope)

    # Eval pretrained checkpoint head+encoders on protocol val/test (reference, not FL result).
    pretrained_eval = _evaluate_global(
        encoder_model, val_records, test_records,
        ptbxl_root=ptbxl_root, normalization=_deploy_norm, text_pipeline=text_pipeline,
        device=device, batch_size=cfg.local_training.batch_size,
    )

    adapters: list[ECGReportFederationAdapter] = []
    for idx, part in enumerate(client_parts):
        if not part:
            raise ValueError(f"Client {idx} has empty partition")
        client_model, tnames, _ = build_client_model_template(encoder_model, scope)
        adapters.append(
            ECGReportFederationAdapter(
                train_records=part,
                model=client_model,
                text_pipeline=text_pipeline,
                normalization=norm_stats,
                ptbxl_root=ptbxl_root,
                trainable_names=tnames,
                federated_trainable_scope=scope,
                client_index=idx,
                num_clients=cfg.participation.num_clients,
                local_epochs=cfg.local_training.local_epochs,
                batch_size=cfg.local_training.batch_size,
                learning_rate=cfg.local_training.learning_rate,
                weight_decay=cfg.local_training.weight_decay,
                device_name=str(device),
                encoder_fingerprint=enc_fp,
            )
        )

    schema = adapters[0].parameter_schema()
    global_arrays, names = adapters[0].get_parameters()
    init_l2 = float(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in global_arrays))

    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    build_fedavg_strategy(
        federation_config=cfg,
        parameter_schema=schema,
        initial_parameters=ndarrays_to_parameters(global_arrays),
    )

    round_history: list[dict[str, Any]] = []
    numpy_clients = [MedShieldNumPyClient(a) for a in adapters]
    round_losses: list[float] = []

    for round_idx in range(1, cfg.num_rounds + 1):
        updates: list[tuple[list[np.ndarray], int]] = []
        round_losses = []
        global_parameters = ndarrays_to_parameters(global_arrays)
        for client in numpy_clients:
            out_arrays, n_examples, metrics = client.fit(
                global_parameters,
                {"local_epochs": cfg.local_training.local_epochs},
            )
            round_losses.append(float(metrics.get("loss", 0.0)))
            updates.append((out_arrays, n_examples))
        global_arrays = weighted_fedavg(updates, schema=schema)
        round_history.append(
            {
                "round": round_idx,
                "mean_client_loss": float(np.mean(round_losses)),
            }
        )

    final_l2 = float(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in global_arrays))
    trained = cfg.local_training.local_epochs > 0 and cfg.num_rounds > 0 and final_l2 != init_l2

    global_model, _, _ = build_client_model_template(encoder_model, scope)
    merge_trainable_into_model(global_model, global_arrays, names, trainable_names)
    global_model.to(device)

    eval_metrics = _evaluate_global(
        global_model, val_records, test_records,
        ptbxl_root=ptbxl_root, normalization=norm_stats, text_pipeline=text_pipeline,
        device=device, batch_size=cfg.local_training.batch_size,
    )

    total_equiv_epochs = cfg.local_training.local_epochs * cfg.num_rounds
    centralized_head = _centralized_fusion_head_baseline(
        encoder_model,
        all_train,
        val_records,
        test_records,
        trainable_scope=scope,
        ptbxl_root=ptbxl_root,
        normalization=norm_stats,
        text_pipeline=text_pipeline,
        device=device,
        total_epochs=total_equiv_epochs,
        batch_size=cfg.local_training.batch_size,
        lr=cfg.local_training.learning_rate,
        weight_decay=cfg.local_training.weight_decay,
        seed=cfg.seed + 99,
    )

    import copy

    pretrained_unified = _evaluate_global(
        copy.deepcopy(encoder_model),
        val_records,
        test_records,
        ptbxl_root=ptbxl_root,
        normalization=norm_stats,
        text_pipeline=text_pipeline,
        device=device,
        batch_size=cfg.local_training.batch_size,
    )
    unified_comparison = {
        "protocol": "identical_val_test_records",
        "normalization": "federated_train_fit_only",
        "val_n": len(val_records),
        "test_n": len(test_records),
        "threshold_policy": "youden_j_on_validation_per_model",
        "pretrained_checkpoint_unchanged": {
            "test": pretrained_unified["test"],
            "validation": pretrained_unified["validation"],
            "youden_threshold": pretrained_unified["youden_threshold"],
        },
        "centralized_trainable_scope": {
            "test": centralized_head["test"],
            "validation": centralized_head["validation"],
            "youden_threshold": centralized_head["youden_threshold"],
        },
        "federated_global_model": {
            "test": eval_metrics["test"],
            "validation": eval_metrics["validation"],
            "youden_threshold": eval_metrics["youden_threshold"],
        },
    }

    meta = FederatedCheckpointMetadata(
        federation_id=cfg.federation_id,
        task_id=cfg.task_id,
        architecture_id=cfg.architecture_id,
        architecture_version=cfg.architecture_version,
        aggregation_strategy=cfg.aggregation_strategy,
        round_number=cfg.num_rounds,
        parameter_schema_signature=schema.signature(),
        seed=cfg.seed,
        trained=trained,
        flower_version=flwr.__version__,
        pytorch_version=torch.__version__,
        preprocessing_contract_id=cfg.preprocessing_contract_id,
        preprocessing_contract_version=cfg.preprocessing_contract_version,
        aggregate_metrics={"mean_final_client_loss": float(np.mean(round_losses))},
    )
    ckpt_path = run_dir / "global_federated_head_checkpoint.pt"
    save_federated_checkpoint(ckpt_path, metadata=meta, global_arrays=global_arrays, parameter_names=names)

    elapsed = time.time() - t0
    manifest = {
        "experiment": "federated_ecg_report_frozen_encoders_fusion_head",
        "experiment_label": experiment_label,
        "run_id": run_id,
        "training_mode": {
            "description": "Federated fusion-head training only; ECG and text encoders frozen.",
            "federated_trainable_scope": scope,
            "trainable_parameter_names": trainable_names,
            "frozen_parameter_count": len(frozen_names),
            "aggregated_parameters": "sample_weighted_fedavg on trainable head only",
        },
        "config": cfg.to_dict(),
        "partition_plan_summary": {
            "train_row_count_total": plan.train_row_count_total,
            "val_row_count": plan.val_row_count,
            "test_row_count": plan.test_row_count,
            "client_train_counts": list(plan.client_train_counts),
        },
        "data_audit": data_audit,
        "encoder_manifest_path": str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "encoder_checkpoint_sha256": enc_fp,
        "deploy_normalization_used_for_pretrained_eval": str(enc_manifest.get("normalization_path")),
        "federated_train_normalization": str(norm_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "round_history": round_history,
        "parameter_l2_initial": init_l2,
        "parameter_l2_final": final_l2,
        "evaluation": eval_metrics,
        "pretrained_encoder_checkpoint_eval": pretrained_eval,
        "centralized_fusion_head_same_train_rows": centralized_head,
        "unified_comparison_same_records": unified_comparison,
        "eval_caps_applied": {
            "eval_max_val_records": cfg.eval_max_val_records,
            "eval_max_test_records": cfg.eval_max_test_records,
            "max_federated_train_records": cfg.max_federated_train_records,
        },
        "comparison_notes": {
            "not_end_to_end_federated": True,
            "same_ptbxl_split_policy": plan.split_policy_version,
            "threshold_selected_on_validation_only": True,
            "pretrained_eval_uses_deploy_normalization": True,
            "federated_eval_uses_train_fitted_normalization": True,
        },
        "elapsed_seconds": elapsed,
        "device": str(device),
        "privacy_note": "FedAvg parameter aggregation only; no raw waveforms or report text sent to server.",
        "simulation_disclaimer": plan.simulation_disclaimer,
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "run_dir": str(run_dir),
        "trained": trained,
        "test_auroc": eval_metrics["test"].get("auroc"),
        "test_accuracy": eval_metrics["test"].get("accuracy"),
        "centralized_test_auroc": centralized_head["test"].get("auroc"),
    }

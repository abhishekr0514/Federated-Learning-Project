"""
Small real tabular federated experiment (cardio_train.csv).

Uses manual FedAvg rounds with MedShield-validated adapters (Flower-compatible parameters).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl.adapters.tabular import TabularFederationAdapter
from fl.aggregation.fedavg import weighted_fedavg
from fl.checkpoints import FederatedCheckpointMetadata, save_federated_checkpoint
from fl.client.medshield_numpy_client import MedShieldNumPyClient
from fl.config import FederationConfig
from fl.parameters.serialize import ndarrays_to_state_dict, state_dict_to_ndarrays
from fl.server.strategy_factory import build_fedavg_strategy, require_flwr_strategy
from fl.tabular.partitions import (
    EXPECTED_DATASET_SHA256,
    build_tabular_federated_partitions,
    save_partition_plan,
)
from ml.datasets.cardio_tabular import CardioTabularDataset
from ml.evaluation.tabular_metrics import compute_tabular_metrics, find_best_youden_threshold
from ml.models.tabular_encoder import TabularBinaryClassifier
from ml.tabular.preprocessing import TabularPreprocessor
from ml.training.train_tabular_baseline import evaluate_model, set_seed, train_one_epoch


def _verify_dataset_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError(f"Dataset SHA256 mismatch for {path}")


def _parameter_l2(arrays: list[np.ndarray]) -> float:
    return float(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in arrays))


def _evaluate_global_model(
    model: TabularBinaryClassifier,
    preprocessor: TabularPreprocessor,
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    device,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    val_x = preprocessor.transform_batch([r["features"] for r in val_records])
    test_x = preprocessor.transform_batch([r["features"] for r in test_records])
    val_y = np.array([r["target"] for r in val_records], dtype=np.float32)
    test_y = np.array([r["target"] for r in test_records], dtype=np.float32)

    val_loader = DataLoader(CardioTabularDataset(val_x, val_y), batch_size=256, shuffle=False)
    test_loader = DataLoader(CardioTabularDataset(test_x, test_y), batch_size=256, shuffle=False)

    model.eval()
    val_y_true, val_probs = evaluate_model(model, val_loader, device)
    best_threshold, youden_j, _grid = find_best_youden_threshold(val_y_true, val_probs)

    test_y_true, test_probs = evaluate_model(model, test_loader, device)
    test_metrics = compute_tabular_metrics(test_y_true, test_probs, threshold=best_threshold)
    val_metrics = compute_tabular_metrics(val_y_true, val_probs, threshold=best_threshold)

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "youden_threshold": best_threshold,
        "youden_j": youden_j,
    }


def _centralized_baseline_same_train(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    preprocessor: TabularPreprocessor,
    *,
    total_epochs: int,
    seed: int,
    device,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    set_seed(seed)
    train_x = preprocessor.transform_batch([r["features"] for r in train_records])
    train_y = np.array([r["target"] for r in train_records], dtype=np.float32)
    train_loader = DataLoader(CardioTabularDataset(train_x, train_y), batch_size=256, shuffle=True)

    model = TabularBinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(total_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    metrics = _evaluate_global_model(model, preprocessor, val_records, test_records, device)
    return {"total_epochs": total_epochs, **metrics}


def run_tabular_federated_experiment(
    cfg: FederationConfig,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    cfg.validate()
    if not cfg.real_tabular_experiment:
        raise ValueError("Config must set real_tabular_experiment=true")

    require_flwr_strategy()
    import flwr
    import torch

    set_seed(cfg.seed)
    data_path = REPO_ROOT / cfg.dataset_path
    _verify_dataset_sha256(data_path, cfg.expected_dataset_sha256)

    client_parts, val_records, test_records, plan = build_tabular_federated_partitions(
        data_path,
        seed=cfg.seed,
        num_clients=cfg.participation.num_clients,
        max_federated_train_rows=cfg.max_federated_train_rows,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (output_root or (REPO_ROOT / cfg.output_dir / "tabular_cardio_runs" / run_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_partition_plan(run_dir / "partition_plan.json", plan)

    # Preprocessor fit on combined federated train rows (protocol: train-only fit).
    all_train = [rec for part in client_parts for rec in part]
    preprocessor = TabularPreprocessor()
    preprocessor.fit([r["features"] for r in all_train])
    preprocessor.save(run_dir / "preprocessor.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapters: list[TabularFederationAdapter] = []
    for idx, part in enumerate(client_parts):
        adapters.append(
            TabularFederationAdapter(
                train_records=part,
                preprocessor=preprocessor,
                data_path=data_path,
                client_index=idx,
                num_clients=cfg.participation.num_clients,
                local_epochs=cfg.local_training.local_epochs,
                batch_size=cfg.local_training.batch_size,
                learning_rate=cfg.local_training.learning_rate,
                weight_decay=cfg.local_training.weight_decay,
                device_name=str(device),
            )
        )

    schema = adapters[0].parameter_schema()
    global_arrays, names = adapters[0].get_parameters()
    init_l2 = _parameter_l2(global_arrays)

    # Flower strategy smoke (same schema/initial params)
    from flwr.common import ndarrays_to_parameters

    build_fedavg_strategy(
        federation_config=cfg,
        parameter_schema=schema,
        initial_parameters=ndarrays_to_parameters(global_arrays),
    )

    round_history: list[dict[str, Any]] = []
    num_rounds = cfg.num_rounds
    numpy_clients = [MedShieldNumPyClient(a) for a in adapters]

    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    round_losses: list[float] = []
    for round_idx in range(1, num_rounds + 1):
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

        prev_global = [a.copy() for a in global_arrays]
        global_arrays = weighted_fedavg(updates, schema=schema)
        delta = math.sqrt(max(_parameter_l2([g - p for g, p in zip(global_arrays, prev_global)]), 0.0))
        round_history.append(
            {
                "round": round_idx,
                "mean_client_loss": float(np.mean(round_losses)),
                "global_parameter_delta_l2": delta,
            }
        )

    final_l2 = _parameter_l2(global_arrays)
    trained = cfg.local_training.local_epochs > 0 and num_rounds > 0 and final_l2 != init_l2

    # Load global weights into model for evaluation
    model = TabularBinaryClassifier().to(device)
    state = ndarrays_to_state_dict(global_arrays, names, template_state_dict=model.state_dict())
    model.load_state_dict(state, strict=True)

    eval_metrics = _evaluate_global_model(model, preprocessor, val_records, test_records, device)

    total_equiv_epochs = cfg.local_training.local_epochs * num_rounds
    baseline = _centralized_baseline_same_train(
        all_train,
        val_records,
        test_records,
        preprocessor,
        total_epochs=total_equiv_epochs,
        seed=cfg.seed + 99,
        device=device,
    )

    meta = FederatedCheckpointMetadata(
        federation_id=cfg.federation_id,
        task_id=cfg.task_id,
        architecture_id=cfg.architecture_id,
        architecture_version=cfg.architecture_version,
        aggregation_strategy=cfg.aggregation_strategy,
        round_number=num_rounds,
        parameter_schema_signature=schema.signature(),
        seed=cfg.seed,
        trained=trained,
        flower_version=flwr.__version__,
        pytorch_version=torch.__version__,
        feature_schema_id=cfg.feature_schema_id,
        feature_schema_version=cfg.feature_schema_version,
        preprocessing_contract_id=cfg.preprocessing_contract_id,
        preprocessing_contract_version=cfg.preprocessing_contract_version,
        aggregate_metrics={"mean_final_client_loss": float(np.mean(round_losses))},
    )

    ckpt_path = run_dir / "global_federated_checkpoint.pt"
    save_federated_checkpoint(ckpt_path, metadata=meta, global_arrays=global_arrays, parameter_names=names)

    manifest = {
        "experiment": "real_tabular_federated_cardio",
        "config": cfg.to_dict(),
        "partition_plan": plan.to_dict(),
        "round_history": round_history,
        "parameter_l2_initial": init_l2,
        "parameter_l2_final": final_l2,
        "evaluation": eval_metrics,
        "centralized_baseline_same_train_rows": baseline,
        "privacy_note": "FedAvg parameter aggregation only; not differential privacy.",
        "simulation_disclaimer": plan.simulation_disclaimer,
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "run_dir": str(run_dir),
        "trained": trained,
        "rounds": num_rounds,
        "clients": cfg.participation.num_clients,
        "checkpoint": str(ckpt_path),
        "round_history": round_history,
        "test_auroc": eval_metrics["test"].get("auroc"),
        "baseline_test_auroc": baseline["test"].get("auroc"),
    }

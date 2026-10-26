"""
MedShield Flower App (Flower 1.30 ServerApp / ClientApp).

Two simulated hospitals train MedShieldGATECGReportNet (fusion head over frozen PTB-XL embeddings).
Non-clinical research demo — parameters and allowlisted scalars only over the wire.
"""

from __future__ import annotations

import os
from typing import Any

from run_state import parameter_vector_hash, read_state, upsert_round_history, write_state

_STATE_PATH: list[str | None] = [None]


def _state_file() -> str | None:
    return _STATE_PATH[0]


from flwr.clientapp import ClientApp
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

_RUN_CFG_ROUNDS: list[int] = [2]

FEDERATION_ID = "medshield-gat-ecg-report"
TASK_ID = "gat-ecg-report-federated"
ARCHITECTURE_ID = "medshield.diagnostic_net.gat_ecg_report"
ARCHITECTURE_VERSION = "1.0.0"
FEDERATED_SCOPE = "gat_fusion_head_over_frozen_embeddings"


def context_run_config_rounds() -> int:
    return _RUN_CFG_ROUNDS[0]


def _import_gat_fl():
    from ml.federated.gat_ecg_report_fl import (
        GATFederatedClientAdapter,
        GATFLBundleError,
        create_model,
        default_bundle_path,
        load_bundle,
        save_global_checkpoint,
        state_dict_to_numpy_list,
    )

    return (
        GATFederatedClientAdapter,
        GATFLBundleError,
        create_model,
        default_bundle_path,
        load_bundle,
        save_global_checkpoint,
        state_dict_to_numpy_list,
    )


def _adapter_for_partition(partition_id: int):
    (
        GATFederatedClientAdapter,
        GATFLBundleError,
        _create_model,
        default_bundle_path,
        load_bundle,
        *_rest,
    ) = _import_gat_fl()
    try:
        bundle = load_bundle(default_bundle_path())
    except GATFLBundleError as exc:
        raise RuntimeError(str(exc)) from exc
    return GATFederatedClientAdapter(partition_id, bundle)


def client_fn(context: Context):
    from flwr.client import NumPyClient

    node = context.node_config
    partition_id = int(node.get("partition-id", node.get("partition_id", 0)))

    class _Client(NumPyClient):
        def __init__(self) -> None:
            self._adapter = _adapter_for_partition(partition_id)

        def get_parameters(self, config: dict[str, Any]):
            return self._adapter.parameter_arrays()

        def fit(self, parameters, config: dict[str, Any]):
            self._adapter.set_parameters(list(parameters))
            out, n, metrics = self._adapter.fit(config)
            return out, n, metrics

        def evaluate(self, parameters, config: dict[str, Any]):
            self._adapter.set_parameters(list(parameters))
            loss, n, metrics = self._adapter.evaluate(config)
            return float(loss), int(n), metrics

    return _Client().to_client()


client_app = ClientApp(client_fn=client_fn)


class _AuditingFedAvg(FedAvg):
    """FedAvg that persists aggregation evidence and federated GAT checkpoints."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            arrays = parameters_to_ndarrays(aggregated)
            total = int(context_run_config_rounds())
            prior = read_state(path=_state_file())
            run_id = str(
                os.environ.get("MEDSHIELD_FL_RUN_ID") or prior.get("run_id") or "unknown"
            )
            g_hash = parameter_vector_hash(arrays)
            history = upsert_round_history(
                prior.get("round_history"),
                {
                    "round": server_round,
                    "successful_clients": len(results),
                    "failed_clients": len(failures),
                    "global_parameter_hash": g_hash,
                },
                run_id=run_id,
            )
            write_state(
                {
                    "lifecycle": "COMPLETED" if server_round >= total else "RUNNING",
                    "current_round": server_round,
                    "total_rounds": total,
                    "successful_clients_last_round": len(results),
                    "failed_clients_last_round": len(failures),
                    "last_aggregation_at": aggregated_metrics,
                    "global_parameter_hash": g_hash,
                    "aggregation_status": "OK",
                    "round_history": history,
                    "task": TASK_ID,
                    "architecture_id": ARCHITECTURE_ID,
                    "federated_training_scope": FEDERATED_SCOPE,
                },
                path=_state_file(),
            )
            if server_round >= total and run_id != "unknown":
                try:
                    _, _, _, _, _, save_global_checkpoint, _ = _import_gat_fl()
                    save_global_checkpoint(run_id, arrays, round_number=server_round, parameter_hash=g_hash)
                except Exception as exc:
                    write_state(
                        {"last_error": f"checkpoint_save_failed: {exc}", "run_id": run_id},
                        path=_state_file(),
                    )
        return aggregated, aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", os.environ.get("FLOWER_NUM_ROUNDS", 2)))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    _RUN_CFG_ROUNDS[0] = num_rounds
    if os.environ.get("MEDSHIELD_FL_STATE_PATH"):
        _STATE_PATH[0] = os.environ["MEDSHIELD_FL_STATE_PATH"]

    (
        _ClientAdapter,
        GATFLBundleError,
        create_model,
        default_bundle_path,
        load_bundle,
        _save,
        state_dict_to_numpy_list,
    ) = _import_gat_fl()
    try:
        load_bundle(default_bundle_path())
    except GATFLBundleError as exc:
        write_state({"lifecycle": "FAILED", "last_error": str(exc), "task": TASK_ID}, path=_state_file())
        raise RuntimeError(str(exc)) from exc

    init_model = create_model()
    init_arrays = state_dict_to_numpy_list(init_model)
    initial = ndarrays_to_parameters(init_arrays)
    init_hash = parameter_vector_hash(init_arrays)

    run_id = os.environ.get("MEDSHIELD_FL_RUN_ID") or (
        str(context.run_id) if hasattr(context, "run_id") else None
    )
    write_state(
        {
            "lifecycle": "RUNNING",
            "run_id": run_id,
            "total_rounds": num_rounds,
            "current_round": 0,
            "initial_parameter_hash": init_hash,
            "global_parameter_hash": init_hash,
            "local_epochs": local_epochs,
            "task": TASK_ID,
            "architecture_id": ARCHITECTURE_ID,
            "architecture_version": ARCHITECTURE_VERSION,
            "federated_training_scope": FEDERATED_SCOPE,
        },
        path=_state_file(),
    )

    strategy = _AuditingFedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda _rnd: {"local_epochs": local_epochs, "batch_size": 64, "learning_rate": 1e-3},
        initial_parameters=initial,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


server_app = ServerApp(server_fn=server_fn)

"""
MedShield synthetic infrastructure Flower App (Flower 1.30 ServerApp / ClientApp).

Self-contained for FAB deployment: no imports from repo-root ``fl`` package.
Non-clinical: tiny linear model only. No patient data leaves client processes.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from run_state import parameter_vector_hash, write_state

_STATE_PATH: list[str | None] = [None]


def _state_file() -> str | None:
    return _STATE_PATH[0]

from flwr.clientapp import ClientApp
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

_RUN_CFG_ROUNDS: list[int] = [2]

FEDERATION_ID = "medshield-synthetic"
TASK_ID = "infrastructure-smoke"
ARCHITECTURE_ID = "linear-4"
ARCHITECTURE_VERSION = "1"


def context_run_config_rounds() -> int:
    return _RUN_CFG_ROUNDS[0]


class _SyntheticAdapter:
    """Tiny linear model for FL infrastructure tests only (not clinical)."""

    def __init__(self, *, seed: int, num_examples: int) -> None:
        self._num_examples = num_examples
        rng = np.random.default_rng(seed)
        self._w = rng.normal(size=(4, 1)).astype(np.float32)
        self._b = rng.normal(size=(1,)).astype(np.float32)

    def parameter_arrays(self) -> list[np.ndarray]:
        return [self._w.copy(), self._b.copy()]

    def set_parameters(self, arrays: list[np.ndarray]) -> None:
        self._w = np.asarray(arrays[0], dtype=np.float32).reshape(4, 1)
        self._b = np.asarray(arrays[1], dtype=np.float32).reshape(1)

    def num_local_examples(self) -> int:
        return self._num_examples

    def fit(self, config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        epochs = int(config.get("local_epochs", 1))
        lr = 0.05
        for _ in range(epochs):
            self._w += lr * np.sign(self._w) * 0.01
            self._b += lr * np.sign(self._b) * 0.01
        return self.parameter_arrays(), self._num_examples, {"loss": 0.1}


def _adapter_for_partition(partition_id: int) -> _SyntheticAdapter:
    seed = 42 + int(partition_id)
    num_examples = 120 if partition_id == 0 else 60
    return _SyntheticAdapter(seed=seed, num_examples=num_examples)


def client_fn(context: Context):
    from flwr.client import NumPyClient

    node = context.node_config
    partition_id = int(node.get("partition-id", node.get("partition_id", 0)))
    adapter = _adapter_for_partition(partition_id)

    class _Client(NumPyClient):
        def get_parameters(self, config: dict[str, Any]):
            return adapter.parameter_arrays()

        def fit(self, parameters, config: dict[str, Any]):
            adapter.set_parameters(list(parameters))
            out, n, metrics = adapter.fit(config)
            return out, n, metrics

        def evaluate(self, parameters, config: dict[str, Any]):
            adapter.set_parameters(list(parameters))
            return 0.0, adapter.num_local_examples(), {}

    return _Client().to_client()


client_app = ClientApp(client_fn=client_fn)


class _AuditingFedAvg(FedAvg):
    """FedAvg that persists aggregation evidence for the control plane."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            arrays = parameters_to_ndarrays(aggregated)
            total = int(context_run_config_rounds())
            from run_state import read_state

            prior = read_state(path=_state_file())
            history = list(prior.get("round_history") or [])
            history.append(
                {
                    "round": server_round,
                    "successful_clients": len(results),
                    "failed_clients": len(failures),
                    "global_parameter_hash": parameter_vector_hash(arrays),
                }
            )
            write_state(
                {
                    "lifecycle": "COMPLETED" if server_round >= total else "RUNNING",
                    "current_round": server_round,
                    "total_rounds": total,
                    "successful_clients_last_round": len(results),
                    "failed_clients_last_round": len(failures),
                    "last_aggregation_at": aggregated_metrics,
                    "global_parameter_hash": parameter_vector_hash(arrays),
                    "aggregation_status": "OK",
                    "round_history": history,
                },
                path=_state_file(),
            )
        return aggregated, aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", os.environ.get("FLOWER_NUM_ROUNDS", 2)))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    _RUN_CFG_ROUNDS[0] = num_rounds
    if os.environ.get("MEDSHIELD_FL_STATE_PATH"):
        _STATE_PATH[0] = os.environ["MEDSHIELD_FL_STATE_PATH"]

    adapter0 = _adapter_for_partition(0)
    init_arrays = adapter0.parameter_arrays()
    initial = ndarrays_to_parameters(init_arrays)
    init_hash = parameter_vector_hash(init_arrays)

    write_state(
        {
            "lifecycle": "RUNNING",
            "run_id": str(context.run_id) if hasattr(context, "run_id") else None,
            "total_rounds": num_rounds,
            "current_round": 0,
            "initial_parameter_hash": init_hash,
            "global_parameter_hash": init_hash,
            "local_epochs": local_epochs,
            "task": "synthetic_infrastructure",
            "architecture_id": ARCHITECTURE_ID,
        },
        path=_state_file(),
    )

    strategy = _AuditingFedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda _rnd: {"local_epochs": local_epochs},
        initial_parameters=initial,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


server_app = ServerApp(server_fn=server_fn)

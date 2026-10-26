"""Deprecated in-process synthetic loop (tests only)."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_RUN_STATE: Dict[str, Any] = {
    "lifecycle": "SERVER_STOPPED",
    "run_id": None,
    "current_round": 0,
    "total_rounds": 0,
    "registered_clients": 0,
    "selected_clients_last_round": 0,
    "successful_clients_last_round": 0,
    "failed_clients_last_round": 0,
    "last_aggregation_at": None,
    "last_error": None,
    "last_metrics": None,
    "history": [],
}


def _reset_run_state() -> None:
    _RUN_STATE.update(
        {
            "lifecycle": "SERVER_STOPPED",
            "run_id": None,
            "current_round": 0,
            "total_rounds": 0,
            "registered_clients": 0,
            "selected_clients_last_round": 0,
            "successful_clients_last_round": 0,
            "failed_clients_last_round": 0,
            "last_aggregation_at": None,
            "last_error": None,
            "last_metrics": None,
        }
    )


def _run_synthetic_federation(*, run_id: str, num_rounds: int) -> None:
    try:
        from fl.aggregation.fedavg import weighted_fedavg
        from fl.client.adapter import SyntheticInfrastructureAdapter
        from fl.config import default_tabular_infrastructure_config

        cfg = default_tabular_infrastructure_config()
        cfg.num_rounds = max(1, num_rounds)
        cfg.synthetic_infrastructure_smoke = True

        adapters = [
            SyntheticInfrastructureAdapter(
                federation_id=cfg.federation_id,
                task_id=cfg.task_id,
                architecture_id=cfg.architecture_id,
                architecture_version=cfg.architecture_version,
                seed=cfg.seed + i,
                num_examples=80 if i == 0 else 40,
            )
            for i in range(2)
        ]

        with _STATE_LOCK:
            _RUN_STATE["registered_clients"] = len(adapters)
            _RUN_STATE["lifecycle"] = "RUNNING"
            _RUN_STATE["total_rounds"] = cfg.num_rounds

        schema = adapters[0].parameter_schema()
        global_arrays, names = adapters[0].get_parameters()

        for round_idx in range(1, cfg.num_rounds + 1):
            with _STATE_LOCK:
                if _RUN_STATE["lifecycle"] == "STOPPING":
                    _RUN_STATE["lifecycle"] = "SERVER_STOPPED"
                    return
                _RUN_STATE["current_round"] = round_idx

            updates = []
            metrics_acc: list[float] = []
            failed = 0
            for adapter in adapters:
                try:
                    adapter.set_parameters(global_arrays, names)
                    arrays, n, metrics = adapter.fit({"local_epochs": 1, "loss": 0.42})
                    updates.append((arrays, n))
                    metrics_acc.append(float(metrics.get("loss", 0.0)))
                except Exception:
                    failed += 1

            if len(updates) < 1:
                with _STATE_LOCK:
                    _RUN_STATE["lifecycle"] = "FAILED"
                    _RUN_STATE["last_error"] = "All clients failed during fit."
                return

            global_arrays = weighted_fedavg(updates, schema=schema)
            now = datetime.now(timezone.utc).isoformat()
            with _STATE_LOCK:
                _RUN_STATE["selected_clients_last_round"] = len(adapters)
                _RUN_STATE["successful_clients_last_round"] = len(updates)
                _RUN_STATE["failed_clients_last_round"] = failed
                _RUN_STATE["last_aggregation_at"] = now
                _RUN_STATE["last_metrics"] = {
                    "mean_client_loss": sum(metrics_acc) / max(len(metrics_acc), 1),
                    "round": round_idx,
                }
                _RUN_STATE["history"].append(
                    {
                        "round": round_idx,
                        "aggregated_at": now,
                        "successful_clients": len(updates),
                        "failed_clients": failed,
                    }
                )

        with _STATE_LOCK:
            _RUN_STATE["lifecycle"] = "COMPLETED"
    except Exception as exc:
        logger.exception("Flower synthetic run failed")
        with _STATE_LOCK:
            _RUN_STATE["lifecycle"] = "FAILED"
            _RUN_STATE["last_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"


def start(num_rounds: int, *, blocking: bool = False) -> dict[str, Any]:
    with _STATE_LOCK:
        if _RUN_STATE["lifecycle"] == "RUNNING":
            raise RuntimeError("A federated run is already in progress.")
        run_id = str(uuid.uuid4())
        _RUN_STATE["run_id"] = run_id
        _RUN_STATE["history"] = []
        _RUN_STATE["lifecycle"] = "WAITING_FOR_CLIENTS"
        _RUN_STATE["last_error"] = None

    if blocking:
        _run_synthetic_federation(run_id=run_id, num_rounds=num_rounds)
        return {"run_id": run_id, "status": "COMPLETED", "num_rounds": num_rounds}

    thread = threading.Thread(
        target=_run_synthetic_federation,
        kwargs={"run_id": run_id, "num_rounds": num_rounds},
        name=f"flower-legacy-{run_id[:8]}",
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "status": "STARTED", "num_rounds": num_rounds}


def stop() -> dict[str, Any]:
    with _STATE_LOCK:
        if _RUN_STATE["lifecycle"] != "RUNNING":
            return {"status": "NO_ACTIVE_RUN", "lifecycle": _RUN_STATE["lifecycle"]}
        _RUN_STATE["lifecycle"] = "STOPPING"
    return {"status": "STOP_REQUESTED"}


def export_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return dict(_RUN_STATE)

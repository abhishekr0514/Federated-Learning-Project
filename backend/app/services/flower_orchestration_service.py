"""Flower orchestration control plane (real SuperLink deployment + legacy in-process smoke)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, Optional

from backend.app.core.config import settings
from backend.app.services.flower_supervisor import FlowerSupervisor
from fl.flower_deploy.run_state import normalize_round_history

logger = logging.getLogger(__name__)


def _flwr_available() -> bool:
    return importlib.util.find_spec("flwr") is not None


def _flwr_version() -> Optional[str]:
    if not _flwr_available():
        return None
    import flwr

    return flwr.__version__


class FlowerOrchestrationService:
    @staticmethod
    def _completed_run_with_aggregation(file_state: Dict[str, Any]) -> bool:
        total = int(file_state.get("total_rounds") or 2)
        return str(file_state.get("lifecycle") or "").upper() == "COMPLETED" and FlowerSupervisor._aggregation_evidence(
            file_state, num_rounds=total
        )

    @staticmethod
    def _demo_supernodes(
        *,
        snap: Dict[str, Any],
        file_state: Dict[str, Any],
        lifecycle: str,
    ) -> list[Dict[str, Any]]:
        """Two synthetic demo partitions (non-clinical); maps to SuperNode processes + Control API node info."""
        reg_list = list(file_state.get("registered_nodes") or [])
        procs = {p.get("name"): p for p in snap.get("processes", [])}
        run_completed = lifecycle == "COMPLETED" and FlowerOrchestrationService._completed_run_with_aggregation(
            file_state
        )
        nodes: list[Dict[str, Any]] = []
        for i in range(2):
            label = "Demo Hospital A" if i == 0 else "Demo Hospital B"
            code = f"partition-{i}"
            proc = procs.get(f"supernode-{i}")
            reg = reg_list[i] if i < len(reg_list) else {}
            process_alive = bool(proc and proc.get("alive"))
            fleet_status = str(reg.get("status") or "unknown").lower() if reg else "unknown"
            if process_alive:
                raw = fleet_status if fleet_status != "unknown" else "connecting"
                connection = raw if raw in ("online", "registered", "offline") else "connecting"
                participation_status = "live"
            elif run_completed and reg.get("node_id") is not None:
                participation_status = "completed_run"
                connection = "offline"
            else:
                participation_status = "idle"
                connection = "offline"
            nodes.append(
                {
                    "demo_id": f"supernode-{i}",
                    "display_name": label,
                    "code": code,
                    "partition_id": i,
                    "node_id": reg.get("node_id"),
                    "connection_status": connection,
                    "participation_status": participation_status,
                    "fleet_status": fleet_status if reg else None,
                    "process_pid": proc.get("pid") if proc else None,
                    "process_alive": process_alive,
                }
            )
        return nodes

    @staticmethod
    def get_status() -> Dict[str, Any]:
        dependency_installed = _flwr_available()
        orchestration_enabled = settings.FLOWER_ORCHESTRATION_ENABLED
        training_enabled = settings.FLOWER_TRAINING_ENABLED
        sec_ok, sec_blockers = FlowerSupervisor.security_readiness()
        snap = FlowerSupervisor.get_snapshot()
        file_state = snap.get("file_state") or {}
        lifecycle = snap.get("lifecycle") or file_state.get("lifecycle") or "SERVER_STOPPED"

        if not dependency_installed:
            summary = "MISSING_DEPENDENCY"
            reason = "Flower (flwr) Python package is not installed in this runtime."
        elif not orchestration_enabled:
            summary = "DISABLED_BY_CONFIGURATION"
            reason = "FLOWER_ORCHESTRATION_ENABLED is false."
        elif training_enabled and settings.FLOWER_REQUIRE_TLS and not sec_ok:
            summary = "SECURITY_NOT_CONFIGURED"
            reason = "; ".join(sec_blockers) or "TLS / SuperNode auth material missing."
        elif not training_enabled:
            summary = "DISABLED_BY_CONFIGURATION"
            reason = "FLOWER_TRAINING_ENABLED is false; orchestration idle."
        elif lifecycle == "RUNNING":
            summary = "RUNNING"
            reason = "Flower SuperLink run in progress."
        elif lifecycle == "COMPLETED":
            summary = "COMPLETED"
            reason = "Last federated run completed successfully."
        elif lifecycle == "FAILED":
            summary = "FAILED"
            reason = snap.get("last_error") or file_state.get("last_error") or "Run failed."
        elif lifecycle == "STOPPING":
            summary = "STOPPING"
            reason = "Run stop requested."
        elif lifecycle in ("WAITING_FOR_CLIENTS", "STARTING"):
            summary = "WAITING_FOR_CLIENTS"
            reason = "SuperLink started; waiting for SuperNodes and run scheduling."
        else:
            summary = "SERVER_STOPPED"
            reason = "No active SuperLink run."

        alive = [p for p in snap.get("processes", []) if p.get("alive")]
        demo_supernodes = FlowerOrchestrationService._demo_supernodes(
            snap=snap,
            file_state=file_state,
            lifecycle=lifecycle,
        )
        demo_nodes_live_count = sum(1 for n in demo_supernodes if n.get("process_alive"))
        demo_nodes_participated_count = sum(
            1 for n in demo_supernodes if n.get("participation_status") == "completed_run"
        )
        post_run_processes_stopped = (
            lifecycle == "COMPLETED"
            and demo_nodes_live_count == 0
            and demo_nodes_participated_count >= 2
        )
        if post_run_processes_stopped:
            reason = (
                "Federated run completed successfully. Local SuperLink and SuperNode processes were "
                "stopped by the supervisor after the run (expected demo teardown)."
            )
        return {
            "summary": summary,
            "reason": reason,
            "dependency_installed": dependency_installed,
            "flower_version": _flwr_version(),
            "deployment_mode": settings.FLOWER_DEPLOYMENT_MODE,
            "orchestration_enabled": orchestration_enabled,
            "training_enabled": training_enabled,
            "tls_required": settings.FLOWER_REQUIRE_TLS,
            "security_configured": sec_ok,
            "security_blockers": sec_blockers,
            "server_process_available": orchestration_enabled and dependency_installed and sec_ok,
            "superlink_control_address": settings.FLOWER_SUPERLINK_CONTROL_ADDRESS,
            "fleet_api_address": settings.FLOWER_FLEET_API_ADDRESS,
            "server_pid": alive[0]["pid"] if alive else None,
            "processes": snap.get("processes", []),
            "client_connected": len([p for p in alive if str(p.get("name", "")).startswith("supernode")]),
            "aggregation_configured": orchestration_enabled,
            "training_round_active": lifecycle == "RUNNING",
            "lifecycle": lifecycle,
            "run_id": snap.get("run_id"),
            "current_round": file_state.get("current_round", 0),
            "total_rounds": file_state.get("total_rounds"),
            "registered_clients": len([p for p in snap.get("processes", []) if "supernode" in p.get("name", "")]),
            "selected_clients_last_round": file_state.get("successful_clients_last_round"),
            "successful_clients_last_round": file_state.get("successful_clients_last_round"),
            "failed_clients_last_round": file_state.get("failed_clients_last_round"),
            "last_successful_round": file_state.get("last_aggregation_at"),
            "participating_clients_last_round": file_state.get("successful_clients_last_round"),
            "last_metrics": file_state.get("last_aggregation_at"),
            "initial_parameter_hash": file_state.get("initial_parameter_hash"),
            "global_parameter_hash": file_state.get("global_parameter_hash"),
            "global_model_version": file_state.get("architecture_id"),
            "last_error": snap.get("last_error"),
            "supervisor_phase_failed": bool(file_state.get("supervisor_phase_failed")),
            "aggregation_succeeded": FlowerSupervisor._aggregation_evidence(
                file_state,
                num_rounds=int(file_state.get("total_rounds") or 2),
            ),
            "history": normalize_round_history(file_state.get("round_history")),
            "demo_supernodes": demo_supernodes,
            "demo_nodes_live_count": demo_nodes_live_count,
            "demo_nodes_participated_count": demo_nodes_participated_count,
            "post_run_processes_stopped": post_run_processes_stopped,
            "demo_local_setup_hint": (
                "Local SuperLink demo: set FLOWER_ORCHESTRATION_ENABLED=true, "
                "FLOWER_TRAINING_ENABLED=true, FLOWER_DEPLOYMENT_MODE=superlink, "
                "FLOWER_REQUIRE_TLS=true; run scripts/flower/generate_dev_materials.ps1; restart API."
            ),
            "secure_aggregation": False,
            "differential_privacy": False,
            "note": (
                "Research FedAvg on MedShieldGATECGReportNet (fusion head over frozen PTB-XL embeddings) "
                "via Flower SuperLink + two local SuperNodes. No secure aggregation or DP. "
                "Simulated hospitals ≠ production multi-site deployment."
            ),
        }

    @staticmethod
    def start_run(*, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        if not settings.FLOWER_ORCHESTRATION_ENABLED:
            raise RuntimeError("Flower orchestration is disabled (FLOWER_ORCHESTRATION_ENABLED=false).")
        if not settings.FLOWER_TRAINING_ENABLED:
            raise RuntimeError("Flower training is disabled (FLOWER_TRAINING_ENABLED=false).")
        if settings.FLOWER_DEPLOYMENT_MODE == "superlink":
            from ml.federated.gat_ecg_report_fl import bundle_readiness

            ok, blockers = bundle_readiness()
            if not ok:
                raise RuntimeError("GAT_FL_NOT_READY: " + "; ".join(blockers))
        if not _flwr_available():
            raise RuntimeError("Flower (flwr) is not installed.")
        rounds = num_rounds or settings.FLOWER_NUM_ROUNDS
        if settings.FLOWER_DEPLOYMENT_MODE == "legacy_inprocess":
            return _start_legacy_inprocess(rounds)
        return FlowerSupervisor.start(num_rounds=rounds)

    @staticmethod
    def stop_run() -> Dict[str, Any]:
        if settings.FLOWER_DEPLOYMENT_MODE == "legacy_inprocess":
            return _stop_legacy_inprocess()
        return FlowerSupervisor.stop()

    @staticmethod
    def run_synthetic_smoke_blocking(*, num_rounds: int = 1) -> Dict[str, Any]:
        """Synchronous legacy in-process smoke (unit tests only — not a real SuperLink run)."""
        if not _flwr_available():
            raise RuntimeError("flwr not installed")
        _start_legacy_inprocess(num_rounds, blocking=True)
        from backend.app.services.flower_orchestration_legacy import export_state

        legacy = export_state()
        return {
            **FlowerOrchestrationService.get_status(),
            "lifecycle": legacy.get("lifecycle"),
            "successful_clients_last_round": legacy.get("successful_clients_last_round"),
            "last_successful_round": legacy.get("last_aggregation_at"),
            "last_error": legacy.get("last_error"),
        }


def _start_legacy_inprocess(num_rounds: int, *, blocking: bool = False) -> Dict[str, Any]:
    from backend.app.services import flower_orchestration_legacy as legacy

    return legacy.start(num_rounds, blocking=blocking)


def _stop_legacy_inprocess() -> Dict[str, Any]:
    from backend.app.services import flower_orchestration_legacy as legacy

    return legacy.stop()

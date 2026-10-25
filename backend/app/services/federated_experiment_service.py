"""Read-only access to saved tabular federated experiment artifacts (no raw client data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.config import settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _federated_runs_root() -> Path:
    return _repo_root() / "data" / "processed" / "federated" / "tabular_cardio_runs"


def list_tabular_run_ids() -> list[str]:
    root = _federated_runs_root()
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _sanitize_partition_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove per-record hash lists; keep aggregate counts only."""
    return {
        "seed": plan.get("seed"),
        "num_clients": plan.get("num_clients"),
        "max_federated_train_rows": plan.get("max_federated_train_rows"),
        "simulation_disclaimer": plan.get("simulation_disclaimer"),
        "client_train_counts": plan.get("client_train_counts"),
        "val_row_count": plan.get("val_row_count"),
        "test_row_count": plan.get("test_row_count"),
        "train_row_count_total": plan.get("train_row_count_total"),
    }


def _sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    participation = cfg.get("participation", {})
    local = cfg.get("local_training", {})
    eval_block = report.get("evaluation", {})
    test_m = eval_block.get("test", {})
    val_m = eval_block.get("validation", {})
    baseline = report.get("centralized_baseline_same_train_rows", {})

    return {
        "experiment": report.get("experiment"),
        "privacy_note": report.get("privacy_note"),
        "simulation_disclaimer": report.get("simulation_disclaimer"),
        "config_summary": {
            "federation_id": cfg.get("federation_id"),
            "task_id": cfg.get("task_id"),
            "architecture_id": cfg.get("architecture_id"),
            "seed": cfg.get("seed"),
            "num_rounds": cfg.get("num_rounds"),
            "max_federated_train_rows": cfg.get("max_federated_train_rows"),
            "num_clients": participation.get("num_clients"),
            "local_epochs": local.get("local_epochs"),
            "aggregation_strategy": cfg.get("aggregation_strategy"),
        },
        "partition_plan_summary": _sanitize_partition_plan(report.get("partition_plan", {})),
        "round_history": report.get("round_history", []),
        "parameter_l2_initial": report.get("parameter_l2_initial"),
        "parameter_l2_final": report.get("parameter_l2_final"),
        "evaluation_summary": {
            "test_auroc": test_m.get("auroc"),
            "test_auprc": test_m.get("auprc"),
            "test_f1": test_m.get("f1"),
            "validation_auroc": val_m.get("auroc"),
            "youden_threshold": eval_block.get("youden_threshold"),
        },
        "baseline_comparison": {
            "centralized_test_auroc": baseline.get("test", {}).get("auroc")
            if isinstance(baseline, dict)
            else None,
            "centralized_total_epochs": baseline.get("total_epochs")
            if isinstance(baseline, dict)
            else None,
        },
    }


def _checkpoint_verification(
    run_dir: Path,
    report: dict[str, Any],
    ckpt_path: Path,
) -> dict[str, Any]:
    """Lightweight verification without exposing filesystem paths in API."""
    expected_rounds = report.get("config", {}).get("num_rounds")
    history_len = len(report.get("round_history", []))
    result: dict[str, Any] = {
        "report_present": True,
        "checkpoint_present": ckpt_path.is_file(),
        "rounds_match": expected_rounds == history_len,
        "parameters_changed": report.get("parameter_l2_initial") != report.get("parameter_l2_final"),
    }
    if not ckpt_path.is_file():
        result["ok"] = False
        return result
    try:
        import torch
        from ml.models.tabular_encoder import TabularBinaryClassifier
        from fl.parameters.schema import try_schema_from_torch_state_dict

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = payload.get("metadata", {})
        schema = try_schema_from_torch_state_dict(
            TabularBinaryClassifier().state_dict(),
            architecture_id="medshield.tabular.cardio_survey.classifier",
            architecture_version="1.0.0",
        )
        result["trained"] = bool(meta.get("trained"))
        result["round_number"] = meta.get("round_number")
        result["schema_signature_match"] = meta.get("parameter_schema_signature") == schema.signature()
        result["ok"] = (
            result["trained"]
            and result["rounds_match"]
            and result["parameters_changed"]
            and result["schema_signature_match"]
        )
    except Exception:
        result["ok"] = False
        result["error"] = "verification_failed"
    return result


def get_tabular_experiment_status(*, run_id: str | None = None) -> dict[str, Any]:
    runs = list_tabular_run_ids()
    if not runs:
        return {
            "status": "NO_RUNS",
            "message": "No saved tabular federated runs found under data/processed/federated.",
            "runs_available": [],
        }

    selected = run_id if run_id in runs else runs[-1]
    run_dir = _federated_runs_root() / selected
    report_path = run_dir / "report.json"
    ckpt_path = run_dir / "global_federated_checkpoint.pt"

    if not report_path.is_file():
        return {
            "status": "INCOMPLETE",
            "run_id": selected,
            "message": "Report missing for run.",
            "runs_available": runs,
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    sanitized = _sanitize_report(report)

    checkpoint_meta: dict[str, Any] | None = None
    if ckpt_path.is_file():
        try:
            import torch

            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            meta = payload.get("metadata", {})
            checkpoint_meta = {
                "trained": meta.get("trained"),
                "round_number": meta.get("round_number"),
                "architecture_id": meta.get("architecture_id"),
                "flower_version": meta.get("flower_version"),
            }
        except Exception:
            checkpoint_meta = {"error": "checkpoint_unreadable"}

    verification = _checkpoint_verification(run_dir, report, ckpt_path)

    return {
        "status": "AVAILABLE",
        "run_id": selected,
        "experiment_timestamp_utc": selected if len(selected) >= 8 else None,
        "runs_available_count": len(runs),
        "checkpoint_summary": checkpoint_meta,
        "checkpoint_verification": verification,
        "report": sanitized,
        "inference_enabled": settings.INFERENCE_ENABLED,
        "privacy_limitations": (
            "FedAvg exchanges model weights and allowlisted scalar metrics only. "
            "This is not differential privacy or secure aggregation. "
            "Simulated clients are slices of cardio_train.csv, not verified hospitals."
        ),
        "note": "Survey-target research metrics only; not clinical deployment.",
    }

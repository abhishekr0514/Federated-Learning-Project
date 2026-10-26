"""Read-only access to saved GAT federated Flower run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.config import settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _gat_runs_root() -> Path:
    return _repo_root() / "data" / "processed" / "federated" / "gat_ecg_report_runs"


def list_gat_run_ids() -> list[str]:
    root = _gat_runs_root()
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    participation = cfg.get("participation", {})
    local = cfg.get("local_training", {})
    eval_block = report.get("evaluation", {})
    test_m = eval_block.get("test", {})
    baseline = report.get("centralized_baseline_same_test_embeddings", {})

    test_auroc = test_m.get("auroc")
    test_auroc_reason = None
    if test_auroc is None and test_m.get("error"):
        test_auroc_reason = str(test_m.get("error"))
    elif test_auroc is None and not test_m:
        test_auroc_reason = "Held-out evaluation not recorded for this run."

    baseline_test = baseline.get("test") if isinstance(baseline, dict) else None
    baseline_auroc = baseline_test.get("auroc") if isinstance(baseline_test, dict) else None

    return {
        "experiment": report.get("experiment"),
        "federated_training_scope": report.get("federated_training_scope"),
        "target_definition": report.get("target_definition"),
        "privacy_note": report.get("privacy_note"),
        "simulation_disclaimer": report.get("simulation_disclaimer"),
        "config_summary": {
            "federation_id": cfg.get("federation_id"),
            "task_id": cfg.get("task_id"),
            "architecture_id": cfg.get("architecture_id") or report.get("architecture_id"),
            "seed": cfg.get("seed"),
            "num_rounds": cfg.get("num_rounds"),
            "num_clients": participation.get("num_clients"),
            "local_epochs": local.get("local_epochs"),
            "aggregation_strategy": cfg.get("aggregation_strategy"),
        },
        "partition_plan_summary": report.get("partition_plan"),
        "round_history": report.get("round_history", []),
        "parameter_hash_initial": report.get("parameter_l2_initial"),
        "parameter_hash_final": report.get("parameter_l2_final"),
        "evaluation_summary": {
            "test_auroc": test_auroc,
            "test_auroc_unavailable_reason": test_auroc_reason,
            "test_n_samples": test_m.get("n_samples"),
            "test_class_counts": test_m.get("class_counts"),
            "held_out_split": eval_block.get("held_out_split"),
            "evaluated_at_utc": eval_block.get("evaluated_at_utc"),
        },
        "baseline_comparison": {
            "centralized_test_auroc": baseline_auroc,
            "note": baseline.get("note") if isinstance(baseline, dict) else None,
            "valid_same_test_split": baseline_auroc is not None and test_auroc is not None,
        },
        "checkpoint_summary": report.get("checkpoint_summary"),
    }


def _checkpoint_verification(report: dict[str, Any], ckpt_path: Path) -> dict[str, Any]:
    expected_rounds = report.get("config", {}).get("num_rounds")
    history_len = len(report.get("round_history", []))
    init_h = report.get("parameter_l2_initial")
    final_h = report.get("parameter_l2_final")
    result: dict[str, Any] = {
        "report_present": True,
        "checkpoint_present": ckpt_path.is_file(),
        "rounds_match": expected_rounds == history_len if expected_rounds is not None else None,
        "parameters_changed": bool(init_h and final_h and init_h != final_h),
    }
    if not ckpt_path.is_file():
        result["ok"] = False
        return result
    try:
        from ml.models.gat_ecg_report_fusion import ARCHITECTURE_ID, checkpoint_summary

        summary = checkpoint_summary(ckpt_path)
        result["architecture_id"] = summary.get("architecture_id")
        result["trained"] = summary.get("trained")
        result["ok"] = (
            summary.get("architecture_id") == ARCHITECTURE_ID
            and result["parameters_changed"]
            and result["checkpoint_present"]
        )
    except Exception:
        result["ok"] = False
        result["error"] = "verification_failed"
    return result


def get_gat_federated_experiment_status(*, run_id: str | None = None) -> dict[str, Any]:
    runs = list_gat_run_ids()
    if not runs:
        return {
            "status": "NO_RUNS",
            "message": (
                "No saved GAT federated runs found. Complete a SuperLink run with a valid "
                "embedding bundle (scripts/build_gat_fl_embedding_bundle.py)."
            ),
            "runs_available": [],
        }

    selected = run_id if run_id in runs else runs[-1]
    run_dir = _gat_runs_root() / selected
    report_path = run_dir / "report.json"
    ckpt_path = run_dir / "global_federated_checkpoint.pt"

    if not report_path.is_file():
        return {
            "status": "INCOMPLETE",
            "run_id": selected,
            "message": "Report missing for GAT federated run.",
            "runs_available": runs,
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    sanitized = _sanitize_report(report)
    verification = _checkpoint_verification(report, ckpt_path)

    return {
        "status": "AVAILABLE",
        "run_id": selected,
        "flower_run_id": report.get("flower_run_id", selected),
        "runs_available_count": len(runs),
        "checkpoint_verification": verification,
        "report": sanitized,
        "inference_enabled": settings.INFERENCE_ENABLED,
        "privacy_limitations": report.get("privacy_note"),
        "note": (
            "Research demonstration on PTB-XL proxy labels with simulated hospitals — "
            "not clinical performance or multi-site deployment evidence."
        ),
    }

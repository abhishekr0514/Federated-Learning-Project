"""Load empirical GAT evaluation summaries from reports/model_evaluation/ (no fabrication)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REPORT_DIR = _REPO_ROOT / "reports" / "model_evaluation"

_SUMMARY_MODEL_KEYS = (
    "n_samples",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "class_1_precision",
    "class_1_recall_sensitivity",
    "class_0_specificity",
    "auroc",
    "average_precision",
    "mcc",
    "confusion_matrix_raw",
    "score_semantics",
    "confidence_note",
    "positive_class_label",
)


def _summarize_model_block(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _SUMMARY_MODEL_KEYS:
        if key in raw:
            out[key] = raw[key]
    return out


def load_offline_gat_evaluation_summary(
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Read evaluation_manifest.json + model_evaluation_metrics.json produced by
    scripts/generate_model_evaluation_report.py.
    """
    root = (report_dir or _DEFAULT_REPORT_DIR).resolve()
    manifest_path = root / "evaluation_manifest.json"
    metrics_path = root / "model_evaluation_metrics.json"

    if not manifest_path.is_file() or not metrics_path.is_file():
        return {
            "status": "REPORT_NOT_FOUND",
            "message": (
                "No offline evaluation report on disk. Generate artifacts with: "
                "python scripts/generate_model_evaluation_report.py"
            ),
            "expected_directory": str(root.relative_to(_REPO_ROOT)) if root.is_relative_to(_REPO_ROOT) else str(root),
            "metrics": None,
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "status": "REPORT_INVALID",
            "message": f"Could not parse evaluation report JSON: {type(exc).__name__}",
            "metrics": None,
        }

    manifest_inner = metrics_doc.get("manifest") or manifest
    models_raw = metrics_doc.get("models") or {}
    models_out: Dict[str, Any] = {}
    for model_id, block in models_raw.items():
        if isinstance(block, dict):
            models_out[model_id] = _summarize_model_block(block)

    dataset = manifest_inner.get("dataset") or {}
    dataset_public = {
        "test_in_bundle": dataset.get("split_counts", {}).get("test_in_bundle"),
        "train_in_bundle": dataset.get("split_counts", {}).get("train_in_bundle"),
        "test_eligible_full_split": dataset.get("split_counts", {}).get("test_eligible_full_split"),
        "test_split_mode": dataset.get("test_split_mode"),
        "smoke_caps": dataset.get("smoke_caps"),
        "target_definition": dataset.get("target_definition") or manifest_inner.get("target_definition"),
        "split_policy": dataset.get("split_policy"),
    }

    return {
        "status": "AVAILABLE",
        "source": "offline_holdout_evaluation",
        "generated_at_utc": manifest_inner.get("generated_at_utc"),
        "class_mapping": manifest_inner.get("class_mapping"),
        "research_disclaimer": manifest_inner.get("research_disclaimer"),
        "comparable_examples": manifest_inner.get("comparable_examples"),
        "dataset": dataset_public,
        "checkpoints": manifest_inner.get("models"),
        "models": models_out,
        "report_artifacts": {
            "directory": str(root.relative_to(_REPO_ROOT)) if root.is_relative_to(_REPO_ROOT) else str(root),
            "files": {
                "html": "model_evaluation_report.html",
                "manifest": "evaluation_manifest.json",
                "metrics_json": "model_evaluation_metrics.json",
                "metrics_csv": "model_evaluation_metrics.csv",
                "comparison_plot": "model_comparison.png",
                "confusion_centralized": "gat_centralized_confusion_matrix.png",
                "confusion_federated": "gat_fusion_confusion_matrix.png",
                "roc_plot": "fl_model_roc_curve.png",
            },
        },
        "metrics": models_out,
    }

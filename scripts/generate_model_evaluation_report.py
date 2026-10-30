"""Generate reproducible GAT centralized vs federated evaluation report artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.gat_bundle_evaluator import (  # noqa: E402
    DEFAULT_CENTRALIZED_CKPT,
    build_evaluation_manifest,
    evaluate_checkpoint_on_bundle,
    find_latest_fl_checkpoint,
)
from ml.evaluation.report_artifacts import write_evaluation_artifacts  # noqa: E402
from ml.federated.gat_ecg_report_fl import default_bundle_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MedShield GAT / FL evaluation report artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "model_evaluation",
        help="Directory for HTML, CSV, JSON, and plots.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Recorded for reproducibility (inference is deterministic).")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to federated_fl_bundle.pt (default: data/processed/gat_ecg_report/federated_fl_bundle.pt).",
    )
    parser.add_argument(
        "--gat-checkpoint",
        type=Path,
        default=DEFAULT_CENTRALIZED_CKPT,
        help="Centralized GAT checkpoint path.",
    )
    parser.add_argument(
        "--fl-checkpoint",
        type=Path,
        default=None,
        help="Federated global checkpoint (.pt). Default: newest under gat_ecg_report_runs/.",
    )
    parser.add_argument("--skip-centralized", action="store_true")
    parser.add_argument("--skip-federated", action="store_true")
    args = parser.parse_args()

    bundle_path = args.bundle or default_bundle_path()
    if not bundle_path.is_file():
        print(json.dumps({"ok": False, "error": f"Missing bundle: {bundle_path}"}))
        return 2

    centralized = None
    federated = None
    errors: list[str] = []

    if not args.skip_centralized:
        try:
            centralized = evaluate_checkpoint_on_bundle(
                args.gat_checkpoint,
                model_role="GAT centralized (ECG+report fusion head)",
                bundle_path=bundle_path,
            )
        except Exception as exc:
            errors.append(f"centralized: {type(exc).__name__}: {exc}")

    fl_ckpt = args.fl_checkpoint
    fl_note = "explicit --fl-checkpoint"
    if not args.skip_federated:
        if fl_ckpt is None:
            fl_ckpt = find_latest_fl_checkpoint()
            fl_note = "newest global_federated_checkpoint.pt by mtime under gat_ecg_report_runs/"
        if fl_ckpt is None or not Path(fl_ckpt).is_file():
            errors.append("federated: no global_federated_checkpoint.pt found")
        else:
            try:
                federated = evaluate_checkpoint_on_bundle(
                    Path(fl_ckpt),
                    model_role="GAT federated global (FedAvg)",
                    bundle_path=bundle_path,
                )
            except Exception as exc:
                errors.append(f"federated: {type(exc).__name__}: {exc}")

    if centralized is None and federated is None:
        print(json.dumps({"ok": False, "errors": errors}))
        return 1

    manifest = build_evaluation_manifest(
        centralized=centralized,
        federated=federated,
        seed=args.seed,
        output_dir=args.output_dir,
        fl_selection_note=fl_note,
    )
    manifest["generation_errors"] = errors

    paths = write_evaluation_artifacts(
        args.output_dir,
        centralized=centralized,
        federated=federated,
        manifest=manifest,
    )

    summary = {
        "ok": True,
        "output_dir": str(args.output_dir.resolve()),
        "artifacts": paths,
        "centralized_checkpoint": str(centralized.checkpoint_path) if centralized else None,
        "federated_checkpoint": str(federated.checkpoint_path) if federated else None,
        "bundle": str(bundle_path.resolve()),
        "comparable_examples": manifest.get("comparable_examples"),
        "errors": errors,
    }
    if centralized:
        summary["centralized_metrics"] = {
            k: centralized.metrics.get(k)
            for k in ("n_samples", "accuracy", "balanced_accuracy", "macro_f1", "auroc", "mcc")
        }
    if federated:
        summary["federated_metrics"] = {
            k: federated.metrics.get(k)
            for k in ("n_samples", "accuracy", "balanced_accuracy", "macro_f1", "auroc", "mcc")
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

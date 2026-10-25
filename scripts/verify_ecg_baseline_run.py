"""Verify saved PTB-XL ECG baseline run (ECG-only; not ECG+report or GAT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_latest_run(root: Path) -> Path:
    runs = root / "runs"
    dirs = sorted([p for p in runs.iterdir() if p.is_dir()]) if runs.is_dir() else []
    if not dirs:
        raise FileNotFoundError("No ECG baseline runs found")
    return dirs[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    base = REPO_ROOT / "data" / "processed" / "ecg_baseline"
    run_dir = args.run_dir or Path("data/processed/ecg_baseline/runs/20261023T112552Z")
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    report_path = run_dir / "report.json"
    ckpt_path = run_dir / "best_checkpoint.pt"
    norm_path = run_dir / "normalization.json"
    manifest_path = run_dir / "manifest.json"

    missing = [p.name for p in (report_path, ckpt_path, norm_path, manifest_path) if not p.is_file()]
    if missing:
        print(json.dumps({"ok": False, "missing": missing, "run_dir": str(run_dir)}))
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test = report.get("test_metrics", {})
    n_samples = test.get("calibration", {}).get("expected_calibration_error", {}).get("n_samples")
    cm = test.get("confusion_matrix", {})
    n_from_cm = sum(cm.get(k, 0) for k in ("tn", "fp", "fn", "tp"))

    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    has_weights = "model_state_dict" in ckpt

    out = {
        "ok": True,
        "experiment": "ecg_baseline_ptbxl_only",
        "not_ecg_report_fusion": True,
        "not_three_modality_gat": True,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "test_auroc": test.get("auroc"),
        "best_val_auroc": report.get("best_val_auroc"),
        "test_n_samples_calibration": n_samples,
        "test_n_from_confusion_matrix": n_from_cm,
        "partition_test_count": manifest.get("partition_counts", {}).get("test"),
        "checkpoint_has_weights": has_weights,
        "completed_training": report.get("completed_training"),
        "warning": report.get("warning"),
    }
    consistent = (
        n_samples == 1840
        and n_from_cm == 1840
        and abs(float(test.get("auroc", 0)) - 0.95637548207804) < 1e-9
    )
    out["metrics_consistent_with_1840_test_records"] = consistent
    print(json.dumps(out, indent=2))
    return 0 if consistent and has_weights else 3


if __name__ == "__main__":
    raise SystemExit(main())

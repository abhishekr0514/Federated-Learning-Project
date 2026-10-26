"""Compare unified test metrics across ECG+report FL experiment reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(label: str, report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    u = data.get("unified_comparison_same_records", {})
    test_n = u.get("test_n")
    return {
        "label": label,
        "run_dir": str(report_path.parent),
        "test_n": test_n,
        "pretrained": u.get("pretrained_checkpoint_unchanged", {}).get("test"),
        "centralized": u.get("centralized_trainable_scope", {}).get("test"),
        "federated": u.get("federated_global_model", {}).get("test"),
        "scope": data.get("training_mode", {}).get("federated_trainable_scope"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-a", type=Path, default=REPO_ROOT / "data/processed/federated/ecg_report_experiments/full_frozen_head_s42/report.json")
    parser.add_argument("--experiment-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.experiment_a.is_file() or not args.experiment_b.is_file():
        raise SystemExit("Missing report.json for A or B")
    out = {
        "experiment_a": _row("A", args.experiment_a),
        "experiment_b": _row("B", args.experiment_b),
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare full-cohort GAT vs federated ECG+report experiments (Youden test metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _test_row(label: str, report_path: Path, *, federated_key: str) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if federated_key == "gat_global_model":
        u = data.get("unified_comparison_same_records", {})
        test = u.get("gat_global_model", {}).get("test") or data.get("evaluation", {}).get("test")
    else:
        u = data.get("unified_comparison_same_records", {})
        test = u.get(federated_key, {}).get("test")
    return {
        "label": label,
        "report": str(report_path),
        "test_n": test.get("n_samples") if test else None,
        "metrics": test,
        "comparable_notes": data.get("comparability_notes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gat-full",
        type=Path,
        default=REPO_ROOT / "data/processed/federated/ecg_report_experiments/full_gat_s42/report.json",
    )
    parser.add_argument(
        "--gat-legacy",
        type=Path,
        default=REPO_ROOT / "data/processed/gat_ecg_report/runs/20261025T182109Z/report.json",
    )
    parser.add_argument(
        "--experiment-a",
        type=Path,
        default=REPO_ROOT / "data/processed/federated/ecg_report_experiments/full_frozen_head_s42/report.json",
    )
    parser.add_argument(
        "--experiment-b",
        type=Path,
        default=REPO_ROOT / "data/processed/federated/ecg_report_experiments/full_encoder_ft_s42/report.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    legacy = json.loads(args.gat_legacy.read_text(encoding="utf-8")) if args.gat_legacy.is_file() else {}
    legacy_test = legacy.get("test_metrics_gat", {})
    legacy_cm = legacy_test.get("confusion_matrix", {})
    legacy_acc = None
    if legacy_cm:
        tn, fp, fn, tp = legacy_cm["tn"], legacy_cm["fp"], legacy_cm["fn"], legacy_cm["tp"]
        legacy_acc = (tn + tp) / legacy.get("test_n", tn + fp + fn + tp)

    out = {
        "threshold_policy_fair_rows": "Youden on validation (FL unified + GAT full report); legacy GAT used 0.5 in report",
        "gat_full_cohort": _test_row("GAT full", args.gat_full, federated_key="gat_global_model")
        if args.gat_full.is_file()
        else {"label": "GAT full", "missing": True},
        "gat_legacy_capped_1500": {
            "label": "GAT legacy (train cap 1500)",
            "report": str(args.gat_legacy),
            "test_n": legacy.get("test_n"),
            "metrics": {
                **legacy_test,
                "accuracy": legacy_acc,
                "applied_threshold": 0.5,
                "note": "Not directly comparable: capped train, default threshold 0.5",
            },
        },
        "fl_experiment_a": _test_row("FL A fusion head", args.experiment_a, federated_key="federated_global_model"),
        "fl_experiment_b": _test_row("FL B encoder+head", args.experiment_b, federated_key="federated_global_model"),
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

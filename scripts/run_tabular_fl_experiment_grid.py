"""
Controlled hyperparameter grid for tabular FedAvg (cardio_train.csv).

Writes each run under:
  data/processed/federated/tabular_cardio_experiments/<label>/

Does not modify existing tabular_cardio_runs/* checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl.config import FederationConfig, load_federation_config  # noqa: E402
from fl.training.tabular_federated import run_tabular_federated_experiment  # noqa: E402

DEFAULT_BASE = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1_real_smoke.json"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "federated" / "tabular_cardio_experiments"


def _metric_row(report: dict) -> dict:
    test = report.get("evaluation", {}).get("test", {})
    val = report.get("evaluation", {}).get("validation", {})
    base = report.get("centralized_baseline_same_train_rows", {}).get("test", {})
    return {
        "test_accuracy": test.get("accuracy"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "test_auroc": test.get("auroc"),
        "test_auprc": test.get("auprc"),
        "test_f1": test.get("f1"),
        "test_sensitivity": test.get("sensitivity"),
        "test_specificity": test.get("specificity"),
        "test_brier": test.get("brier_score"),
        "youden_threshold": report.get("evaluation", {}).get("youden_threshold"),
        "val_auroc": val.get("auroc"),
        "baseline_test_accuracy": base.get("accuracy"),
        "baseline_test_auroc": base.get("auroc"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tabular FL hyperparameter grid")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rounds", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--local-epochs", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-4, 1e-3, 3e-3])
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default: from base config)")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max experiments (0 = all combinations)")
    args = parser.parse_args()

    if not args.confirm_run and not args.dry_run:
        print("Refusing: pass --confirm-run or --dry-run", file=sys.stderr)
        return 2

    base = load_federation_config(args.base_config)
    if args.seed is not None:
        base.seed = args.seed

    combos: list[tuple[int, int, float]] = []
    for r in args.rounds:
        for le in args.local_epochs:
            for lr in args.learning_rates:
                combos.append((r, le, lr))
    if args.limit > 0:
        combos = combos[: args.limit]

    if args.dry_run:
        print(json.dumps({"would_run": len(combos), "combos": combos[:20]}, indent=2))
        return 0

    args.output_parent.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for rounds, local_epochs, lr in combos:
        label = f"r{rounds}_e{local_epochs}_lr{lr:g}".replace(".", "p")
        run_dir = args.output_parent / label
        if (run_dir / "report.json").is_file():
            print(f"Skip existing: {label}")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            row = {"label": label, "skipped": True, **_metric_row(report)}
            summary.append(row)
            continue

        cfg = copy.deepcopy(base)
        cfg.num_rounds = rounds
        cfg.local_training.local_epochs = local_epochs
        cfg.local_training.learning_rate = lr
        cfg.participation.min_fit_clients = cfg.participation.num_clients
        cfg.participation.min_evaluate_clients = cfg.participation.num_clients
        cfg.participation.min_available_clients = cfg.participation.num_clients

        print(f"Running {label} ...")
        result = run_tabular_federated_experiment(
            cfg,
            output_root=run_dir,
            experiment_label=label,
        )
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        row = {
            "label": label,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "rounds": rounds,
            "local_epochs": local_epochs,
            "learning_rate": lr,
            "seed": cfg.seed,
            **_metric_row(report),
            "trained": result.get("trained"),
        }
        summary.append(row)

    summary_path = args.output_parent / "grid_summary.json"
    best = max(
        summary,
        key=lambda x: (x.get("test_auroc") or -1.0, x.get("test_accuracy") or -1.0),
    )
    payload = {
        "base_config": str(args.base_config),
        "n_runs": len(summary),
        "best_by_test_auroc": best,
        "runs": summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

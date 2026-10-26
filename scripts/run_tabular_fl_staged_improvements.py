"""
Staged tabular FedAvg improvements toward higher test accuracy.

Writes timestamped runs under:
  data/processed/federated/tabular_cardio_experiments/staged_<UTC>/

Does not modify tabular_cardio_runs/20261025T053718Z or other existing artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl.config import load_federation_config  # noqa: E402
from fl.training.tabular_federated import run_tabular_federated_experiment  # noqa: E402
from scripts.run_tabular_fl_experiment_grid import _metric_row  # noqa: E402

DEFAULT_FULL = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1_full_train_experiment.json"
DEFAULT_SMOKE = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1_real_smoke.json"
DEFAULT_PARENT = REPO_ROOT / "data" / "processed" / "federated" / "tabular_cardio_experiments"


def main() -> int:
    parser = argparse.ArgumentParser(description="Staged tabular FL improvement runs")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-only", action="store_true", help="Small capped train pilot before full train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()

    if not args.confirm_run and not args.dry_run:
        print("Refusing: pass --confirm-run or --dry-run", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.output_parent / f"staged_{stamp}"

    pilots = [
        {
            "label": "pilot_fulltrain_r10_e3_lr5e4_strat_cw",
            "config_path": DEFAULT_FULL,
            "overrides": {"num_rounds": 10, "max_federated_train_rows": None},
        },
        {
            "label": "pilot_cap2400_r20_e5_lr5e4_baseline_partition",
            "config_path": DEFAULT_SMOKE,
            "overrides": {
                "num_rounds": 20,
                "local_training": {
                    "local_epochs": 5,
                    "batch_size": 256,
                    "learning_rate": 0.0005,
                    "weight_decay": 0.0001,
                    "class_weighted_loss": False,
                },
                "client_partition_strategy": "contiguous_record_id",
            },
        },
    ]
    full_runs = [
        {
            "label": "fulltrain_r20_e3_lr5e4_strat_cw",
            "config_path": DEFAULT_FULL,
            "overrides": {},
        },
        {
            "label": "fulltrain_r30_e2_lr5e4_strat_cw",
            "config_path": DEFAULT_FULL,
            "overrides": {"num_rounds": 30, "local_training": {"local_epochs": 2}},
        },
    ]

    plan = pilots if args.pilot_only else pilots + full_runs

    if args.dry_run:
        print(json.dumps({"output_root": str(out_root), "planned": [p["label"] for p in plan]}, indent=2))
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for seed in args.seeds:
        for spec in plan:
            label = f"{spec['label']}_s{seed}"
            run_dir = out_root / label
            if (run_dir / "report.json").is_file():
                report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
                summary.append({"label": label, "skipped": True, **_metric_row(report)})
                continue

            cfg = load_federation_config(spec["config_path"])
            cfg.seed = seed
            for key, val in spec.get("overrides", {}).items():
                if key == "local_training":
                    for lk, lv in val.items():
                        setattr(cfg.local_training, lk, lv)
                elif key.startswith("local_training."):
                    setattr(cfg.local_training, key.split(".", 1)[1], val)
                else:
                    setattr(cfg, key, val)

            print(f"Running {label} ...", flush=True)
            run_tabular_federated_experiment(cfg, output_root=run_dir, experiment_label=label)
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            row = {"label": label, "seed": seed, **_metric_row(report)}
            summary.append(row)
            print(json.dumps(row, indent=2), flush=True)

    best = max(
        (r for r in summary if r.get("test_auroc") is not None),
        key=lambda r: (r.get("test_auroc") or 0.0, r.get("test_accuracy") or 0.0),
        default=None,
    )
    out = {"output_root": str(out_root), "runs": summary, "best_by_test_auroc": best}
    with (out_root / "staged_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"best_by_test_auroc": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

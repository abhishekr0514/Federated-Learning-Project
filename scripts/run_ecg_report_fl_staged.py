"""
Staged ECG+report FL: smoke then optional full-cohort evaluation.

Does not overwrite prior experiment directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "configs" / "federations" / "ecg_report_v1_frozen_head_smoke.json"
FULL = REPO_ROOT / "configs" / "federations" / "ecg_report_v1_frozen_head_full.json"
ENC_FT = REPO_ROOT / "configs" / "federations" / "ecg_report_v1_encoder_ft_full.json"


def _run_config(config: Path, label: str) -> dict:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from fl.config import load_federation_config
    from fl.training.ecg_report_federated import run_ecg_report_federated_experiment

    out = REPO_ROOT / "data" / "processed" / "federated" / "ecg_report_experiments" / label
    cfg = load_federation_config(config)
    print(f"Running {label} -> {out}", flush=True)
    return run_ecg_report_federated_experiment(cfg, output_root=out, experiment_label=label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--include-encoder-ft", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run and not args.dry_run:
        print("Pass --confirm-run or --dry-run", file=sys.stderr)
        return 2
    if args.smoke_only and args.full_only:
        print("Choose at most one of --smoke-only and --full-only", file=sys.stderr)
        return 2
    if args.full_only:
        plan = ["full_frozen_head"]
    elif args.smoke_only:
        plan = ["smoke"]
    else:
        plan = ["smoke", "full_frozen_head"]
    if args.include_encoder_ft and "full_frozen_head" in plan:
        plan.append("full_encoder_ft")
    if args.dry_run:
        print(json.dumps({"would_run": plan}, indent=2))
        return 0
    summary = []
    if "smoke" in plan:
        summary.append({"label": "smoke_recheck", **_run_config(SMOKE, "staged_smoke_recheck")})
    if "full_frozen_head" in plan:
        summary.append({"label": "full_frozen_head", **_run_config(FULL, "full_frozen_head_s42")})
    if "full_encoder_ft" in plan:
        summary.append({"label": "encoder_ft", **_run_config(ENC_FT, "full_encoder_ft_s42")})
    out_path = REPO_ROOT / "data" / "processed" / "federated" / "ecg_report_experiments" / "staged_summary.json"
    out_path.write_text(json.dumps({"runs": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

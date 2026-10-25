"""Verify a saved tabular federated run (checkpoint + report consistency)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from fl.checkpoints import load_federated_checkpoint
from fl.errors import CheckpointError
from ml.models.tabular_encoder import TabularBinaryClassifier


def find_latest_run(root: Path) -> Path:
    runs = root / "tabular_cardio_runs"
    if not runs.is_dir():
        raise FileNotFoundError(f"No tabular_cardio_runs under {root}")
    candidates = sorted([p for p in runs.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No run directories found")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory (default: latest under data/processed/federated/tabular_cardio_runs)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(REPO_ROOT / "data" / "processed" / "federated")
    ckpt_path = run_dir / "global_federated_checkpoint.pt"
    report_path = run_dir / "report.json"

    if not ckpt_path.is_file() or not report_path.is_file():
        print(json.dumps({"ok": False, "error": "Missing checkpoint or report", "run_dir": str(run_dir)}))
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = TabularBinaryClassifier()
    from fl.parameters.schema import try_schema_from_torch_state_dict

    schema = try_schema_from_torch_state_dict(
        model.state_dict(),
        architecture_id="medshield.tabular.cardio_survey.classifier",
        architecture_version="1.0.0",
    )

    try:
        meta, arrays, names = load_federated_checkpoint(ckpt_path, expected_schema=schema)
    except CheckpointError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 3

    expected_rounds = report.get("config", {}).get("num_rounds")
    history = report.get("round_history", [])
    out = {
        "ok": True,
        "run_dir": str(run_dir),
        "checkpoint": {
            "trained": meta.trained,
            "round_number": meta.round_number,
            "architecture_id": meta.architecture_id,
            "federation_id": meta.federation_id,
            "num_tensors": len(arrays),
            "parameter_names": names,
            "schema_signature_match": meta.parameter_schema_signature == schema.signature(),
        },
        "report": {
            "rounds_in_history": len(history),
            "num_rounds_config": expected_rounds,
            "clients": report.get("config", {}).get("participation", {}).get("num_clients"),
            "test_auroc": report.get("evaluation", {}).get("test", {}).get("auroc"),
            "parameter_l2_initial": report.get("parameter_l2_initial"),
            "parameter_l2_final": report.get("parameter_l2_final"),
        },
        "consistency": {
            "rounds_match": meta.round_number == expected_rounds == len(history),
            "params_changed": report.get("parameter_l2_initial") != report.get("parameter_l2_final"),
        },
    }
    print(json.dumps(out, indent=2))
    return 0 if out["consistency"]["rounds_match"] and out["consistency"]["params_changed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

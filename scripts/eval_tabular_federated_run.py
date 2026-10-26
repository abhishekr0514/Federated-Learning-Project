"""Re-evaluate a saved tabular federated run (checkpoint + frozen preprocessor)."""

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
from fl.parameters.schema import try_schema_from_torch_state_dict
from fl.parameters.serialize import ndarrays_to_state_dict
from fl.tabular.partitions import build_tabular_federated_partitions
from fl.training.tabular_federated import _evaluate_global_model
from ml.models.tabular_encoder import TabularBinaryClassifier
from ml.tabular.preprocessing import TabularPreprocessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    report_path = run_dir / "report.json"
    ckpt_path = run_dir / "global_federated_checkpoint.pt"
    prep_path = run_dir / "preprocessor.json"
    if not all(p.is_file() for p in (report_path, ckpt_path, prep_path)):
        print(json.dumps({"ok": False, "error": "Missing report, checkpoint, or preprocessor"}))
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = report.get("config", {})
    data_path = REPO_ROOT / cfg.get("dataset_path", "data/datasets/cardio_train.csv")
    seed = int(cfg.get("seed", 42))
    num_clients = int(cfg.get("participation", {}).get("num_clients", 3))
    max_rows = cfg.get("max_federated_train_rows")
    partition_strategy = cfg.get("client_partition_strategy", "contiguous_record_id")

    _parts, val_records, test_records, _plan = build_tabular_federated_partitions(
        data_path,
        seed=seed,
        num_clients=num_clients,
        max_federated_train_rows=max_rows,
        client_partition_strategy=partition_strategy,
    )

    preprocessor = TabularPreprocessor()
    preprocessor.load(prep_path)
    model = TabularBinaryClassifier()
    schema = try_schema_from_torch_state_dict(
        model.state_dict(),
        architecture_id="medshield.tabular.cardio_survey.classifier",
        architecture_version="1.0.0",
    )
    _meta, arrays, names = load_federated_checkpoint(ckpt_path, expected_schema=schema)
    state = ndarrays_to_state_dict(arrays, names, template_state_dict=model.state_dict())
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = _evaluate_global_model(model, preprocessor, val_records, test_records, device)

    out = {
        "ok": True,
        "run_dir": str(run_dir),
        "reproduced_evaluation": metrics,
        "historical_test_auroc": report.get("evaluation", {}).get("test", {}).get("auroc"),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

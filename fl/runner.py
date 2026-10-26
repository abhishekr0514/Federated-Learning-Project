"""
MedShield-FL Flower runner (Prompt 20).

Default: validation and readiness only — does NOT start training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl.adapters.ecg import ecg_federation_readiness
from fl.adapters.gat import gat_federation_readiness
from fl.adapters.ecg_report_federation import ecg_report_federation_readiness
from fl.adapters.tabular import tabular_federation_readiness
from fl.adapters.text import nbme_federation_readiness
from fl.aggregation.fedavg import weighted_fedavg
from fl.checkpoints import FederatedCheckpointMetadata, save_federated_checkpoint
from fl.client.adapter import SyntheticInfrastructureAdapter
from fl.config import FederationConfig, default_tabular_infrastructure_config, load_federation_config
from fl.training.ecg_report_federated import run_ecg_report_federated_experiment
from fl.training.ecg_report_gat_head_federated import run_ecg_report_gat_head_federated_experiment
from fl.training.tabular_federated import run_tabular_federated_experiment
from fl.errors import FederatedLearningError, FlowerUnavailableError
from fl.registry import federation_summary
from fl.server.strategy_factory import build_fedavg_strategy, require_flwr_strategy


def cmd_validate_config(path: Path) -> int:
    cfg = load_federation_config(path)
    print(json.dumps({"ok": True, "federation_id": cfg.federation_id, "task_id": cfg.task_id}, indent=2))
    return 0


def cmd_readiness(config_path: Path | None) -> int:
    cfg = load_federation_config(config_path) if config_path else default_tabular_infrastructure_config()
    cfg.validate()

    flwr_ok = True
    flwr_version = None
    flwr_error = None
    try:
        require_flwr_strategy()
        import flwr

        flwr_version = flwr.__version__
    except FlowerUnavailableError as exc:
        flwr_ok = False
        flwr_error = str(exc)

    tabular = tabular_federation_readiness()
    report = {
        "infrastructure_level": 1,
        "task_experiment_level": 2,
        "clinical_multimodal_level": 3,
        "note": "Level 1 complete does not imply levels 2–3 are ready.",
        "federation_registry": federation_summary(),
        "active_config": cfg.to_dict(),
        "flower": {"available": flwr_ok, "version": flwr_version, "error": flwr_error},
        "tabular_adapter": tabular,
        "ecg": ecg_federation_readiness(),
        "nbme": nbme_federation_readiness(),
        "gat": gat_federation_readiness(),
        "ecg_report": ecg_report_federation_readiness(),
        "training_auto_start": False,
        "full_cohort_training": "deferred",
    }
    print(json.dumps(report, indent=2))
    return 0 if flwr_ok and tabular.get("ready") else 0  # readiness is informational; exit 0


def cmd_simulate(config_path: Path, *, confirm: bool, synthetic: bool) -> int:
    if not confirm:
        print("Refusing simulation without --confirm-run", file=sys.stderr)
        return 2
    if not synthetic:
        print("Refusing non-synthetic simulation in Prompt 20 (use --synthetic-infrastructure)", file=sys.stderr)
        return 2

    cfg = load_federation_config(config_path)
    cfg.synthetic_infrastructure_smoke = True
    cfg.num_rounds = max(cfg.num_rounds, 1)
    cfg.validate()

    spec = cfg
    adapter_a = SyntheticInfrastructureAdapter(
        federation_id=spec.federation_id,
        task_id=spec.task_id,
        architecture_id=spec.architecture_id,
        architecture_version=spec.architecture_version,
        seed=spec.seed,
        num_examples=100,
    )
    adapter_b = SyntheticInfrastructureAdapter(
        federation_id=spec.federation_id,
        task_id=spec.task_id,
        architecture_id=spec.architecture_id,
        architecture_version=spec.architecture_version,
        seed=spec.seed + 1,
        num_examples=50,
    )
    schema = adapter_a.parameter_schema()
    global_arrays, names = adapter_a.get_parameters()

    rounds = min(cfg.num_rounds, 1)
    for round_idx in range(1, rounds + 1):
        updates = []
        for adapter in (adapter_a, adapter_b):
            adapter.set_parameters(global_arrays, names)
            arrays, n, _metrics = adapter.fit({"local_epochs": 0, "loss": 0.4})
            updates.append((arrays, n))
        global_arrays = weighted_fedavg(updates, schema=schema)

    out_dir = REPO_ROOT / cfg.output_dir / "synthetic_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "global_round_1.pt"

    import flwr

    meta = FederatedCheckpointMetadata(
        federation_id=cfg.federation_id,
        task_id=cfg.task_id,
        architecture_id=cfg.architecture_id,
        architecture_version=cfg.architecture_version,
        aggregation_strategy=cfg.aggregation_strategy,
        round_number=rounds,
        parameter_schema_signature=schema.signature(),
        seed=cfg.seed,
        trained=False,
        flower_version=flwr.__version__,
        feature_schema_id=cfg.feature_schema_id,
        feature_schema_version=cfg.feature_schema_version,
        preprocessing_contract_id=cfg.preprocessing_contract_id,
        preprocessing_contract_version=cfg.preprocessing_contract_version,
    )
    save_federated_checkpoint(ckpt_path, metadata=meta, global_arrays=global_arrays, parameter_names=names)

    # Strategy factory smoke (no Ray server)
    try:
        from flwr.common import ndarrays_to_parameters

        build_fedavg_strategy(
            federation_config=cfg,
            parameter_schema=schema,
            initial_parameters=ndarrays_to_parameters(global_arrays),
        )
        strategy_ok = True
    except Exception as exc:
        strategy_ok = False
        print(f"Strategy build warning: {exc}", file=sys.stderr)

    print(
        json.dumps(
            {
                "simulation": "synthetic_infrastructure_only",
                "rounds": rounds,
                "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "trained": False,
                "strategy_initialized": strategy_ok,
            },
            indent=2,
        )
    )
    return 0


def cmd_run_ecg_report_gat_head(config_path: Path, *, confirm: bool) -> int:
    if not confirm:
        print("Refusing GAT-head federated experiment without --confirm-run", file=sys.stderr)
        return 2
    cfg = load_federation_config(config_path)
    if cfg.federated_trainable_scope != "gat_head":
        print("Config must set federated_trainable_scope=gat_head", file=sys.stderr)
        return 2
    result = run_ecg_report_gat_head_federated_experiment(cfg)
    print(json.dumps(result, indent=2))
    return 0


def cmd_run_ecg_report(config_path: Path, *, confirm: bool) -> int:
    if not confirm:
        print("Refusing ECG+report federated experiment without --confirm-run", file=sys.stderr)
        return 2
    cfg = load_federation_config(config_path)
    if not cfg.real_ecg_report_experiment:
        print("Config must set real_ecg_report_experiment=true", file=sys.stderr)
        return 2
    result = run_ecg_report_federated_experiment(cfg)
    print(json.dumps(result, indent=2))
    return 0


def cmd_run_tabular(config_path: Path, *, confirm: bool) -> int:
    if not confirm:
        print("Refusing tabular federated experiment without --confirm-run", file=sys.stderr)
        return 2
    cfg = load_federation_config(config_path)
    if not cfg.real_tabular_experiment:
        print("Config must set real_tabular_experiment=true", file=sys.stderr)
        return 2
    result = run_tabular_federated_experiment(cfg)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedShield-FL Flower infrastructure runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-config", help="Validate federation JSON config")
    p_val.add_argument("config", type=Path)

    p_ready = sub.add_parser("readiness", help="Inspect FL readiness (does not train)")
    p_ready.add_argument("--config", type=Path, default=None)

    p_sim = sub.add_parser("simulate", help="Synthetic one-round infrastructure smoke")
    p_sim.add_argument("config", type=Path)
    p_sim.add_argument("--confirm-run", action="store_true")
    p_sim.add_argument("--synthetic-infrastructure", action="store_true")

    p_tab = sub.add_parser("run-tabular-experiment", help="Small real tabular FL experiment")
    p_tab.add_argument("config", type=Path)
    p_tab.add_argument("--confirm-run", action="store_true")

    p_ecg = sub.add_parser("run-ecg-report-experiment", help="PTB-XL ECG+report FL (fusion head)")
    p_ecg.add_argument("config", type=Path)
    p_ecg.add_argument("--confirm-run", action="store_true")

    p_gat = sub.add_parser("run-ecg-report-gat-head-experiment", help="PTB-XL GAT-head FL on frozen embeddings")
    p_gat.add_argument("config", type=Path)
    p_gat.add_argument("--confirm-run", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            return cmd_validate_config(args.config)
        if args.command == "readiness":
            return cmd_readiness(args.config)
        if args.command == "simulate":
            return cmd_simulate(
                args.config,
                confirm=args.confirm_run,
                synthetic=args.synthetic_infrastructure,
            )
        if args.command == "run-tabular-experiment":
            return cmd_run_tabular(args.config, confirm=args.confirm_run)
        if args.command == "run-ecg-report-experiment":
            return cmd_run_ecg_report(args.config, confirm=args.confirm_run)
        if args.command == "run-ecg-report-gat-head-experiment":
            return cmd_run_ecg_report_gat_head(args.config, confirm=args.confirm_run)
    except FederatedLearningError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

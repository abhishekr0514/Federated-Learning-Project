"""Tabular FL experiment grid and data audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fl.config import load_federation_config
from fl.tabular.partitions import audit_partition_data, build_tabular_federated_partitions
from ml.evaluation.tabular_metrics import compute_tabular_metrics
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "datasets" / "cardio_train.csv"
REAL_CONFIG = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1_real_smoke.json"

pytestmark_data = pytest.mark.skipif(not DATA.is_file(), reason="cardio_train.csv missing")
torch = pytest.importorskip("torch")


def test_tabular_metrics_include_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_tabular_metrics(y, p, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert "brier_score" in m


@pytestmark_data
def test_stratified_round_robin_balances_clients():
    parts_cont, _, _, _ = build_tabular_federated_partitions(
        DATA,
        seed=42,
        num_clients=3,
        max_federated_train_rows=2400,
        client_partition_strategy="contiguous_record_id",
    )
    parts_strat, _, _, _ = build_tabular_federated_partitions(
        DATA,
        seed=42,
        num_clients=3,
        max_federated_train_rows=2400,
        client_partition_strategy="stratified_round_robin",
    )
    audit_cont = audit_partition_data(parts_cont, [], [])
    audit_strat = audit_partition_data(parts_strat, [], [])
    assert audit_cont["client_positive_rate_spread"] > audit_strat["client_positive_rate_spread"]


@pytestmark_data
def test_audit_partition_no_leakage():
    parts, val, test, _plan = build_tabular_federated_partitions(
        DATA, seed=42, num_clients=3, max_federated_train_rows=600
    )
    audit = audit_partition_data(parts, val, test)
    assert audit["split_leakage_detected"] is False
    assert audit["train_disjoint_across_clients"] is True
    assert audit["client_label_stats"]


@pytestmark_data
def test_grid_runner_dry_run(monkeypatch):
    import sys

    from scripts.run_tabular_fl_experiment_grid import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tabular_fl_experiment_grid.py",
            "--dry-run",
            "--limit",
            "3",
            "--base-config",
            str(REAL_CONFIG),
        ],
    )
    assert main() == 0


@pytestmark_data
def test_single_experiment_writes_audit(tmp_path):
    cfg = load_federation_config(REAL_CONFIG)
    cfg.max_federated_train_rows = 400
    cfg.num_rounds = 1
    cfg.local_training.local_epochs = 1
    cfg.participation.num_clients = 2
    cfg.participation.min_fit_clients = 2
    cfg.participation.min_evaluate_clients = 2
    cfg.participation.min_available_clients = 2

    from fl.training.tabular_federated import run_tabular_federated_experiment

    run_tabular_federated_experiment(cfg, output_root=tmp_path / "exp1", experiment_label="exp1")
    audit = json.loads((tmp_path / "exp1" / "data_audit.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "exp1" / "report.json").read_text(encoding="utf-8"))
    assert audit["split_leakage_detected"] is False
    assert "accuracy" in report["evaluation"]["test"]
    assert report["comparison_notes"]["threshold_selected_on_validation_only"] is True

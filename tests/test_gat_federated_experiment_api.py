"""GAT federated experiment read API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.services.gat_federated_experiment_service import get_gat_federated_experiment_status


def test_gat_api_no_tabular_auroc_leak(tmp_path, monkeypatch):
    runs = tmp_path / "gat_ecg_report_runs" / "run-abc"
    runs.mkdir(parents=True)
    report = {
        "experiment": "gat_ecg_report_federated",
        "architecture_id": "medshield.diagnostic_net.gat_ecg_report",
        "config": {"num_rounds": 2, "participation": {"num_clients": 2}, "local_training": {"local_epochs": 1}},
        "round_history": [{"round": 1}, {"round": 2}],
        "parameter_l2_initial": "aaa",
        "parameter_l2_final": "bbb",
        "evaluation": {"test": {"auroc": 0.812, "n_samples": 100, "class_counts": {"0": 50, "1": 50}}},
        "centralized_baseline_same_test_embeddings": {"test": {"auroc": 0.831}, "note": "same split"},
        "flower_run_id": "run-abc",
    }
    (runs / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        "backend.app.services.gat_federated_experiment_service._gat_runs_root",
        lambda: tmp_path / "gat_ecg_report_runs",
    )
    st = get_gat_federated_experiment_status()
    assert st["status"] == "AVAILABLE"
    assert st["report"]["evaluation_summary"]["test_auroc"] == 0.812
    assert "0.770" not in json.dumps(st)
    assert st["report"]["config_summary"]["architecture_id"] == "medshield.diagnostic_net.gat_ecg_report"

"""Real tabular federated experiment tests (small data; not clinical claims)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fl.adapters.tabular import TabularFederationAdapter
from fl.config import FederationConfig, load_federation_config
from fl.tabular.partitions import assign_client_partitions, build_tabular_federated_partitions
from ml.datasets.cardio_tabular import load_and_filter_cardio_cohort, stratified_cardio_split
from ml.tabular.preprocessing import TabularPreprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "datasets" / "cardio_train.csv"
REAL_CONFIG = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1_real_smoke.json"

pytestmark_data = pytest.mark.skipif(not DATA.is_file(), reason="cardio_train.csv missing")
torch = pytest.importorskip("torch")


@pytestmark_data
def test_client_partitions_disjoint():
    parts, _val, _test, plan = build_tabular_federated_partitions(
        DATA, seed=42, num_clients=3, max_federated_train_rows=600
    )
    assert len(parts) == 3
    assert sum(len(p) for p in parts) == plan.train_row_count_total
    all_ids = []
    for p in parts:
        all_ids.extend(int(r["record_id"]) for r in p)
    assert len(all_ids) == len(set(all_ids))


@pytestmark_data
def test_local_training_changes_parameters():
    accepted, _, _ = load_and_filter_cardio_cohort(DATA)
    train_ids, _, _ = stratified_cardio_split(accepted, seed=42)
    train_records = [r for r in accepted if r["record_id"] in train_ids][:400]
    parts = assign_client_partitions(train_records, num_clients=2)
    preprocessor = TabularPreprocessor()
    preprocessor.fit([r["features"] for r in train_records])

    adapter = TabularFederationAdapter(
        train_records=parts[0],
        preprocessor=preprocessor,
        client_index=0,
        num_clients=2,
        local_epochs=1,
        device_name="cpu",
    )
    before, _ = adapter.get_parameters()
    before_l2 = sum(float(np.sum(a**2)) for a in before)
    after, _n, metrics = adapter.fit({"local_epochs": 1})
    after_l2 = sum(float(np.sum(a**2)) for a in after)
    assert metrics["loss"] >= 0.0
    assert after_l2 != before_l2


@pytestmark_data
def test_federated_run_report_has_no_raw_ids(tmp_path):
    cfg = load_federation_config(REAL_CONFIG)
    cfg.max_federated_train_rows = 400
    cfg.num_rounds = 1
    cfg.local_training.local_epochs = 1
    cfg.participation.num_clients = 2
    cfg.participation.min_fit_clients = 2
    cfg.participation.min_evaluate_clients = 2
    cfg.participation.min_available_clients = 2

    from fl.training.tabular_federated import run_tabular_federated_experiment

    result = run_tabular_federated_experiment(cfg, output_root=tmp_path / "run")
    report = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    blob = json.dumps(report)
    assert "patient_id" not in blob
    assert result["trained"] is True
    meta = torch.load(tmp_path / "run" / "global_federated_checkpoint.pt", weights_only=False)
    assert meta["metadata"]["trained"] is True


def test_real_config_validates():
    if not REAL_CONFIG.is_file():
        pytest.skip("config missing")
    cfg = load_federation_config(REAL_CONFIG)
    assert cfg.real_tabular_experiment is True

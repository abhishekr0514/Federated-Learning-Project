"""GAT federated learning (embedding-head) unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.federated.gat_ecg_report_fl import (
    ARCHITECTURE_ID,
    GATFederatedClientAdapter,
    assign_client_for_patient,
    build_bundle_from_tensors,
    create_model,
    evaluate_model_on_split,
    finalize_federated_run,
    numpy_list_to_state_dict,
    run_dir_for,
    save_global_checkpoint,
    state_dict_to_numpy_list,
)
from ml.models.gat_ecg_report_fusion import load_gat_ecg_report_checkpoint
from ml.models.tabular_encoder import TabularBinaryClassifier


def _tiny_bundle(tmp_path: Path) -> Path:
    n_train, n_test = 80, 40
    train_ecg = torch.randn(n_train, 128)
    train_text = torch.randn(n_train, 128)
    train_y = torch.randint(0, 2, (n_train,))
    test_ecg = torch.randn(n_test, 128)
    test_text = torch.randn(n_test, 128)
    test_y = torch.randint(0, 2, (n_test,))
    train_pids = [f"p{i // 4}" for i in range(n_train)]
    test_pids = [f"t{i // 4}" for i in range(n_test)]
    out = tmp_path / "bundle.pt"
    build_bundle_from_tensors(
        train_ecg=train_ecg,
        train_text=train_text,
        train_y=train_y,
        train_patient_ids=train_pids,
        test_ecg=test_ecg,
        test_text=test_text,
        test_y=test_y,
        test_patient_ids=test_pids,
        meta={"target_definition": "synthetic test labels"},
        out_path=out,
    )
    return out


def test_client_partitions_disjoint_by_patient(tmp_path, monkeypatch):
    bundle_path = _tiny_bundle(tmp_path)
    monkeypatch.setenv("MEDSHIELD_GAT_FL_BUNDLE", str(bundle_path))
    from ml.federated import gat_ecg_report_fl as mod

    bundle = mod.load_bundle()
    c0 = set(bundle["clients"][0]["patient_ids"])
    c1 = set(bundle["clients"][1]["patient_ids"])
    assert not c0.intersection(c1)
    assert c0 and c1


def test_federated_client_trains_gat_not_tabular(tmp_path, monkeypatch):
    bundle_path = _tiny_bundle(tmp_path)
    monkeypatch.setenv("MEDSHIELD_GAT_FL_BUNDLE", str(bundle_path))
    from ml.federated.gat_ecg_report_fl import GATFederatedClientAdapter, load_bundle

    adapter = GATFederatedClientAdapter(0, load_bundle())
    assert not isinstance(adapter.model, TabularBinaryClassifier)
    init = state_dict_to_numpy_list(adapter.model)
    adapter.set_parameters(init)
    out, n, _ = adapter.fit({"local_epochs": 2, "batch_size": 16})
    assert n == adapter.num_local_examples()
    assert len(out) == len(init)
    changed = any(not np.allclose(a, b) for a, b in zip(out, init))
    assert changed


def test_aggregation_updates_global_parameters(tmp_path, monkeypatch):
    bundle_path = _tiny_bundle(tmp_path)
    monkeypatch.setenv("MEDSHIELD_GAT_FL_BUNDLE", str(bundle_path))
    from ml.federated.gat_ecg_report_fl import GATFederatedClientAdapter, load_bundle

    a0 = GATFederatedClientAdapter(0, load_bundle())
    a1 = GATFederatedClientAdapter(1, load_bundle())
    base = state_dict_to_numpy_list(a0.model)
    a0.set_parameters(base)
    a1.set_parameters(base)
    w0, n0, _ = a0.fit({"local_epochs": 1, "batch_size": 16})
    w1, n1, _ = a1.fit({"local_epochs": 1, "batch_size": 16})
    assert n0 + n1 > 0
    avg = [(x * n0 + y * n1) / (n0 + n1) for x, y in zip(w0, w1)]
    assert any(not np.allclose(p, q) for p, q in zip(base, avg))


def test_evaluate_held_out_and_finalize_run_id(tmp_path, monkeypatch):
    bundle_path = _tiny_bundle(tmp_path)
    monkeypatch.setenv("MEDSHIELD_GAT_FL_BUNDLE", str(bundle_path))
    from ml.federated import gat_ecg_report_fl as mod

    bundle = mod.load_bundle()
    monkeypatch.setattr(mod, "GAT_FL_RUNS_ROOT", tmp_path / "runs")
    run_id = "test-run-uuid"
    model = create_model(init_checkpoint=None)
    arrays = state_dict_to_numpy_list(model)
    save_global_checkpoint(run_id, arrays, round_number=2, parameter_hash="abc123")
    live_state = {
        "initial_parameter_hash": "init111",
        "global_parameter_hash": "final222",
        "total_rounds": 2,
        "local_epochs": 1,
        "round_history": [{"round": 1, "run_id": run_id}, {"round": 2, "run_id": run_id}],
    }
    report = finalize_federated_run(run_id, live_state=live_state)
    assert report["flower_run_id"] == run_id
    assert report["architecture_id"] == ARCHITECTURE_ID
    test_m = report["evaluation"]["test"]
    assert test_m.get("n_samples") == bundle["test"]["labels"].size(0)
    assert "auroc" in test_m
    report_path = run_dir_for(run_id)
    assert (report_path / "report.json").is_file()
    loaded = json.loads((report_path / "report.json").read_text(encoding="utf-8"))
    assert loaded["flower_run_id"] == run_id
    assert "0.770" not in json.dumps(loaded)


def test_assign_client_stable():
    assert assign_client_for_patient("patient-42") in (0, 1)
    assert assign_client_for_patient("patient-42") == assign_client_for_patient("patient-42")


@pytest.mark.skipif(
    not Path("models/gat_ecg_report/best_gat_checkpoint.pt").is_file(),
    reason="deploy GAT checkpoint missing",
)
def test_init_from_deploy_checkpoint_compatible():
    model = create_model()
    ckpt = load_gat_ecg_report_checkpoint("models/gat_ecg_report/best_gat_checkpoint.pt")
    assert model.state_dict().keys() == ckpt.state_dict().keys()

"""ECG + report federated experiment tests (fixtures only; no full PTB-XL training in CI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fl.aggregation.fedavg import weighted_fedavg
from fl.config import load_federation_config
from fl.ecg_report.partitions import audit_ecg_report_partitions, assign_patient_disjoint_clients
from fl.ecg_report.trainable import FUSION_HEAD_KEYS, apply_trainable_scope
from fl.parameters.schema import schema_from_named_arrays
from fl.registry import FEDERATION_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
PTBXL = REPO_ROOT / "data" / "raw" / "ptb-xl"
ECG_REPORT_CONFIG = REPO_ROOT / "configs" / "federations" / "ecg_report_v1_frozen_head_smoke.json"
TABULAR_CONFIG = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1_real_smoke.json"

torch = pytest.importorskip("torch")


def test_ecg_report_federation_registered():
    spec = FEDERATION_REGISTRY["medshield.fl.ecg_report.v1"]
    assert spec.blocked is False
    assert spec.architecture_id == "medshield.ecg_report_fusion.v1"


def test_ecg_report_smoke_config_loads():
    cfg = load_federation_config(ECG_REPORT_CONFIG)
    assert cfg.real_ecg_report_experiment is True
    assert cfg.real_tabular_experiment is False
    assert cfg.federated_trainable_scope == "fusion_head"


def test_tabular_config_unchanged_by_ecg_report_task():
    cfg = load_federation_config(TABULAR_CONFIG)
    assert cfg.real_tabular_experiment is True
    assert cfg.real_ecg_report_experiment is False


def test_patient_disjoint_client_partitions():
    records = []
    for i in range(12):
        records.append(
            {
                "ecg_id": 1000 + i,
                "patient_id": f"P{i // 2}",
                "label": i % 2,
                "strat_fold": 1,
                "report": "sample report",
            }
        )
    parts = assign_patient_disjoint_clients(records, num_clients=3, seed=42, strategy="stratified_round_robin")
    assert sum(len(p) for p in parts) == len(records)
    audit = audit_ecg_report_partitions(parts, val_records=[{"ecg_id": 9, "label": 0}], test_records=[])
    assert audit["split_leakage_detected"] is False


def test_ecg_encoder_ft_scope():
    from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig
    from ml.models.ecg_report_fusion import ECGReportFusionClassifier
    from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline

    pipe, _ = ClinicalTextEncodingPipeline.from_pretrained(
        config=ClinicalTextEncoderConfig(max_sequence_length=64), device=torch.device("cpu")
    )
    model = ECGReportFusionClassifier(pipe.encoder)
    trainable, frozen = apply_trainable_scope(model, "fusion_head_and_ecg_encoder")
    assert set(FUSION_HEAD_KEYS).issubset(set(trainable))
    assert any(n.startswith("ecg_encoder.") for n in trainable)
    for name, param in model.named_parameters():
        if name in trainable:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_fusion_head_only_trainable():
    from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig
    from ml.models.ecg_report_fusion import ECGReportFusionClassifier
    from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline

    pipe, _ = ClinicalTextEncodingPipeline.from_pretrained(
        config=ClinicalTextEncoderConfig(max_sequence_length=64), device=torch.device("cpu")
    )
    model = ECGReportFusionClassifier(pipe.encoder)
    trainable, frozen = apply_trainable_scope(model, "fusion_head")
    assert set(trainable) == set(FUSION_HEAD_KEYS)
    for name, param in model.named_parameters():
        if name in FUSION_HEAD_KEYS:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_weighted_fedavg_on_head_fixture():
    w1 = np.array([[0.1, 0.2]], dtype=np.float32)
    b1 = np.array([0.0], dtype=np.float32)
    w2 = np.array([[0.3, 0.4]], dtype=np.float32)
    b2 = np.array([0.2], dtype=np.float32)
    schema = schema_from_named_arrays(
        [("head.bias", b1), ("head.weight", w1)],
        architecture_id="medshield.ecg_report_fusion.v1",
        architecture_version="1.0.0",
    )
    agg = weighted_fedavg(
        [([b1, w1], 100), ([b2, w2], 300)],
        schema=schema,
    )
    expected_b = (b1 * 0.25 + b2 * 0.75).astype(np.float32)
    expected_w = (w1 * 0.25 + w2 * 0.75).astype(np.float32)
    assert np.allclose(agg[0], expected_b)
    assert np.allclose(agg[1], expected_w)


@pytest.mark.skipif(not PTBXL.is_dir(), reason="PTB-XL not installed")
def test_encoder_bundle_device_placement():
    manifest = REPO_ROOT / "models" / "ecg_report_fusion" / "manifest.json"
    ckpt = REPO_ROOT / json.loads(manifest.read_text())["checkpoint_path"]
    if not ckpt.is_file():
        pytest.skip("encoder checkpoint missing")
    from fl.adapters.ecg_report_federation import load_encoder_bundle
    from fl.ecg_report.device import assert_model_on_device, resolve_fl_device

    device = resolve_fl_device()
    model, _pipe, _norm, _m, _fp = load_encoder_bundle(REPO_ROOT, manifest, device)
    assert_model_on_device(model, device)


@pytest.mark.skipif(not PTBXL.is_dir(), reason="PTB-XL not installed")
def test_build_partitions_smoke():
    from fl.ecg_report.partitions import build_ecg_report_federated_partitions

    parts, val, test, plan = build_ecg_report_federated_partitions(
        PTBXL, seed=42, num_clients=2, max_federated_train_records=32, client_partition_strategy="stratified_round_robin"
    )
    assert plan.train_row_count_total <= 32
    assert len(val) > 0 and len(test) > 0
    audit = audit_ecg_report_partitions(parts, val, test)
    assert audit["split_leakage_detected"] is False


@pytest.mark.skipif(not PTBXL.is_dir(), reason="PTB-XL not installed")
@pytest.mark.slow
def test_ecg_report_fl_smoke_run(tmp_path):
    manifest = REPO_ROOT / "models" / "ecg_report_fusion" / "manifest.json"
    ckpt = REPO_ROOT / json.loads(manifest.read_text())["checkpoint_path"]
    if not ckpt.is_file():
        pytest.skip("ECG-report encoder checkpoint missing")
    cfg = load_federation_config(ECG_REPORT_CONFIG)
    cfg.num_rounds = 1
    cfg.max_federated_train_records = 16
    cfg.local_training.local_epochs = 1
    cfg.local_training.batch_size = 2
    cfg.participation.num_clients = 2
    from fl.training.ecg_report_federated import run_ecg_report_federated_experiment

    run_ecg_report_federated_experiment(cfg, output_root=tmp_path / "run1", experiment_label="pytest_smoke")
    report = json.loads((tmp_path / "run1" / "report.json").read_text(encoding="utf-8"))
    assert report["training_mode"]["federated_trainable_scope"] == "fusion_head"
    assert "evaluation" in report
    assert report["evaluation"]["test"]["n_samples"] > 0

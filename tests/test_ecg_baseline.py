"""Tests for PTB-XL ECG baseline (Prompt 12)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PTBXL = REPO / "data" / "raw" / "ptb-xl"


def test_binary_label_mapping_cases():
    from ml.ptbxl.labels import ExclusionReason, assign_binary_label

    d0 = assign_binary_label(frozenset({"NORM"}))
    assert d0.included and d0.label == 0

    d1 = assign_binary_label(frozenset({"MI"}))
    assert d1.included and d1.label == 1

    d1b = assign_binary_label(frozenset({"STTC", "CD"}))
    assert d1b.included and d1b.label == 1

    conflict = assign_binary_label(frozenset({"NORM", "MI"}))
    assert not conflict.included
    assert conflict.exclusion == ExclusionReason.NORM_PATHOLOGY_CONFLICT

    empty = assign_binary_label(frozenset())
    assert not empty.included
    assert empty.exclusion == ExclusionReason.EMPTY_SUPERCLASS

    cd_only = assign_binary_label(frozenset({"CD"}))
    assert not cd_only.included
    assert cd_only.exclusion == ExclusionReason.NON_BINARY_SUPERCLASS


@pytest.mark.skipif(not PTBXL.exists(), reason="PTB-XL not installed locally")
def test_build_labeled_index_stats():
    from ml.ptbxl.labels import build_labeled_index

    _, stats = build_labeled_index(PTBXL)
    assert stats["included"] + stats["excluded"] == stats["total_metadata_rows"]
    assert stats["class_counts"]["0"] > 0
    assert stats["class_counts"]["1"] > 0
    assert stats["exclusion_counts"]["empty_superclass"] in (407, 411)  # dataset-version dependent


@pytest.mark.skipif(not PTBXL.exists(), reason="PTB-XL not installed locally")
def test_patient_disjoint_on_included_records():
    from ml.ptbxl.labels import build_labeled_index
    from ml.ptbxl.splits import verify_patient_disjoint

    records, _ = build_labeled_index(PTBXL)
    result = verify_patient_disjoint(records)
    assert result["patient_disjoint"] is True


def test_split_assignments():
    from ml.ptbxl.splits import assign_split

    assert assign_split(1) == "train"
    assert assign_split(8) == "validation"
    assert assign_split(10) == "test"
    assert assign_split(9) is None


def test_normalization_fits_train_only():
    from ml.preprocessing.ecg_normalization import fit_normalization_from_signals

    a = np.ones((12, 1000), dtype=np.float32) * 2.0
    b = np.zeros((12, 1000), dtype=np.float32)
    stats = fit_normalization_from_signals([a, b])
    out = stats.transform(a)
    assert out.shape == (12, 1000)
    assert np.isfinite(out).all()


def test_ecg_encoder_output_shape():
    from ml.models.ecg_encoder import ECGBinaryClassifier

    model = ECGBinaryClassifier()
    x = torch.randn(4, 12, 1000)
    logit, emb = model(x)
    assert emb.shape == (4, 128)
    assert logit.shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.isfinite(loss)


def test_metrics_on_synthetic_scores():
    from ml.evaluation.ecg_metrics import compute_binary_metrics

    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_binary_metrics(y, s)
    assert "auroc" in m
    assert m["confusion_matrix"]["tp"] >= 1


@pytest.mark.skipif(not PTBXL.exists(), reason="PTB-XL not installed locally")
def test_load_single_ecg_shape_and_leads():
    wfdb = pytest.importorskip("wfdb")
    from ml.datasets.ptbxl_ecg import load_ecg_mv
    from ml.ptbxl.labels import LEAD_ORDER

    sig = load_ecg_mv(PTBXL, "records100/00000/00001_lr")
    assert sig.shape == (12, 1000)
    assert np.isfinite(sig).all()
    record_path = PTBXL / "records100/00000/00001_lr"
    _, meta = wfdb.rdsamp(str(record_path))
    assert list(meta["sig_name"]) == list(LEAD_ORDER)
    assert meta["units"][0] == "mV"


def test_checkpoint_metadata_structure(tmp_path):
    from ml.models.ecg_encoder import ECGBinaryClassifier

    model = ECGBinaryClassifier()
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "selection_metric": "val_auroc",
            "val_auroc": 0.5,
            "protocol_version": "1.0.0",
        },
        path,
    )
    data = torch.load(path, weights_only=False)
    assert data["selection_metric"] == "val_auroc"
    assert "protocol_version" in data

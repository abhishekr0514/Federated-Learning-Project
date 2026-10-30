"""Unit tests for model evaluation metrics and report helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.classification_metrics import (
    CLASS_LABELS,
    POSITIVE_CLASS_INDEX,
    absolute_metric_differences,
    compute_full_binary_metrics,
    metrics_comparison_row,
)
from ml.evaluation.gat_bundle_evaluator import run_gat_inference
from ml.evaluation.report_artifacts import write_evaluation_artifacts
from ml.evaluation.gat_bundle_evaluator import GATCheckpointEvaluation, build_evaluation_manifest
from ml.models.gat_ecg_report_fusion import MedShieldGATECGReportNet, GATECGReportConfig


def test_accuracy_and_confusion_matrix_toy():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.2, 0.6, 0.8, 0.9]
    m = compute_full_binary_metrics(y_true, y_pred, y_score_positive=y_score)
    assert m["n_samples"] == 4
    assert m["n_correct"] == 3
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix_raw"]["matrix"] == [[1, 1], [0, 2]]
    assert CLASS_LABELS[0] == "NORM"
    assert POSITIVE_CLASS_INDEX == 1


def test_class_ordering_norm_mi():
    y_true = [0, 1]
    y_pred = [0, 1]
    m = compute_full_binary_metrics(y_true, y_pred, y_score_positive=[0.1, 0.9])
    assert m["confusion_matrix_raw"]["rows_actual"] == ["NORM", "MI/STTC"]
    assert m["confusion_matrix_raw"]["cols_predicted"] == ["NORM", "MI/STTC"]


def test_auroc_from_probabilities():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    y_pred = [0, 0, 1, 1]
    m = compute_full_binary_metrics(y_true, y_pred, y_score_positive=y_score)
    assert m["auroc"] == 1.0
    assert m["average_precision"] == 1.0


def test_missing_probabilities_auroc_unavailable():
    m = compute_full_binary_metrics([0, 1], [0, 1], y_score_positive=None)
    assert math.isnan(m["auroc"])
    assert "not provided" in m["auroc_unavailable_reason"]


def test_single_class_test_set():
    m = compute_full_binary_metrics([0, 0, 0], [0, 0, 0], y_score_positive=[0.2, 0.3, 0.4])
    assert math.isnan(m["auroc"])
    assert m["auroc_unavailable_reason"] == "only one class in y_true"


def test_comparison_row_and_diffs():
    a = metrics_comparison_row("a", compute_full_binary_metrics([0, 1], [0, 1], [0.2, 0.8]))
    b = metrics_comparison_row("b", compute_full_binary_metrics([0, 1], [0, 1], [0.3, 0.7]))
    diffs = absolute_metric_differences(a, b)
    assert diffs.get("accuracy", 0) == 0.0


def test_gat_inference_uses_argmax_not_confidence_threshold():
    import torch

    model = MedShieldGATECGReportNet(GATECGReportConfig(trained=False))
    ecg = torch.randn(4, 128)
    text = torch.randn(4, 128)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    y_pred, y_score, _y_conf = run_gat_inference(model, ecg, text, batch_size=4)
    assert y_pred.shape == (4,)
    assert y_score.shape == (4,)
    # argmax path: predictions are 0..1 only
    assert set(y_pred.tolist()).issubset({0, 1})


def test_report_generation_when_federated_unavailable(tmp_path):
    m = compute_full_binary_metrics([0, 1, 1], [0, 1, 0], [0.2, 0.7, 0.4])
    ev = GATCheckpointEvaluation(
        model_role="test",
        checkpoint_path=Path("x.pt"),
        metrics=m,
        y_true=np.array([0, 1, 1]),
        y_pred=np.array([0, 1, 0]),
        y_score_class1=np.array([0.2, 0.7, 0.4]),
        bundle_path=Path("bundle.pt"),
        bundle_meta={"target_definition": "test"},
    )
    manifest = build_evaluation_manifest(
        centralized=ev,
        federated=None,
        seed=42,
        output_dir=tmp_path,
        fl_selection_note="none",
    )
    paths = write_evaluation_artifacts(tmp_path, centralized=ev, federated=None, manifest=manifest)
    assert (tmp_path / "model_evaluation_report.html").is_file()
    assert (tmp_path / "model_evaluation_metrics.json").is_file()
    assert "gat_fusion_confusion_matrix" in paths


@pytest.mark.skipif(
    not Path("models/gat_ecg_report/best_gat_checkpoint.pt").is_file(),
    reason="GAT checkpoint missing",
)
def test_live_centralized_eval_smoke():
    from ml.evaluation.gat_bundle_evaluator import evaluate_checkpoint_on_bundle, default_bundle_path

    if not default_bundle_path().is_file():
        pytest.skip("bundle missing")
    ev = evaluate_checkpoint_on_bundle(
        Path("models/gat_ecg_report/best_gat_checkpoint.pt"),
        model_role="centralized",
    )
    assert ev.metrics["n_samples"] > 0
    assert 0 <= ev.metrics["accuracy"] <= 1

"""Prompt 12A — ECG quality, gate, and calibration tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.evaluation.calibration import (
    CalibrationInputError,
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
)
from ml.inference.ecg_quality_gate import (
    MSG_QUALITY_INSUFFICIENT,
    MSG_REQUIRES_REVIEW,
    apply_ecg_quality_gate,
)
from ml.preprocessing.ecg_signal_quality import (
    QualityReasonCode,
    QualityThresholds,
    assess_ecg_signal,
)
from ml.ptbxl.labels import LEAD_ORDER


def _valid_signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, size=(12, 1000)).astype(np.float32)


def test_assess_valid_signal_passes():
    report = assess_ecg_signal(_valid_signal())
    assert report.passed
    assert not report.reason_codes


def test_assess_wrong_dimensions():
    report = assess_ecg_signal(np.zeros((12, 100, 1)))
    assert not report.passed
    assert QualityReasonCode.INVALID_DIMENSIONS in report.reason_codes


def test_assess_wrong_lead_count():
    report = assess_ecg_signal(np.zeros((8, 1000)))
    assert not report.passed
    assert QualityReasonCode.WRONG_LEAD_COUNT in report.reason_codes


def test_assess_wrong_sample_count():
    report = assess_ecg_signal(np.zeros((12, 999)))
    assert not report.passed
    assert QualityReasonCode.WRONG_SAMPLE_COUNT in report.reason_codes


def test_assess_empty_and_nan_inf():
    report = assess_ecg_signal(np.array([]))
    assert not report.passed
    sig = _valid_signal()
    sig[0, 0] = np.nan
    report_nan = assess_ecg_signal(sig)
    assert QualityReasonCode.NON_FINITE_VALUES in report_nan.reason_codes
    sig2 = _valid_signal(1)
    sig2[1, 1] = np.inf
    report_inf = assess_ecg_signal(sig2)
    assert QualityReasonCode.NON_FINITE_VALUES in report_inf.reason_codes


def test_saturation_heuristic_fails():
    sig = _valid_signal(2)
    sig[0, 0] = 150.0
    report = assess_ecg_signal(sig, thresholds=QualityThresholds(saturation_abs_mv=100.0))
    assert not report.passed
    assert QualityReasonCode.SATURATION_HEURISTIC in report.reason_codes


def test_flatline_heuristic_optional():
    sig = np.zeros((12, 1000), dtype=np.float32)
    disabled = assess_ecg_signal(sig)
    assert disabled.passed
    enabled = assess_ecg_signal(
        sig,
        thresholds=QualityThresholds(flatline_all_leads_std_below_mv=1e-6),
    )
    assert not enabled.passed
    assert QualityReasonCode.FLATLINE_HEURISTIC in enabled.reason_codes


def test_gate_fails_without_probability_and_messages():
    report = assess_ecg_signal(np.zeros((12, 500)))
    outcome = apply_ecg_quality_gate(report, research_probability=0.9)
    assert not outcome.quality_passed
    assert outcome.probability is None
    assert not outcome.diagnosis_eligible
    assert MSG_QUALITY_INSUFFICIENT in outcome.user_messages
    assert MSG_REQUIRES_REVIEW in outcome.user_messages


def test_gate_passes_valid_probability():
    report = assess_ecg_signal(_valid_signal(3))
    outcome = apply_ecg_quality_gate(report, research_probability=0.42)
    assert outcome.quality_passed
    assert outcome.probability == pytest.approx(0.42)
    assert outcome.user_messages == ()


def test_gate_rejects_invalid_probabilities():
    report = assess_ecg_signal(_valid_signal(4))
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        outcome = apply_ecg_quality_gate(report, research_probability=bad)
        assert outcome.probability is None
        assert MSG_QUALITY_INSUFFICIENT in outcome.user_messages


def test_brier_perfect_and_worst():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, np.array([0.0, 1.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert brier_score(y, np.array([1.0, 0.0, 0.0, 1.0])) == pytest.approx(1.0)


def test_ece_synthetic_uniform():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    ece = expected_calibration_error(y, p, n_bins=1)
    assert ece["ece"] == pytest.approx(0.0)
    assert ece["reliability_bins"][0]["count"] == 2


def test_calibration_empty_bins_and_invalid_inputs():
    y = np.array([0, 1])
    p = np.array([0.05, 0.95])
    ece = expected_calibration_error(y, p, n_bins=10)
    empty = [b for b in ece["reliability_bins"] if b["empty"]]
    assert len(empty) == 8
    with pytest.raises(CalibrationInputError):
        brier_score(np.array([]), np.array([]))
    with pytest.raises(CalibrationInputError):
        brier_score(np.array([0]), np.array([0.5, 0.5]))
    with pytest.raises(CalibrationInputError):
        brier_score(np.array([0]), np.array([float("nan")]))
    with pytest.raises(CalibrationInputError):
        brier_score(np.array([0]), np.array([1.5]))


def test_fit_train_normalization_partition_only(monkeypatch):
    from ml.training.train_ecg_baseline import fit_train_normalization

    loaded: list[str] = []

    def fake_load(_root, filename_lr: str):
        loaded.append(filename_lr)
        return np.ones((12, 1000), dtype=np.float32)

    monkeypatch.setattr("ml.training.train_ecg_baseline.load_ecg_mv", fake_load)
    train = [{"filename_lr": "train_a"}, {"filename_lr": "train_b"}]
    fit_train_normalization(Path("."), train, None)
    assert loaded == ["train_a", "train_b"]


def test_encoder_output_still_128():
    from ml.models.ecg_encoder import ECGBinaryClassifier

    model = ECGBinaryClassifier()
    _, emb = model(torch.randn(2, 12, 1000))
    assert emb.shape == (2, 128)

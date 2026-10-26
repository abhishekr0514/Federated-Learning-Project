"""GAT full-cohort training protocol tests (no full GPU training)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.gat_cohort import (
    apply_head_record_caps,
    canonical_ecg_report_partitions,
    ecg_id_set_fingerprint,
    fl_canonical_test_ecg_ids,
    verify_test_cohort_against_reference,
)
from ml.training.train_ecg_report_fusion import PTBXL_ROOT
from ml.training.train_gat_ecg_report import GATTrainingConfig, run_gat_ecg_report_training

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_GAT_REPORT = REPO_ROOT / "data/processed/gat_ecg_report/runs/20261025T182109Z/report.json"
FL_A_REPORT = REPO_ROOT / "data/processed/federated/ecg_report_experiments/full_frozen_head_s42/report.json"


@pytest.mark.skipif(not PTBXL_ROOT.is_dir(), reason="PTB-XL not present")
def test_full_train_split_has_no_cap_by_default():
    parts, _ = canonical_ecg_report_partitions(PTBXL_ROOT)
    assert len(parts["train"]) == 12898
    assert len(parts["validation"]) == 1813
    assert len(parts["test"]) == 1838


@pytest.mark.skipif(not PTBXL_ROOT.is_dir(), reason="PTB-XL not present")
def test_head_cap_matches_legacy_gat_train_count():
    parts, _ = canonical_ecg_report_partitions(PTBXL_ROOT)
    capped = apply_head_record_caps(parts, max_train=1500, max_val=None, max_test=None)
    assert len(capped["train"]) == 1500
    assert len(capped["validation"]) == 1813
    assert len(capped["test"]) == 1838


@pytest.mark.skipif(not PTBXL_ROOT.is_dir(), reason="PTB-XL not present")
def test_test_ecg_ids_match_fl_canonical():
    parts, _ = canonical_ecg_report_partitions(PTBXL_ROOT)
    ref = fl_canonical_test_ecg_ids(PTBXL_ROOT)
    check = verify_test_cohort_against_reference(parts["test"], ref)
    assert check["match"] is True
    assert check["test_count"] == 1838


def test_report_required_keys_from_smoke(tmp_path):
    if not PTBXL_ROOT.is_dir():
        pytest.skip("PTB-XL not present")
    out = tmp_path / "gat_smoke"
    result = run_gat_ecg_report_training(
        GATTrainingConfig(
            smoke=True,
            output_dir=out,
            experiment_label="pytest_smoke",
            skip_deploy=True,
            batch_size=8,
            device="cpu",
        )
    )
    report_path = out / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["completed_training"] is True
    assert report["deploy_updated"] is False
    assert report["evaluation"]["threshold_policy"] == "youden_j_on_validation_per_model"
    assert "accuracy" in report["evaluation"]["test"]
    assert (out / "split_ecg_ids.json").is_file()
    assert (out / "best_gat_checkpoint.pt").is_file()
    assert result["test_auroc"] is not None


@pytest.mark.skipif(not LEGACY_GAT_REPORT.is_file(), reason="legacy GAT report missing")
@pytest.mark.skipif(not FL_A_REPORT.is_file(), reason="FL experiment A report missing")
def test_legacy_gat_same_test_n_but_not_comparable_train():
    legacy = json.loads(LEGACY_GAT_REPORT.read_text(encoding="utf-8"))
    fl_a = json.loads(FL_A_REPORT.read_text(encoding="utf-8"))
    assert legacy["test_n"] == fl_a["unified_comparison_same_records"]["test_n"] == 1838
    assert legacy["partition_counts"]["train"] == 1500
    assert fl_a["partition_plan_summary"]["train_row_count_total"] == 12898


def test_fingerprint_stable():
    fp1 = ecg_id_set_fingerprint([3, 1, 2])
    fp2 = ecg_id_set_fingerprint([1, 2, 3])
    assert fp1 == fp2

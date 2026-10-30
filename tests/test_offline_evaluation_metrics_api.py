"""Offline evaluation loader reads real report artifacts (no fabrication)."""

import pytest

from backend.app.services.offline_evaluation_metrics_service import (
    load_offline_gat_evaluation_summary,
)


def test_load_offline_summary_when_report_exists():
    summary = load_offline_gat_evaluation_summary()
    if summary["status"] != "AVAILABLE":
        pytest.skip("reports/model_evaluation not generated in this environment")
    assert summary["models"]["gat_centralized"]["n_samples"] == 1838
    assert "roc_curve" not in summary["models"]["gat_centralized"]
    assert summary["models"]["gat_centralized"]["auroc"] > 0.9


def test_load_offline_summary_missing_dir(tmp_path):
    summary = load_offline_gat_evaluation_summary(report_dir=tmp_path / "empty")
    assert summary["status"] == "REPORT_NOT_FOUND"

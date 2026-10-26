"""Causal design review vs DoWhy pipeline execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.services.causal_analysis_service import CausalAnalysisService
from backend.app.services.causal_design_review import assess_gat_ptbxl_cohort_design
from backend.app.services.gat_causal_cohort_service import COHORT_ID, GATCohortBuildResult


def test_platform_design_not_defensible_by_default():
    review = assess_gat_ptbxl_cohort_design(metadata_aligned=True, provenance={"alignment": {"aligned": True}})
    assert review["causal_question_defensible"] is False
    assert "Cross-sectional" in review["limitations"][0]


def test_readiness_separates_pipeline_from_identifiability(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "CAUSAL_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(settings, "CAUSAL_EFFECT_ESTIMATION_ALLOWED", False)
    monkeypatch.setattr(
        "backend.app.services.causal_analysis_service._dowhy_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.app.services.gat_causal_cohort_service.GATCausalCohortService.gat_checkpoint_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.app.services.gat_causal_cohort_service.GATCausalCohortService.build_cohort",
        lambda **_: GATCohortBuildResult(
            ok=True,
            blockers=[],
            cohort_mode="DESCRIPTIVE",
            metadata_aligned=True,
            frame=pd.DataFrame({"research_proxy_label": [0, 1]}),
            provenance={"alignment": {"aligned": True}, "sample_count": 2},
            descriptive_stats={"n_rows": 2},
        ),
    )
    r = CausalAnalysisService.get_readiness()
    assert r["causal_pipeline_available"] is True
    assert r["causal_estimand_identifiable"] is False
    assert r["causal_claim_validated"] is False
    assert r["summary"] == "CAUSAL_DESIGN_REQUIRED"


def test_analyze_blocked_without_override_flag(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "CAUSAL_EFFECT_ESTIMATION_ALLOWED", False)
    with pytest.raises(RuntimeError, match="CAUSAL_EFFECT_ESTIMATION_ALLOWED"):
        CausalAnalysisService.run_cohort_analysis(
            treatment="sex",
            outcome="research_proxy_label",
            confounders=["age_years"],
        )


def test_dowhy_workflow_does_not_imply_validated_claim(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "CAUSAL_EFFECT_ESTIMATION_ALLOWED", True)
    monkeypatch.setattr(settings, "CAUSAL_ANALYSIS_MIN_SAMPLE_SIZE", 10)
    df = pd.DataFrame(
        {
            "sex": [0, 1, 0, 1] * 50,
            "age_years": [45.0, 50.0, 55.0, 60.0] * 50,
            "research_proxy_label": [0, 1, 0, 1] * 50,
        }
    )
    build = GATCohortBuildResult(
        ok=True,
        blockers=[],
        cohort_mode="DESCRIPTIVE",
        metadata_aligned=True,
        frame=df,
        provenance={"cohort_version": "1.0.0", "alignment": {"aligned": True}},
        descriptive_stats={"n_rows": len(df)},
    )

    review = assess_gat_ptbxl_cohort_design(metadata_aligned=True, provenance={"alignment": {"aligned": True}})
    monkeypatch.setattr(
        CausalAnalysisService,
        "get_readiness",
        lambda: {
            "summary": "CAUSAL_DESIGN_REQUIRED",
            "reason": "test",
            "causal_pipeline_available": True,
            "metadata_aligned": True,
            "causal_estimand_identifiable": False,
            "causal_claim_validated": False,
            "causal_design_review": review,
        },
    )
    monkeypatch.setattr(CausalAnalysisService, "_load_cohort_frame", lambda: (build, df))

    mock_model = MagicMock()
    mock_model.identify_effect.return_value = MagicMock()
    mock_model.estimate_effect.return_value = MagicMock(value=0.05)
    mock_model.refute_estimate.return_value = MagicMock(new_effect=0.04, refutation_result="ok")

    with patch("dowhy.CausalModel", return_value=mock_model):
        result = CausalAnalysisService.run_cohort_analysis(
            treatment="sex",
            outcome="research_proxy_label",
            confounders=["age_years"],
        )
    assert result["status"] == "ESTIMATED"
    assert result["causal_claim_validated"] is False
    assert result["causal_estimand_identifiable"] is False
    assert result["cohort_id"] == COHORT_ID

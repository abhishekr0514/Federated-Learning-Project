"""GAT-grounded DoWhy causal analysis service tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.services.causal_analysis_service import CausalAnalysisService
from backend.app.services.gat_causal_cohort_service import COHORT_ID, GATCohortBuildResult


class TestCausalReadiness:
    def test_readiness_missing_dowhy(self, monkeypatch):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "CAUSAL_ANALYSIS_ENABLED", True)
        monkeypatch.setattr(
            "backend.app.services.causal_analysis_service._dowhy_available",
            lambda: False,
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
                metadata_aligned=False,
                frame=pd.DataFrame({"research_proxy_label": [0, 1]}),
                provenance={"sample_count": 2},
                descriptive_stats={"n_rows": 2},
            ),
        )
        r = CausalAnalysisService.get_readiness()
        assert r["summary"] == "MISSING_DEPENDENCY"

    def test_list_cohorts_uses_gat_cohort_id(self, monkeypatch):
        monkeypatch.setattr(
            "backend.app.services.gat_causal_cohort_service.GATCausalCohortService.list_cohort_metadata",
            lambda: {"cohort_id": COHORT_ID, "cohort_mode": "DESCRIPTIVE", "variables": []},
        )
        monkeypatch.setattr(CausalAnalysisService, "get_readiness", lambda: {"summary": "CAUSAL_DESIGN_REQUIRED"})
        r = CausalAnalysisService.list_cohorts()
        assert r["cohorts"][0]["cohort_id"] == COHORT_ID


class TestCausalAnalysisValidation:
    def test_gat_variable_rejected_for_causal(self, monkeypatch):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "CAUSAL_EFFECT_ESTIMATION_ALLOWED", True)
        monkeypatch.setattr(
            CausalAnalysisService,
            "get_readiness",
            lambda: {
                "summary": "CAUSAL_DESIGN_REQUIRED",
                "reason": "ok",
                "causal_pipeline_available": True,
                "metadata_aligned": True,
                "causal_design_review": {},
            },
        )
        with pytest.raises(ValueError, match="Model-derived"):
            CausalAnalysisService.run_cohort_analysis(
                treatment="gat_prob_class_1",
                outcome="research_proxy_label",
                confounders=["age_years"],
            )

    def test_descriptive_cohort_blocks_estimation(self, monkeypatch):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "CAUSAL_EFFECT_ESTIMATION_ALLOWED", False)
        with pytest.raises(RuntimeError, match="CAUSAL_EFFECT_ESTIMATION_ALLOWED"):
            CausalAnalysisService.run_cohort_analysis(
                treatment="sex",
                outcome="research_proxy_label",
                confounders=["age_years"],
            )


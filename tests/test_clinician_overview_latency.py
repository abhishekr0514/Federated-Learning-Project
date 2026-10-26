"""Clinician overview must not block on full GAT cohort build."""

from __future__ import annotations

import time
from unittest.mock import patch

from backend.app.services.capability_registry import get_capability_details


def test_capability_details_completes_without_full_cohort_build():
    with patch(
        "backend.app.services.gat_causal_cohort_service.GATCausalCohortService.build_cohort",
    ) as build:
        start = time.perf_counter()
        details = get_capability_details()
        elapsed = time.perf_counter() - start
        build.assert_not_called()
        assert "dowhy_causal" in details
        assert elapsed < 15.0, f"capability_details took {elapsed:.1f}s (expected fast probe path)"

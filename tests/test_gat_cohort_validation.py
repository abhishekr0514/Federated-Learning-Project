"""GAT cohort alignment and descriptive statistics."""

from __future__ import annotations

import pandas as pd

from backend.app.services.gat_causal_cohort_service import _compute_descriptive_stats


def test_descriptive_stats_reports_class_balance():
    df = pd.DataFrame(
        {
            "research_proxy_label": [0, 0, 1, 1],
            "gat_predicted_class": [0, 1, 1, 1],
            "sex": [0, 1, None, 1],
            "age_years": [50.0, 60.0, 55.0, None],
        }
    )
    stats = _compute_descriptive_stats(df, label_mismatch_count=0)
    assert stats["n_rows"] == 4
    assert stats["class_balance_proxy_label"]["0"] == 2
    assert stats["missing_sex"] == 1
    assert stats["gat_accuracy_vs_proxy_label"] == 0.75

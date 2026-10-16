"""
Prompt 18 — Unit & regression tests for tabular feature schema, gender mapping, and stratified protocol.
Runs cleanly without deep-learning dependencies (requires only standard library and numpy).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from ml.datasets.cardio_tabular import (
    load_and_filter_cardio_cohort,
    row_to_features_and_target,
    stratified_cardio_split,
)
from ml.evaluation.tabular_metrics import (
    compute_tabular_metrics,
    find_best_youden_threshold,
)
from ml.tabular.errors import TabularRowValidationError, TabularSchemaError
from ml.tabular.feature_schema import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    N_TABULAR_FEATURES,
    FeatureKind,
    extract_feature_vector_from_row,
    validate_feature_name_list,
)
from ml.tabular.preprocessing import TabularPreprocessor

REPO = Path(__file__).resolve().parents[1]
CARDIO_CSV = REPO / "data" / "datasets" / "cardio_train.csv"


def _make_row(**overrides) -> dict[str, str]:
    base = {
        "id": "100",
        "age": str(int(50 * 365.25)),
        "gender": "1",
        "height": "165",
        "weight": "68",
        "ap_hi": "120",
        "ap_lo": "80",
        "cholesterol": "1",
        "gluc": "1",
        "smoke": "0",
        "alco": "0",
        "active": "1",
        "cardio": "0",
    }
    base.update(overrides)
    return base


def test_gender_female_spec_binary():
    spec = next(s for s in FEATURE_SPECS if s.name == "gender_female")
    assert spec.kind == FeatureKind.BINARY
    assert spec.ordinal_allowed == frozenset({0, 1}), (
        "gender_female allowed derived values must be {0, 1}"
    )


def test_female_record_mapping():
    row_f = _make_row(gender="1")
    feats = extract_feature_vector_from_row(row_f)
    assert feats["gender_female"] == 1.0


def test_male_record_mapping():
    row_m = _make_row(gender="2")
    feats = extract_feature_vector_from_row(row_m)
    assert feats["gender_female"] == 0.0


@pytest.mark.parametrize("invalid_gender", ["0", "3", "-1", "99", "1.5", "male", "female", ""])
def test_invalid_raw_gender_rejected(invalid_gender):
    row_bad = _make_row(gender=invalid_gender)
    with pytest.raises(TabularRowValidationError):
        extract_feature_vector_from_row(row_bad)


def test_representative_records_through_preprocessor(tmp_path: Path):
    row_female = _make_row(id="1", gender="1", ap_hi="120", ap_lo="80")
    row_male = _make_row(id="2", gender="2", ap_hi="130", ap_lo="85")

    feats_f = extract_feature_vector_from_row(row_female)
    feats_m = extract_feature_vector_from_row(row_male)

    prep = TabularPreprocessor()
    prep.fit([feats_f, feats_m])

    tf = prep.transform(feats_f)
    tm = prep.transform(feats_m)

    idx = FEATURE_NAMES.index("gender_female")
    # Binary features are unscaled pass-through in TabularPreprocessor
    assert tf[idx] == 1.0
    assert tm[idx] == 0.0
    assert np.isfinite(tf).all()
    assert np.isfinite(tm).all()


@pytest.mark.skipif(not CARDIO_CSV.exists(), reason="cardio_train.csv not available locally")
def test_real_cohort_male_female_counts_and_exclusions():
    accepted, rejected_bp, total = load_and_filter_cardio_cohort(CARDIO_CSV)
    assert total == 70000
    assert len(rejected_bp) == 1234, "Expected exactly 1234 records with ap_lo > ap_hi"
    assert len(accepted) == 68766

    males = sum(1 for r in accepted if r["features"]["gender_female"] == 0.0)
    females = sum(1 for r in accepted if r["features"]["gender_female"] == 1.0)

    assert males == 23973, "All valid male records must be preserved without schema rejection"
    assert females == 44793
    assert males + females == 68766


@pytest.mark.skipif(not CARDIO_CSV.exists(), reason="cardio_train.csv not available locally")
def test_stratified_split_disjointness_and_ratios():
    accepted, rejected_bp, _ = load_and_filter_cardio_cohort(CARDIO_CSV)
    train_ids, val_ids, test_ids = stratified_cardio_split(accepted, seed=42)

    assert len(train_ids) == 48136
    assert len(val_ids) == 10314
    assert len(test_ids) == 10316
    assert len(train_ids) + len(val_ids) + len(test_ids) == 68766

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Excluded BP rows must not appear in any partition
    rejected_set = set(rejected_bp)
    assert train_ids.isdisjoint(rejected_set)
    assert val_ids.isdisjoint(rejected_set)
    assert test_ids.isdisjoint(rejected_set)

    # Reproducibility check with same seed
    t2, v2, te2 = stratified_cardio_split(accepted, seed=42)
    assert train_ids == t2
    assert val_ids == v2
    assert test_ids == te2

    # Verify class balance across partitions
    train_targets = [r["target"] for r in accepted if r["record_id"] in train_ids]
    val_targets = [r["target"] for r in accepted if r["record_id"] in val_ids]
    test_targets = [r["target"] for r in accepted if r["record_id"] in test_ids]

    train_rate = np.mean(train_targets)
    val_rate = np.mean(val_targets)
    test_rate = np.mean(test_targets)

    assert abs(train_rate - val_rate) < 0.001
    assert abs(train_rate - test_rate) < 0.001


def test_youden_threshold_selection_logic():
    # Synthetic validation set where threshold 0.4 perfectly separates
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.45, 0.8, 0.9])

    best_thresh, best_j, grid = find_best_youden_threshold(y_true, y_prob)
    assert 0.35 < best_thresh <= 0.45
    assert best_j == 1.0

    # Test metrics with chosen threshold
    metrics = compute_tabular_metrics(y_true, y_prob, threshold=best_thresh)
    assert metrics["auroc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert 0.0 <= metrics["ece"] <= 1.0


def test_youden_threshold_single_class_fallback():
    # If validation set only has single class (e.g. all 0), fallback to 0.5
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    best_thresh, best_j, _ = find_best_youden_threshold(y_true, y_prob, fallback_threshold=0.5)
    assert best_thresh == 0.5

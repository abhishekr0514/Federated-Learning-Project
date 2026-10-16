"""Prompt 14 — Neural Tabular Encoder, schema, and preprocessing (synthetic data only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from ml.datasets.cardio_tabular import row_to_features_and_target
from ml.models.tabular_encoder import (
    DROPOUT_P,
    EMBED_DIM,
    NeuralTabularEncoder,
    TabularBinaryClassifier,
)
from ml.tabular.errors import TabularRowValidationError, TabularSchemaError, TabularTransformError
from ml.tabular.feature_schema import (
    EXCLUDED_FROM_FEATURES,
    FEATURE_NAMES,
    N_TABULAR_FEATURES,
    SCHEMA_VERSION,
    extract_feature_vector_from_row,
    validate_feature_name_list,
)
from ml.tabular.preprocessing import OutlierPolicy, TabularPreprocessor


def _synthetic_row(**overrides) -> dict[str, str]:
    base = {
        "id": "1",
        "age": str(int(45 * 365.25)),
        "gender": "1",
        "height": "170",
        "weight": "70",
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


def _synthetic_features(**overrides) -> dict[str, float]:
    feats, _, _ = row_to_features_and_target(_synthetic_row(**overrides))
    feats.update({k: float(v) for k, v in overrides.items() if k in FEATURE_NAMES})
    return feats


def test_encoder_output_shape_and_layers():
    torch.manual_seed(0)
    model = NeuralTabularEncoder()
    x = torch.randn(4, N_TABULAR_FEATURES)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, EMBED_DIM)
    assert isinstance(model.fc1, nn.Linear) and model.fc1.in_features == 10 and model.fc1.out_features == 64
    assert isinstance(model.bn, nn.BatchNorm1d) and model.bn.num_features == 64
    assert model.dropout.p == DROPOUT_P == pytest.approx(0.2)
    assert isinstance(model.fc2, nn.Linear) and model.fc2.in_features == 64 and model.fc2.out_features == 64


def test_encoder_rejects_bad_input():
    model = NeuralTabularEncoder()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 9))
    with pytest.raises(ValueError, match="non-finite"):
        model(torch.tensor([[float("nan")] * N_TABULAR_FEATURES]))


def test_batchnorm_dropout_modes():
    torch.manual_seed(1)
    model = NeuralTabularEncoder()
    x = torch.randn(8, N_TABULAR_FEATURES)
    model.train()
    y_train = model(x)
    model.eval()
    with torch.no_grad():
        y_eval = model(x)
    assert y_train.shape == y_eval.shape == (8, EMBED_DIM)


def test_schema_order_and_version():
    validate_feature_name_list(FEATURE_NAMES)
    with pytest.raises(TabularSchemaError, match="order"):
        validate_feature_name_list(list(reversed(FEATURE_NAMES)))
    with pytest.raises(TabularSchemaError, match="version"):
        validate_feature_name_list(FEATURE_NAMES, schema_version="0.0.1")


def test_schema_duplicate_and_extra_names():
    bad = list(FEATURE_NAMES)
    bad[0] = bad[1]
    with pytest.raises(TabularSchemaError, match="Duplicate"):
        validate_feature_name_list(bad)
    with pytest.raises(TabularSchemaError, match="Expected 10"):
        validate_feature_name_list(FEATURE_NAMES[:5])


def test_id_and_target_not_in_features():
    assert "id" in EXCLUDED_FROM_FEATURES
    assert "cardio" in EXCLUDED_FROM_FEATURES
    feats, target, rid = row_to_features_and_target(_synthetic_row())
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert "id" not in feats and "cardio" not in feats
    assert target in (0, 1) and rid == 1


def test_invalid_bp_rejected():
    with pytest.raises(TabularRowValidationError, match="ap_lo"):
        extract_feature_vector_from_row({k: _synthetic_row(ap_hi="90", ap_lo="120")[k] for k in _synthetic_row()})


def test_training_only_fit_and_frozen_val_transform(tmp_path: Path):
    train_rows = [_synthetic_features(ap_hi="110"), _synthetic_features(ap_hi="130")]
    val_row = _synthetic_features(ap_hi="125")
    prep = TabularPreprocessor()
    prep.fit(train_rows)
    state_before = prep.state.to_json()
    prep.transform(val_row)
    assert prep.state.to_json() == state_before
    prep.save(tmp_path / "prep.json")
    prep2 = TabularPreprocessor()
    prep2.load(tmp_path / "prep.json")
    np.testing.assert_allclose(prep.transform(val_row), prep2.transform(val_row))


def test_transform_requires_fit():
    prep = TabularPreprocessor()
    with pytest.raises(TabularSchemaError, match="fit"):
        prep.transform(_synthetic_features())


def test_non_finite_transform_rejected():
    prep = TabularPreprocessor()
    row = _synthetic_features()
    prep.fit([row])
    bad = dict(row)
    bad["bmi"] = float("inf")
    with pytest.raises(TabularTransformError):
        prep.transform(bad)


def test_binary_features_not_rescaled():
    prep = TabularPreprocessor()
    rows = [_synthetic_features(smoke="1"), _synthetic_features(smoke="0")]
    prep.fit(rows)
    out = prep.transform(_synthetic_features(smoke="1"))
    idx = FEATURE_NAMES.index("smoke")
    assert out[idx] == 1.0


def test_outlier_policy_documented_no_silent_delete():
    policy = OutlierPolicy(reject_invalid_bp=True, clip_to_train_quantiles=False)
    assert policy.reject_invalid_bp is True
    with pytest.raises(TabularRowValidationError):
        row_to_features_and_target(_synthetic_row(ap_hi="100", ap_lo="110"))


def test_reordered_feature_dict_rejected():
    feats = _synthetic_features()
    ordered = {name: feats[name] for name in reversed(FEATURE_NAMES)}
    prep = TabularPreprocessor()
    prep.fit([feats])
    with pytest.raises(TabularSchemaError, match="order"):
        validate_feature_name_list(list(ordered.keys()))


def test_encoder_deterministic_eval():
    torch.manual_seed(42)
    model = NeuralTabularEncoder()
    model.eval()
    x = torch.randn(2, N_TABULAR_FEATURES)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)


def test_missing_and_unexpected_features():
    feats = _synthetic_features()
    del feats["smoke"]
    with pytest.raises(TabularSchemaError, match="Missing"):
        prep = TabularPreprocessor()
        prep.fit([_synthetic_features()])
        prep.transform(feats)
    with pytest.raises(TabularRowValidationError):
        extract_feature_vector_from_row({k: _synthetic_row(gender="3")[k] for k in _synthetic_row()})


def test_transform_cannot_update_stats():
    prep = TabularPreprocessor()
    prep.fit([_synthetic_features()])
    with pytest.raises(TabularSchemaError, match="update"):
        prep.transform(_synthetic_features(), allow_update_stats=True)


def test_smoke_forward_software_only():
    """Software smoke test — not predictive performance evidence."""
    torch.manual_seed(0)
    prep = TabularPreprocessor()
    prep.fit([_synthetic_features()])
    x = torch.from_numpy(prep.transform(_synthetic_features()).reshape(1, -1))
    model = NeuralTabularEncoder()
    model.eval()
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (1, 64)


def test_tabular_binary_classifier_shapes():
    torch.manual_seed(42)
    classifier = TabularBinaryClassifier()
    classifier.eval()

    # Batch of 4
    x4 = torch.randn(4, N_TABULAR_FEATURES)
    logits4, emb4 = classifier(x4)
    assert logits4.shape == (4,)
    assert emb4.shape == (4, EMBED_DIM)
    assert torch.isfinite(logits4).all()
    assert torch.isfinite(emb4).all()

    # Batch of 1 (must remain 1D shape (1,), not scalar ())
    x1 = torch.randn(1, N_TABULAR_FEATURES)
    logits1, emb1 = classifier(x1)
    assert logits1.shape == (1,)
    assert emb1.shape == (1, EMBED_DIM)


def test_tabular_binary_classifier_gradients():
    torch.manual_seed(42)
    classifier = TabularBinaryClassifier()
    classifier.train()

    x = torch.randn(8, N_TABULAR_FEATURES)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    logits, emb = classifier(x)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    loss.backward()

    # Verify gradients reach classification head and encoder layers
    assert classifier.classifier.weight.grad is not None
    assert torch.isfinite(classifier.classifier.weight.grad).all()
    assert classifier.encoder.fc1.weight.grad is not None
    assert torch.isfinite(classifier.encoder.fc1.weight.grad).all()
    assert classifier.encoder.fc2.weight.grad is not None
    assert torch.isfinite(classifier.encoder.fc2.weight.grad).all()


def test_tabular_binary_classifier_rejects_bad_input():
    classifier = TabularBinaryClassifier()
    with pytest.raises(ValueError, match="shape"):
        classifier(torch.randn(4, 9))
    with pytest.raises(ValueError, match="non-finite"):
        classifier(torch.tensor([[float("nan")] * N_TABULAR_FEATURES]))


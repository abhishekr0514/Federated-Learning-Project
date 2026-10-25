"""
Flower federated-learning infrastructure tests (Prompt 20).

Synthetic arrays only — no clinical accuracy claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fl.aggregation.fedavg import weighted_fedavg
from fl.checkpoints import (
    FederatedCheckpointMetadata,
    load_federated_checkpoint,
    save_federated_checkpoint,
)
from fl.client.adapter import SyntheticInfrastructureAdapter
from fl.config import FederationConfig, load_federation_config
from fl.errors import FederationBlockedError, FederationCompatibilityError, ParameterValidationError
from fl.parameters.schema import schema_from_named_arrays
from fl.parameters.serialize import state_dict_to_ndarrays
from fl.parameters.validate import validate_ndarray_update, validate_sample_count
from fl.privacy import FORBIDDEN_MESSAGE_KEYS, PRIVACY_LIMITATIONS, assert_safe_metadata_dict
from fl.registry import GAT_MULTIMODAL_V1, resolve_federation

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "federations" / "tabular_cardio_v1.example.json"

pytestmark_infra = pytest.mark.parametrize("infrastructure_only", [True])


def test_valid_federation_config_example():
    cfg = load_federation_config(EXAMPLE_CONFIG)
    assert cfg.federation_id == "medshield.fl.tabular_cardio.v1"
    assert cfg.auto_start_training is False


def test_reject_incompatible_task_id():
    raw = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    raw["task_id"] = "wrong.task"
    cfg = FederationConfig.from_dict(raw)
    with pytest.raises(FederationCompatibilityError):
        cfg.validate()


def test_gat_federation_blocked():
    with pytest.raises(FederationBlockedError):
        GAT_MULTIMODAL_V1.assert_eligible()


def test_parameter_roundtrip_tabular_schema():
    torch = pytest.importorskip("torch")
    from ml.models.tabular_encoder import TabularBinaryClassifier

    model = TabularBinaryClassifier()
    arrays, names = state_dict_to_ndarrays(model.state_dict())
    from fl.parameters.schema import try_schema_from_torch_state_dict

    schema = try_schema_from_torch_state_dict(
        model.state_dict(),
        architecture_id="medshield.tabular.cardio_survey.classifier",
        architecture_version="1.0.0",
    )
    assert len(arrays) == schema.num_tensors
    validate_ndarray_update(
        arrays,
        schema=schema,
        architecture_id=schema.architecture_id,
        architecture_version=schema.architecture_version,
    )
    assert names == schema.expected_names()


def test_reject_wrong_parameter_count():
    schema = schema_from_named_arrays(
        [("a", np.zeros((2, 2), dtype=np.float32))],
        architecture_id="test.arch",
        architecture_version="1.0.0",
    )
    with pytest.raises(ParameterValidationError):
        validate_ndarray_update(
            [np.zeros((3, 3))],
            schema=schema,
            architecture_id="test.arch",
            architecture_version="1.0.0",
        )


def test_reject_nan_parameters():
    schema = schema_from_named_arrays(
        [("a", np.zeros((2,), dtype=np.float32))],
        architecture_id="test.arch",
        architecture_version="1.0.0",
    )
    bad = np.array([np.nan, 1.0], dtype=np.float32)
    with pytest.raises(ParameterValidationError):
        validate_ndarray_update(
            [bad],
            schema=schema,
            architecture_id="test.arch",
            architecture_version="1.0.0",
        )


def test_weighted_fedavg_synthetic():
    schema = schema_from_named_arrays(
        [("w", np.zeros((2,), dtype=np.float32))],
        architecture_id="infra.test",
        architecture_version="1.0.0",
    )
    u1 = ([np.array([1.0, 0.0], dtype=np.float32)], 100)
    u2 = ([np.array([0.0, 1.0], dtype=np.float32)], 100)
    out = weighted_fedavg([u1, u2], schema=schema)
    np.testing.assert_allclose(out[0], [0.5, 0.5])


def test_reject_zero_sample_count():
    with pytest.raises(ParameterValidationError):
        validate_sample_count(0)


def test_partial_failure_policy_strict():
    cfg = load_federation_config(EXAMPLE_CONFIG)
    assert cfg.participation.accept_failures is False


def test_checkpoint_metadata_and_corruption(tmp_path):
    schema = schema_from_named_arrays(
        [("w", np.zeros((2,), dtype=np.float32))],
        architecture_id="infra.test",
        architecture_version="1.0.0",
    )
    meta = FederatedCheckpointMetadata(
        federation_id="medshield.fl.tabular_cardio.v1",
        task_id="medshield.tabular.cardio_survey.baseline",
        architecture_id="infra.test",
        architecture_version="1.0.0",
        aggregation_strategy="fedavg",
        round_number=0,
        parameter_schema_signature=schema.signature(),
        seed=42,
        trained=False,
    )
    path = tmp_path / "ck.pt"
    save_federated_checkpoint(
        path, metadata=meta, global_arrays=[np.zeros(2, dtype=np.float32)], parameter_names=["w"]
    )
    loaded_meta, arrays, names = load_federated_checkpoint(path, expected_schema=schema)
    assert loaded_meta.trained is False
    assert names == ["w"]
    path.write_bytes(b"not-a-torch-file")
    from fl.errors import CheckpointError

    with pytest.raises(CheckpointError):
        load_federated_checkpoint(path)


def test_metadata_forbids_patient_identifiers():
    with pytest.raises(ValueError):
        assert_safe_metadata_dict({"patient_id": "x"}, context="test")


def test_federation_config_has_no_auto_start():
    cfg = load_federation_config(EXAMPLE_CONFIG)
    assert cfg.auto_start_training is False
    cfg.auto_start_training = True
    with pytest.raises(Exception):
        cfg.validate()


def test_synthetic_adapter_properties():
    spec = resolve_federation("medshield.fl.tabular_cardio.v1")
    adapter = SyntheticInfrastructureAdapter(
        federation_id=spec.federation_id,
        task_id=spec.task_id,
        architecture_id=spec.architecture_id,
        architecture_version=spec.architecture_version,
        seed=1,
        num_examples=10,
    )
    arrays, _ = adapter.get_parameters()
    assert len(arrays) == 2


def test_privacy_limitations_documented():
    assert "NOT differential privacy" in PRIVACY_LIMITATIONS
    assert "patient_id" in FORBIDDEN_MESSAGE_KEYS


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("flwr"),
    reason="flwr not installed",
)
def test_flower_api_surface():
    import flwr
    from flwr.client import NumPyClient
    from flwr.server.strategy import FedAvg

    assert flwr.__version__
    assert NumPyClient is not None
    assert FedAvg is not None


def test_runner_readiness_subprocess():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "fl.runner", "readiness", "--config", str(EXAMPLE_CONFIG)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["training_auto_start"] is False
    assert data["gat"]["blocked"] is True

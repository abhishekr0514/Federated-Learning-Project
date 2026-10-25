"""Federated global model checkpoint metadata (no raw client data)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fl.errors import CheckpointError
from fl.parameters.schema import ParameterSchema
from fl.privacy import assert_safe_metadata_dict


@dataclass
class FederatedCheckpointMetadata:
    federation_id: str
    task_id: str
    architecture_id: str
    architecture_version: str
    aggregation_strategy: str
    round_number: int
    parameter_schema_signature: str
    seed: int
    trained: bool
    flower_version: str | None = None
    pytorch_version: str | None = None
    feature_schema_id: str | None = None
    feature_schema_version: str | None = None
    preprocessing_contract_id: str | None = None
    preprocessing_contract_version: str | None = None
    aggregate_metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.round_number < 0:
            raise CheckpointError("round_number must be >= 0")
        assert_safe_metadata_dict(self.aggregate_metrics, context="aggregate_metrics")
        assert_safe_metadata_dict(asdict(self), context="checkpoint_metadata")


def save_federated_checkpoint(
    path: Path | str,
    *,
    metadata: FederatedCheckpointMetadata,
    global_arrays: list[np.ndarray],
    parameter_names: list[str],
) -> None:
    metadata.validate()
    if metadata.trained and metadata.round_number < 1:
        raise CheckpointError("trained=true requires round_number >= 1")

    payload = {
        "metadata": asdict(metadata),
        "parameter_names": parameter_names,
        "global_state_ndarrays": global_arrays,
    }
    assert_safe_metadata_dict(payload["metadata"], context="checkpoint_payload")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_federated_checkpoint(
    path: Path | str,
    *,
    expected_schema: ParameterSchema | None = None,
) -> tuple[FederatedCheckpointMetadata, list[np.ndarray], list[str]]:
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"Missing checkpoint: {p}")
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"Corrupted checkpoint file: {p}") from exc

    if "metadata" not in payload or "global_state_ndarrays" not in payload:
        raise CheckpointError("Checkpoint missing required keys")

    meta = FederatedCheckpointMetadata(**payload["metadata"])
    meta.validate()
    names = list(payload.get("parameter_names", []))
    arrays = payload["global_state_ndarrays"]

    if expected_schema is not None:
        if meta.parameter_schema_signature != expected_schema.signature():
            raise CheckpointError("parameter_schema_signature mismatch")
        if meta.architecture_id != expected_schema.architecture_id:
            raise CheckpointError("architecture_id mismatch in checkpoint")

    if len(arrays) != len(names):
        raise CheckpointError("parameter_names length mismatch")

    return meta, arrays, names

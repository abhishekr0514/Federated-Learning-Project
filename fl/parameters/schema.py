"""Stable parameter ordering for Flower ndarray exchange."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ParameterSchema:
    architecture_id: str
    architecture_version: str
    tensors: tuple[ParameterTensorSpec, ...]

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    def signature(self) -> str:
        payload = {
            "architecture_id": self.architecture_id,
            "architecture_version": self.architecture_version,
            "tensors": [
                {"name": t.name, "shape": list(t.shape), "dtype": t.dtype} for t in self.tensors
            ],
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def expected_shapes(self) -> list[tuple[int, ...]]:
        return [t.shape for t in self.tensors]

    def expected_names(self) -> list[str]:
        return [t.name for t in self.tensors]


def schema_from_named_arrays(
    named_arrays: Sequence[tuple[str, np.ndarray]],
    *,
    architecture_id: str,
    architecture_version: str,
) -> ParameterSchema:
    tensors: list[ParameterTensorSpec] = []
    for name, arr in named_arrays:
        tensors.append(
            ParameterTensorSpec(name=name, shape=tuple(int(x) for x in arr.shape), dtype=str(arr.dtype))
        )
    return ParameterSchema(
        architecture_id=architecture_id,
        architecture_version=architecture_version,
        tensors=tuple(tensors),
    )


def try_schema_from_torch_state_dict(
    state_dict: dict[str, Any],
    *,
    architecture_id: str,
    architecture_version: str,
) -> ParameterSchema:
    import torch

    named: list[tuple[str, np.ndarray]] = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {key}")
        named.append((key, tensor.detach().cpu().numpy()))
    return schema_from_named_arrays(
        named, architecture_id=architecture_id, architecture_version=architecture_version
    )

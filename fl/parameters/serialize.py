"""Convert between named torch state and Flower-ordered ndarray lists."""

from __future__ import annotations

from typing import Any

import numpy as np

from fl.parameters.schema import ParameterSchema


def state_dict_to_ndarrays(state_dict: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    names = sorted(state_dict.keys())
    arrays: list[np.ndarray] = []
    for name in names:
        tensor = state_dict[name]
        arrays.append(tensor.detach().cpu().numpy().copy())
    return arrays, names


def ndarrays_to_state_dict(
    arrays: list[np.ndarray],
    names: list[str],
    *,
    template_state_dict: dict[str, Any],
) -> dict[str, Any]:
    if len(arrays) != len(names):
        raise ValueError("arrays and names length mismatch")
    import torch

    out: dict[str, Any] = {}
    for name, arr, template in zip(names, arrays, [template_state_dict[n] for n in names]):
        out[name] = torch.as_tensor(arr, dtype=template.dtype, device=template.device)
    return out


def align_ndarrays_to_schema(
    arrays: list[np.ndarray],
    names: list[str],
    schema: ParameterSchema,
) -> list[np.ndarray]:
    expected_names = schema.expected_names()
    if names != expected_names:
        raise ValueError(f"Parameter name order mismatch: got {names}, expected {expected_names}")
    if len(arrays) != schema.num_tensors:
        raise ValueError(f"Expected {schema.num_tensors} tensors, got {len(arrays)}")
    ordered: list[np.ndarray] = []
    for arr, spec in zip(arrays, schema.tensors):
        if tuple(arr.shape) != spec.shape:
            raise ValueError(f"Shape mismatch for {spec.name}: {arr.shape} vs {spec.shape}")
        ordered.append(arr)
    return ordered

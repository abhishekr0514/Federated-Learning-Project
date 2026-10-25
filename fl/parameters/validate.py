"""Validate client parameter payloads before aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np

from fl.errors import ParameterValidationError
from fl.parameters.schema import ParameterSchema


def _assert_finite(arr: np.ndarray, *, context: str) -> None:
    if not np.isfinite(arr).all():
        raise ParameterValidationError(f"Non-finite values in {context}")


def validate_sample_count(num_examples: int) -> None:
    if num_examples < 0:
        raise ParameterValidationError("num_examples must be >= 0")
    if num_examples == 0:
        raise ParameterValidationError("num_examples must be > 0 for weighted aggregation")


def validate_client_metrics(metrics: dict[str, Any], allowed: frozenset[str] | set[str]) -> None:
    for key in metrics:
        if key not in allowed:
            raise ParameterValidationError(f"Disallowed client metric key: {key}")
        val = metrics[key]
        if isinstance(val, (int, float)) and not np.isfinite(val):
            raise ParameterValidationError(f"Non-finite metric value for {key}")


def validate_ndarray_update(
    arrays: list[np.ndarray],
    *,
    schema: ParameterSchema,
    architecture_id: str,
    architecture_version: str,
) -> None:
    if architecture_id != schema.architecture_id:
        raise ParameterValidationError(
            f"architecture_id mismatch: {architecture_id} vs {schema.architecture_id}"
        )
    if architecture_version != schema.architecture_version:
        raise ParameterValidationError("architecture_version mismatch")

    if len(arrays) != schema.num_tensors:
        raise ParameterValidationError(
            f"Expected {schema.num_tensors} parameter tensors, got {len(arrays)}"
        )

    for arr, spec in zip(arrays, schema.tensors):
        if tuple(arr.shape) != spec.shape:
            raise ParameterValidationError(
                f"Shape mismatch for {spec.name}: expected {spec.shape}, got {arr.shape}"
            )
        if str(arr.dtype) not in (spec.dtype, "float32", "float64"):
            raise ParameterValidationError(f"Unsupported dtype for {spec.name}: {arr.dtype}")
        _assert_finite(arr, context=spec.name)


def validate_aggregated_arrays(arrays: list[np.ndarray], *, schema: ParameterSchema) -> None:
    validate_ndarray_update(
        arrays,
        schema=schema,
        architecture_id=schema.architecture_id,
        architecture_version=schema.architecture_version,
    )

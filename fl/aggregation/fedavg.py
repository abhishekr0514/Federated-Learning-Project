"""Sample-weighted FedAvg over ndarray lists (infrastructure + tests)."""

from __future__ import annotations

import numpy as np

from fl.errors import ParameterValidationError
from fl.parameters.validate import validate_aggregated_arrays, validate_sample_count
from fl.parameters.schema import ParameterSchema


def weighted_fedavg(
    client_updates: list[tuple[list[np.ndarray], int]],
    *,
    schema: ParameterSchema,
) -> list[np.ndarray]:
    """
    FedAvg: average client tensors weighted by num_examples.

    Raises on empty input, invalid sample counts, shape mismatch, or non-finite results.
    """
    if not client_updates:
        raise ParameterValidationError("No client updates to aggregate")

    total = 0
    for _, n in client_updates:
        validate_sample_count(n)
        total += n

    num_tensors = len(client_updates[0][0])
    acc = [np.zeros_like(client_updates[0][0][i], dtype=np.float64) for i in range(num_tensors)]

    for arrays, n in client_updates:
        if len(arrays) != num_tensors:
            raise ParameterValidationError("Inconsistent parameter count across clients")
        weight = n / total
        for i, arr in enumerate(arrays):
            if arr.shape != acc[i].shape:
                raise ParameterValidationError("Inconsistent parameter shapes across clients")
            acc[i] += arr.astype(np.float64) * weight

    result = [a.astype(arrays[0].dtype) for a in acc]
    validate_aggregated_arrays(result, schema=schema)
    return result

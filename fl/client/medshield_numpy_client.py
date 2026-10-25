"""Flower NumPyClient wrapper with MedShield validation."""

from __future__ import annotations

from typing import Any

from fl.client.adapter import TaskClientAdapter
from fl.errors import FederationCompatibilityError, FlowerUnavailableError, ParameterValidationError
from fl.parameters.validate import validate_client_metrics, validate_ndarray_update
from fl.privacy import ALLOWED_CLIENT_METRIC_KEYS

try:
    from flwr.client import NumPyClient
except ImportError as exc:  # pragma: no cover - handled in tests via skip
    NumPyClient = None  # type: ignore[misc, assignment]
    _FLWR_IMPORT_ERROR = exc
else:
    _FLWR_IMPORT_ERROR = None


def require_flwr() -> None:
    if NumPyClient is None:
        raise FlowerUnavailableError(str(_FLWR_IMPORT_ERROR))


class MedShieldNumPyClient(NumPyClient if NumPyClient else object):  # type: ignore[misc]
    def __init__(self, adapter: TaskClientAdapter, *, allowed_metrics: frozenset[str] | None = None) -> None:
        require_flwr()
        self.adapter = adapter
        self.allowed_metrics = allowed_metrics or ALLOWED_CLIENT_METRIC_KEYS
        self._schema = adapter.parameter_schema()

    def get_properties(self, config: dict[str, Any]) -> dict[str, float | int | str | bool]:
        return {
            "federation_id": self.adapter.federation_id,
            "task_id": self.adapter.task_id,
            "architecture_id": self.adapter.architecture_id,
            "architecture_version": self.adapter.architecture_version,
        }

    def get_parameters(self, config: dict[str, Any]):
        from flwr.common import ndarrays_to_parameters

        arrays, names = self.adapter.get_parameters()
        validate_ndarray_update(
            arrays,
            schema=self._schema,
            architecture_id=self.adapter.architecture_id,
            architecture_version=self.adapter.architecture_version,
        )
        if names != self._schema.expected_names():
            raise ParameterValidationError("Client parameter name order drift")
        return ndarrays_to_parameters(arrays)

    def fit(self, parameters, config: dict[str, Any]):
        from flwr.common import parameters_to_ndarrays

        arrays = parameters_to_ndarrays(parameters)
        names = self._schema.expected_names()
        validate_ndarray_update(
            arrays,
            schema=self._schema,
            architecture_id=self.adapter.architecture_id,
            architecture_version=self.adapter.architecture_version,
        )
        self.adapter.set_parameters(arrays, names)
        out_arrays, num_examples, metrics = self.adapter.fit(config)
        validate_ndarray_update(
            out_arrays,
            schema=self._schema,
            architecture_id=self.adapter.architecture_id,
            architecture_version=self.adapter.architecture_version,
        )
        validate_client_metrics(metrics, self.allowed_metrics)
        if num_examples != self.adapter.num_local_examples():
            raise FederationCompatibilityError("Client reported inconsistent num_examples")
        return out_arrays, num_examples, metrics

    def evaluate(self, parameters, config: dict[str, Any]):
        from flwr.common import parameters_to_ndarrays

        arrays = parameters_to_ndarrays(parameters)
        names = self._schema.expected_names()
        self.adapter.set_parameters(arrays, names)
        loss, num_examples, metrics = self.adapter.evaluate(config)
        validate_client_metrics(metrics, self.allowed_metrics)
        return float(loss), num_examples, metrics

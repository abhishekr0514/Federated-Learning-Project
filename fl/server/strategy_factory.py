"""Build Flower FedAvg strategy with MedShield validation hooks."""

from __future__ import annotations

from typing import Any

from fl.config import FederationConfig
from fl.errors import FlowerUnavailableError, ParameterValidationError
from fl.parameters.schema import ParameterSchema
from fl.parameters.validate import validate_ndarray_update, validate_sample_count

try:
    from flwr.common import FitRes, Parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg
except ImportError as exc:  # pragma: no cover
    FedAvg = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_flwr_strategy() -> None:
    if FedAvg is None:
        raise FlowerUnavailableError(str(_IMPORT_ERROR))


class ValidatedFedAvg(FedAvg if FedAvg else object):  # type: ignore[misc]
    """FedAvg that rejects malformed client updates before aggregation."""

    def __init__(
        self,
        *,
        federation_config: FederationConfig,
        parameter_schema: ParameterSchema,
        **kwargs: Any,
    ) -> None:
        require_flwr_strategy()
        super().__init__(**kwargs)
        self._federation_config = federation_config
        self._schema = parameter_schema

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            if not self._federation_config.participation.accept_failures:
                raise ParameterValidationError(
                    f"Client failures in round {server_round}: {len(failures)}"
                )
        validated: list[tuple[list, int]] = []
        for _client, fit_res in results:
            self._validate_fit_res(fit_res)
            arrays = parameters_to_ndarrays(fit_res.parameters)
            validate_ndarray_update(
                arrays,
                schema=self._schema,
                architecture_id=self._schema.architecture_id,
                architecture_version=self._schema.architecture_version,
            )
            validate_sample_count(fit_res.num_examples)
            validated.append((arrays, fit_res.num_examples))

        if not validated:
            raise ParameterValidationError("No valid client updates to aggregate")

        from fl.aggregation.fedavg import weighted_fedavg
        from flwr.common import ndarrays_to_parameters

        aggregated = weighted_fedavg(validated, schema=self._schema)
        return ndarrays_to_parameters(aggregated), {}

    def _validate_fit_res(self, fit_res: FitRes) -> None:
        props = getattr(fit_res, "metrics", {}) or {}
        for key in props:
            if key not in self._federation_config.allowed_client_metrics:
                raise ParameterValidationError(f"Disallowed metric in FitRes: {key}")


def build_fedavg_strategy(
    *,
    federation_config: FederationConfig,
    parameter_schema: ParameterSchema,
    initial_parameters: Parameters | None = None,
):
    require_flwr_strategy()
    p = federation_config.participation
    return ValidatedFedAvg(
        federation_config=federation_config,
        parameter_schema=parameter_schema,
        fraction_fit=p.fraction_fit,
        fraction_evaluate=p.fraction_evaluate,
        min_fit_clients=p.min_fit_clients,
        min_evaluate_clients=p.min_evaluate_clients,
        min_available_clients=p.min_available_clients,
        accept_failures=p.accept_failures,
        initial_parameters=initial_parameters,
    )

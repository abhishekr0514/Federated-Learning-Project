"""Task-agnostic client adapter boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from fl.parameters.schema import ParameterSchema


class TaskClientAdapter(ABC):
    """Loads local model + data; exposes Flower-compatible parameter vectors."""

    @property
    @abstractmethod
    def federation_id(self) -> str: ...

    @property
    @abstractmethod
    def task_id(self) -> str: ...

    @property
    @abstractmethod
    def architecture_id(self) -> str: ...

    @property
    @abstractmethod
    def architecture_version(self) -> str: ...

    @abstractmethod
    def parameter_schema(self) -> ParameterSchema: ...

    @abstractmethod
    def get_parameters(self) -> tuple[list[np.ndarray], list[str]]: ...

    @abstractmethod
    def set_parameters(self, arrays: list[np.ndarray], names: list[str]) -> None: ...

    @abstractmethod
    def num_local_examples(self) -> int: ...

    @abstractmethod
    def fit(self, config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]: ...

    def evaluate(self, config: dict[str, Any]) -> tuple[float, int, dict[str, float]]:
        return 0.0, self.num_local_examples(), {}


class SyntheticInfrastructureAdapter(TaskClientAdapter):
    """Tiny linear model for FL infrastructure tests only (not clinical)."""

    def __init__(
        self,
        *,
        federation_id: str,
        task_id: str,
        architecture_id: str,
        architecture_version: str,
        seed: int,
        num_examples: int,
    ) -> None:
        self._federation_id = federation_id
        self._task_id = task_id
        self._arch_id = architecture_id
        self._arch_ver = architecture_version
        self._num_examples = num_examples
        rng = np.random.default_rng(seed)
        self._w = rng.normal(size=(4, 1)).astype(np.float32)
        self._b = rng.normal(size=(1,)).astype(np.float32)

    @property
    def federation_id(self) -> str:
        return self._federation_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def architecture_id(self) -> str:
        return self._arch_id

    @property
    def architecture_version(self) -> str:
        return self._arch_ver

    def parameter_schema(self) -> ParameterSchema:
        from fl.parameters.schema import schema_from_named_arrays

        named = [("weight", self._w), ("bias", self._b)]
        return schema_from_named_arrays(
            named, architecture_id=self._arch_id, architecture_version=self._arch_ver
        )

    def get_parameters(self) -> tuple[list[np.ndarray], list[str]]:
        return [self._w.copy(), self._b.copy()], ["weight", "bias"]

    def set_parameters(self, arrays: list[np.ndarray], names: list[str]) -> None:
        mapping = dict(zip(names, arrays))
        self._w = mapping["weight"].astype(np.float32)
        self._b = mapping["bias"].astype(np.float32)

    def num_local_examples(self) -> int:
        return self._num_examples

    def fit(self, config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        # Infrastructure-only: optional noise proportional to local_epochs (no real data).
        epochs = int(config.get("local_epochs", 0))
        if epochs > 0:
            self._w += np.float32(1e-2) * epochs
            self._b += np.float32(1e-3) * epochs
        arrays, names = self.get_parameters()
        return arrays, self._num_examples, {"loss": float(config.get("loss", 0.5))}

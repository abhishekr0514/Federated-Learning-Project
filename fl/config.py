"""Federation configuration schema and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fl.errors import FederationConfigError, FederationCompatibilityError
from fl.registry import FEDERATION_REGISTRY, resolve_federation

ALLOWED_STRATEGIES = frozenset({"fedavg"})
ALLOWED_DTYPES = frozenset({"float32", "float64"})


@dataclass
class LocalTrainingConfig:
    local_epochs: int = 0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def validate(self) -> None:
        if self.local_epochs < 0:
            raise FederationConfigError("local_epochs must be >= 0")
        if self.batch_size < 1:
            raise FederationConfigError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise FederationConfigError("learning_rate must be > 0")


@dataclass
class ParticipationConfig:
    num_clients: int = 2
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    accept_failures: bool = False

    def validate(self) -> None:
        if self.num_clients < 1:
            raise FederationConfigError("num_clients must be >= 1")
        if self.min_fit_clients < 1 or self.min_evaluate_clients < 1:
            raise FederationConfigError("min_fit_clients and min_evaluate_clients must be >= 1")
        if self.min_available_clients < self.min_fit_clients:
            raise FederationConfigError("min_available_clients must be >= min_fit_clients")
        if not (0.0 < self.fraction_fit <= 1.0) or not (0.0 < self.fraction_evaluate <= 1.0):
            raise FederationConfigError("fraction_fit and fraction_evaluate must be in (0, 1]")


@dataclass
class FederationConfig:
    federation_id: str
    task_id: str
    architecture_id: str
    architecture_version: str
    aggregation_strategy: str = "fedavg"
    seed: int = 42
    num_rounds: int = 0
    output_dir: str = "data/processed/federated"
    feature_schema_id: str | None = None
    feature_schema_version: str | None = None
    preprocessing_contract_id: str | None = None
    preprocessing_contract_version: str | None = None
    parameter_schema_id: str | None = None
    participation: ParticipationConfig = field(default_factory=ParticipationConfig)
    local_training: LocalTrainingConfig = field(default_factory=LocalTrainingConfig)
    allowed_client_metrics: tuple[str, ...] = ("loss",)
    auto_start_training: bool = False
    synthetic_infrastructure_smoke: bool = False
    real_tabular_experiment: bool = False
    dataset_path: str = "data/datasets/cardio_train.csv"
    expected_dataset_sha256: str = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"
    max_federated_train_rows: int | None = None

    REQUIRED_FIELDS = (
        "federation_id",
        "task_id",
        "architecture_id",
        "architecture_version",
        "aggregation_strategy",
        "seed",
        "output_dir",
    )

    def validate(self) -> None:
        for name in self.REQUIRED_FIELDS:
            if not getattr(self, name):
                raise FederationConfigError(f"Missing required field: {name}")

        if self.aggregation_strategy not in ALLOWED_STRATEGIES:
            raise FederationConfigError(f"Unsupported aggregation_strategy: {self.aggregation_strategy}")

        spec = resolve_federation(self.federation_id, task_id=self.task_id)
        spec.assert_eligible()

        if self.architecture_id != spec.architecture_id:
            raise FederationCompatibilityError(
                f"architecture_id {self.architecture_id!r} != registry {spec.architecture_id!r}"
            )
        if self.architecture_version != spec.architecture_version:
            raise FederationCompatibilityError(
                f"architecture_version mismatch for {self.federation_id}"
            )
        if spec.feature_schema_id and self.feature_schema_id != spec.feature_schema_id:
            raise FederationCompatibilityError("feature_schema_id mismatch with federation registry")
        if spec.feature_schema_version and self.feature_schema_version != spec.feature_schema_version:
            raise FederationCompatibilityError("feature_schema_version mismatch with federation registry")

        self.participation.validate()
        self.local_training.validate()

        if self.auto_start_training:
            raise FederationConfigError(
                "auto_start_training must remain false; use explicit runner flags to start simulation."
            )

        if self.num_rounds > 0 and not (
            self.synthetic_infrastructure_smoke or self.real_tabular_experiment
        ):
            raise FederationConfigError(
                "num_rounds > 0 requires synthetic_infrastructure_smoke or real_tabular_experiment."
            )
        if self.real_tabular_experiment and self.synthetic_infrastructure_smoke:
            raise FederationConfigError(
                "real_tabular_experiment and synthetic_infrastructure_smoke are mutually exclusive."
            )
        if self.real_tabular_experiment:
            if self.federation_id != "medshield.fl.tabular_cardio.v1":
                raise FederationConfigError("real_tabular_experiment only supports tabular cardio federation.")
            if self.max_federated_train_rows is not None and self.max_federated_train_rows < 100:
                raise FederationConfigError("max_federated_train_rows too small for a meaningful experiment.")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["participation"] = asdict(self.participation)
        d["local_training"] = asdict(self.local_training)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederationConfig:
        participation = ParticipationConfig(**data.get("participation", {}))
        local_training = LocalTrainingConfig(**data.get("local_training", {}))
        kwargs = {k: v for k, v in data.items() if k not in ("participation", "local_training")}
        return cls(participation=participation, local_training=local_training, **kwargs)


def load_federation_config(path: Path | str) -> FederationConfig:
    p = Path(path)
    if not p.is_file():
        raise FederationConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = FederationConfig.from_dict(raw)
    cfg.validate()
    return cfg


def default_tabular_infrastructure_config() -> FederationConfig:
    spec = FEDERATION_REGISTRY["medshield.fl.tabular_cardio.v1"]
    return FederationConfig(
        federation_id=spec.federation_id,
        task_id=spec.task_id,
        architecture_id=spec.architecture_id,
        architecture_version=spec.architecture_version,
        feature_schema_id=spec.feature_schema_id,
        feature_schema_version=spec.feature_schema_version,
        preprocessing_contract_id=spec.preprocessing_contract_id,
        preprocessing_contract_version=spec.preprocessing_contract_version,
        num_rounds=0,
        local_training=LocalTrainingConfig(local_epochs=0),
    )

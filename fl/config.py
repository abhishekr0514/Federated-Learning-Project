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
    class_weighted_loss: bool = False

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
    real_ecg_report_experiment: bool = False
    dataset_path: str = "data/datasets/cardio_train.csv"
    expected_dataset_sha256: str = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"
    max_federated_train_rows: int | None = None
    # contiguous_record_id (default) | stratified_round_robin — simulation-only client layout
    client_partition_strategy: str = "contiguous_record_id"
    ptbxl_root: str = "data/raw/ptb-xl"
    expected_ptbxl_database_sha256: str = (
        "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"
    )
    encoder_manifest_path: str = "models/ecg_report_fusion/manifest.json"
    federated_trainable_scope: str = "fusion_head"
    max_federated_train_records: int | None = None
    ecg_report_experiment_subdir: str = "ecg_report_experiments"
    eval_max_val_records: int | None = None
    eval_max_test_records: int | None = None

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

        real_flags = (
            self.synthetic_infrastructure_smoke,
            self.real_tabular_experiment,
            self.real_ecg_report_experiment,
        )
        if sum(1 for f in real_flags if f) > 1:
            raise FederationConfigError(
                "At most one of synthetic_infrastructure_smoke, real_tabular_experiment, "
                "real_ecg_report_experiment may be true."
            )
        if self.num_rounds > 0 and not any(real_flags):
            raise FederationConfigError(
                "num_rounds > 0 requires synthetic_infrastructure_smoke, "
                "real_tabular_experiment, or real_ecg_report_experiment."
            )
        if self.real_tabular_experiment:
            if self.federation_id != "medshield.fl.tabular_cardio.v1":
                raise FederationConfigError("real_tabular_experiment only supports tabular cardio federation.")
            if self.max_federated_train_rows is not None and self.max_federated_train_rows < 100:
                raise FederationConfigError("max_federated_train_rows too small for a meaningful experiment.")
            allowed_partition = frozenset({"contiguous_record_id", "stratified_round_robin"})
            if self.client_partition_strategy not in allowed_partition:
                raise FederationConfigError(
                    f"client_partition_strategy must be one of {sorted(allowed_partition)}"
                )
        if self.real_ecg_report_experiment:
            if self.federation_id != "medshield.fl.ecg_report.v1":
                raise FederationConfigError("real_ecg_report_experiment only supports medshield.fl.ecg_report.v1.")
            allowed_scopes = frozenset(
                {"fusion_head", "fusion_head_and_ecg_encoder", "gat_head"}
            )
            if self.federated_trainable_scope not in allowed_scopes:
                raise FederationConfigError(
                    f"federated_trainable_scope must be one of {sorted(allowed_scopes)}"
                )
            allowed_ecg_partition = frozenset({"stratified_round_robin", "contiguous_patient_id"})
            if self.client_partition_strategy not in allowed_ecg_partition:
                raise FederationConfigError(
                    f"ECG+report client_partition_strategy must be one of {sorted(allowed_ecg_partition)}"
                )
            if self.max_federated_train_records is not None and self.max_federated_train_records < 8:
                raise FederationConfigError("max_federated_train_records too small for ECG+report FL.")

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

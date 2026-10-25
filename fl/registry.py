"""Registered federation task contracts (infrastructure metadata only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fl.errors import FederationBlockedError, FederationCompatibilityError

# Readiness levels (documentation constants)
READINESS_INFRASTRUCTURE = 1
READINESS_TASK_EXPERIMENT = 2
READINESS_CLINICAL_MULTIMODAL = 3


@dataclass(frozen=True)
class TaskFederationSpec:
    federation_id: str
    task_id: str
    architecture_id: str
    architecture_version: str
    feature_schema_id: str | None
    feature_schema_version: str | None
    preprocessing_contract_id: str | None
    preprocessing_contract_version: str | None
    dataset_boundary: str
    blocked: bool = False
    block_reason: str | None = None

    def assert_eligible(self) -> None:
        if self.blocked:
            raise FederationBlockedError(
                self.block_reason or f"Federation {self.federation_id} is blocked."
            )


TABULAR_CARDIO_V1 = TaskFederationSpec(
    federation_id="medshield.fl.tabular_cardio.v1",
    task_id="medshield.tabular.cardio_survey.baseline",
    architecture_id="medshield.tabular.cardio_survey.classifier",
    architecture_version="1.0.0",
    feature_schema_id="medshield.tabular.cardio_survey",
    feature_schema_version="1.0.0",
    preprocessing_contract_id="medshield.tabular.cardio_survey.preprocessing",
    preprocessing_contract_version="1.0.0",
    dataset_boundary="data/datasets/cardio_train.csv (survey cardio target only)",
    blocked=False,
)

ECG_PTbxl_V1 = TaskFederationSpec(
    federation_id="medshield.fl.ecg_ptbxl.v1",
    task_id="medshield.ecg.ptbxl.proxy_superclass",
    architecture_id="medshield.ecg.cnn_bilstm.encoder",
    architecture_version="1.0.0",
    feature_schema_id=None,
    feature_schema_version=None,
    preprocessing_contract_id="medshield.ecg.ptbxl.normalization",
    preprocessing_contract_version="1.0.0",
    dataset_boundary="data/raw/ptb-xl (ECG proxy task; optional locally)",
    blocked=False,
)

TEXT_NBME_V1 = TaskFederationSpec(
    federation_id="medshield.fl.nbme_concepts.v1",
    task_id="medshield.text.nbme.concept_spans",
    architecture_id="medshield.text.span_extractor",
    architecture_version="0.0.0",
    feature_schema_id=None,
    feature_schema_version=None,
    preprocessing_contract_id=None,
    preprocessing_contract_version=None,
    dataset_boundary="NBME concept-span task (not implemented as span model)",
    blocked=True,
    block_reason=(
        "NBME span-extraction model is not implemented; BioClinicalBERT sentence encoder "
        "must not be federated as a completed span task."
    ),
)

GAT_MULTIMODAL_V1 = TaskFederationSpec(
    federation_id="medshield.fl.gat_multimodal.v1",
    task_id="medshield.multimodal.gat.fusion",
    architecture_id="medshield.diagnostic_net.gat_fusion",
    architecture_version="1.0.0",
    feature_schema_id=None,
    feature_schema_version=None,
    preprocessing_contract_id="medshield.multimodal.fusion.preprocessing",
    preprocessing_contract_version="1.0.0",
    dataset_boundary="Requires legitimate linked multimodal cohort (blocked)",
    blocked=True,
    block_reason=(
        "Three-modality GAT federation blocked: no defensible linked multimodal target in repo. "
        "Do not fabricate cross-modal pairings."
    ),
)

FEDERATION_REGISTRY: dict[str, TaskFederationSpec] = {
    TABULAR_CARDIO_V1.federation_id: TABULAR_CARDIO_V1,
    ECG_PTbxl_V1.federation_id: ECG_PTbxl_V1,
    TEXT_NBME_V1.federation_id: TEXT_NBME_V1,
    GAT_MULTIMODAL_V1.federation_id: GAT_MULTIMODAL_V1,
}


def resolve_federation(federation_id: str, *, task_id: str | None = None) -> TaskFederationSpec:
    spec = FEDERATION_REGISTRY.get(federation_id)
    if spec is None:
        raise FederationCompatibilityError(f"Unknown federation_id: {federation_id}")
    if task_id is not None and task_id != spec.task_id:
        raise FederationCompatibilityError(
            f"task_id {task_id!r} does not match federation {federation_id!r} ({spec.task_id})."
        )
    return spec


def federation_summary() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in FEDERATION_REGISTRY.values():
        out.append(
            {
                "federation_id": spec.federation_id,
                "task_id": spec.task_id,
                "architecture_id": spec.architecture_id,
                "blocked": spec.blocked,
                "dataset_boundary": spec.dataset_boundary,
            }
        )
    return out

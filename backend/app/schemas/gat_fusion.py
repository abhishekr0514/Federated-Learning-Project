"""Pydantic schemas for the three-modality GAT fusion inference API.

Architecture: medshield.diagnostic_net.gat_fusion
Inputs: pre-computed embedding vectors for ECG (128-d), clinical text (128-d), tabular (64-d).
At least one modality must be supplied; missing modalities are zero-padded with availability masks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GATFusionEvaluateRequest(BaseModel):
    """Input payload for three-modality GAT fusion proxy evaluation.

    Callers supply *pre-computed* embedding vectors extracted from the upstream
    single-modality encoders:
      - ecg_embedding    : 128-d float list from ECGEmbeddingProvider
      - text_embedding   : 128-d float list from ClinicalTextEncodingPipeline
      - tabular_embedding: 64-d  float list from NeuralTabularEncoder

    At least one embedding must be provided. Missing embeddings are handled via
    masked zero-padding inside MedShieldDiagnosticNet.
    """

    ecg_embedding: Optional[List[float]] = Field(
        default=None,
        description="128-dimensional ECG embedding vector (output of ECGEmbeddingProvider).",
    )
    text_embedding: Optional[List[float]] = Field(
        default=None,
        description="128-dimensional clinical text embedding vector (BioClinicalBERT CLS).",
    )
    tabular_embedding: Optional[List[float]] = Field(
        default=None,
        description="64-dimensional tabular embedding vector (NeuralTabularEncoder output).",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional client idempotency token for safe retry.",
        max_length=128,
    )

    @model_validator(mode="after")
    def at_least_one_modality(self) -> "GATFusionEvaluateRequest":
        if (
            self.ecg_embedding is None
            and self.text_embedding is None
            and self.tabular_embedding is None
        ):
            raise ValueError(
                "At least one of ecg_embedding, text_embedding, or tabular_embedding must be provided."
            )
        return self

    @field_validator("ecg_embedding")
    @classmethod
    def validate_ecg_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 128:
            raise ValueError(f"ECG embedding must have exactly 128 dimensions, got {len(v)}")
        for i, val in enumerate(v):
            fv = float(val)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError(f"ECG embedding contains non-finite value at index {i}")
        return v

    @field_validator("text_embedding")
    @classmethod
    def validate_text_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 128:
            raise ValueError(f"Text embedding must have exactly 128 dimensions, got {len(v)}")
        for i, val in enumerate(v):
            fv = float(val)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError(f"Text embedding contains non-finite value at index {i}")
        return v

    @field_validator("tabular_embedding")
    @classmethod
    def validate_tabular_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 64:
            raise ValueError(f"Tabular embedding must have exactly 64 dimensions, got {len(v)}")
        for i, val in enumerate(v):
            fv = float(val)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError(f"Tabular embedding contains non-finite value at index {i}")
        return v


class GATFusionEvaluateResponse(BaseModel):
    """Output of the three-modality GAT fusion proxy evaluation.

    confidence_uncalibrated is softmax over 2 classes; NOT a calibrated probability.
    All outputs are research-only and must not be used for clinical decisions.
    """

    status: str = Field(default="EVALUATED")
    prediction_id: uuid.UUID
    patient_id: Optional[uuid.UUID] = None
    confidence_uncalibrated: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Uncalibrated softmax score for predicted class (research proxy only).",
    )
    predicted_class_index: int = Field(..., description="Predicted class index (0 or 1).")
    predicted_class_label: str = Field(..., description="Human-readable predicted class label.")
    target_definition: str
    architecture_id: str
    model_version_tag: str
    modalities_used: List[str] = Field(
        default_factory=list,
        description="Which modalities were present (non-null) in this evaluation.",
    )
    is_research_only: bool = True
    proxy_task: bool = True
    disclaimer: str
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GATFusionReadinessResponse(BaseModel):
    """Safe diagnostics describing three-modality GAT pipeline and inference readiness."""

    ready: bool
    architecture_id: str = "medshield.diagnostic_net.gat_fusion"
    architecture_version: str = "1.0.0"
    embedding_contracts: dict = Field(
        default_factory=lambda: {"ecg": 128, "clinical_text": 128, "tabular": 64}
    )
    manifest_present: bool = False
    checkpoint_present: bool = False
    checkpoint_trained: Optional[bool] = None
    model_loaded: bool = False
    linkage_gate_allowed: bool = False
    linkage_gate_verdict: str = "BLOCKED"
    blockers: List[str] = Field(default_factory=list)
    required_to_unblock: List[str] = Field(default_factory=list)
    environment: Optional[str] = "development"

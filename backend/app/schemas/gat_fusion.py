"""Pydantic schemas for two-modality GAT ECG + clinical report fusion API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class GATFusionEvaluateRequest(BaseModel):
    """Pre-computed 128-d ECG and clinical-report embeddings from authorized encoders."""

    ecg_embedding: List[float] = Field(..., description="128-d ECG embedding.")
    text_embedding: List[float] = Field(..., description="128-d clinical text embedding.")

    idempotency_key: Optional[str] = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_dims(self) -> "GATFusionEvaluateRequest":
        if len(self.ecg_embedding) != 128:
            raise ValueError(f"ecg_embedding must have 128 dimensions, got {len(self.ecg_embedding)}")
        if len(self.text_embedding) != 128:
            raise ValueError(f"text_embedding must have 128 dimensions, got {len(self.text_embedding)}")
        for name, vec in ("ecg_embedding", self.ecg_embedding), ("text_embedding", self.text_embedding):
            for i, val in enumerate(vec):
                fv = float(val)
                if fv != fv or fv in (float("inf"), float("-inf")):
                    raise ValueError(f"{name} non-finite at index {i}")
        return self


class GATFusionEvaluateResponse(BaseModel):
    status: str = Field(default="EVALUATED")
    prediction_id: uuid.UUID
    patient_id: Optional[uuid.UUID] = None
    confidence_uncalibrated: float = Field(..., ge=0.0, le=1.0)
    predicted_class_index: int
    predicted_class_label: str
    target_definition: str
    architecture_id: str
    model_version_tag: str
    modalities_used: List[str] = Field(default_factory=lambda: ["ecg", "clinical_text"])
    is_research_only: bool = True
    proxy_task: bool = True
    disclaimer: str
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GATFusionReadinessResponse(BaseModel):
    ready: bool
    architecture_id: str = "medshield.diagnostic_net.gat_ecg_report"
    architecture_version: str = "1.0.0"
    embedding_contracts: dict = Field(default_factory=lambda: {"ecg": 128, "clinical_text": 128})
    manifest_present: bool = False
    checkpoint_present: bool = False
    checkpoint_trained: Optional[bool] = None
    model_loaded: bool = False
    pairing_gate_allowed: bool = False
    pairing_gate_verdict: str = "BLOCKED"
    inference_enabled: bool = False
    blockers: List[str] = Field(default_factory=list)
    required_to_unblock: List[str] = Field(default_factory=list)
    environment: Optional[str] = "development"

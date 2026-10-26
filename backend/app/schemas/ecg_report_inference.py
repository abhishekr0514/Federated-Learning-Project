"""Pydantic schemas for the ECG + clinical report inference API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class ECGReportEvaluateRequest(BaseModel):
    """Input payload for two-modality ECG + clinical report evaluation."""

    ecg_leads: List[List[float]] = Field(
        ...,
        description="12-lead ECG waveform array of shape (12, 1000) in millivolts (mV).",
    )
    report_text: str = Field(
        ...,
        description="Clinical observation or ECG machine report narrative text.",
        min_length=1,
    )
    patient_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional authorized patient identifier to persist prediction.",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional client idempotency token for safe retry.",
        max_length=128,
    )

    @field_validator("ecg_leads")
    @classmethod
    def validate_ecg_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 12:
            raise ValueError(f"ECG signal must have exactly 12 leads, got {len(v)}")
        for i, lead in enumerate(v):
            if len(lead) != 1000:
                raise ValueError(
                    f"ECG lead {i} must have exactly 1000 samples, got {len(lead)}"
                )
            for j, sample in enumerate(lead):
                if not isinstance(sample, (int, float)):
                    raise ValueError(f"ECG sample at lead {i}, index {j} is not numeric")
                fv = float(sample)
                if fv != fv or fv in (float("inf"), float("-inf")):
                    raise ValueError(f"ECG sample at lead {i}, index {j} is non-finite (NaN or Inf)")
        return v

    @field_validator("report_text")
    @classmethod
    def validate_report(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Clinical report text must be non-empty and non-whitespace.")
        return v


class ECGReportEvaluateResponse(BaseModel):
    """Output prediction for two-modality ECG + report proxy task."""

    status: str = Field(default="EVALUATED")
    prediction_id: uuid.UUID
    patient_id: Optional[uuid.UUID] = None
    probability: float = Field(..., ge=0.0, le=1.0, description="Raw sigmoid model score for class 1 (MI/STTC)")
    predicted_class_index: int = Field(..., description="0 for NORM, 1 for MI/STTC")
    predicted_class_label: str = Field(..., description="'NORM' or 'MI/STTC'")
    target_definition: str
    architecture_id: str
    model_version_tag: str
    is_research_only: bool = True
    disclaimer: str
    provenance: str
    privacy_validation_status: str = Field(
        default="VALIDATED",
        description="Verified fail-closed privacy masking status.",
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ECGReportEmbedRequest(BaseModel):
    """Encoder-only request: same modality contract as evaluate (no persistence)."""

    ecg_leads: List[List[float]] = Field(
        ...,
        description="12-lead ECG waveform array of shape (12, 1000) in millivolts (mV).",
    )
    report_text: str = Field(
        ...,
        description="Clinical observation or ECG machine report narrative text.",
        min_length=1,
    )

    @field_validator("ecg_leads")
    @classmethod
    def validate_ecg_shape_embed(cls, v: List[List[float]]) -> List[List[float]]:
        return ECGReportEvaluateRequest.validate_ecg_shape(v)

    @field_validator("report_text")
    @classmethod
    def validate_report_embed(cls, v: str) -> str:
        return ECGReportEvaluateRequest.validate_report(v)


class ECGReportEmbedResponse(BaseModel):
    """128-d encoder outputs for downstream GAT fusion (no raw clinical text)."""

    status: str = Field(default="ENCODED")
    ecg_embedding: List[float] = Field(..., min_length=128, max_length=128)
    text_embedding: List[float] = Field(..., min_length=128, max_length=128)
    architecture_id: str
    model_version_tag: str
    privacy_validation_status: str = Field(default="VALIDATED")
    encoded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ECGReportReadinessResponse(BaseModel):
    """Safe readiness diagnostics for ECG + report model."""

    ready: bool
    blockers: List[str] = Field(default_factory=list)
    manifest_present: bool = False
    checkpoint_present: bool = False
    checkpoint_trained: Optional[bool] = None
    normalization_present: bool = False
    text_model_loaded: bool = False
    model_loaded: bool = False
    architecture_id: str = "medshield.ecg_report_fusion.v1"
    environment: Optional[str] = "development"

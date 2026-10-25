"""Schemas for multimodal proxy inference (precomputed embeddings)."""

from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from ml.inference.embedding_contract import validate_modality_embeddings


class ModalityEmbeddings(BaseModel):
    """128/128/64-d embeddings per MODEL_CONTRACT (post-encoder)."""

    ecg: List[float] = Field(..., min_length=128, max_length=128)
    clinical_text: List[float] = Field(..., min_length=128, max_length=128)
    tabular: List[float] = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def _check_finite_dims(self) -> ModalityEmbeddings:
        validate_modality_embeddings(self.ecg, self.clinical_text, self.tabular)
        return self


class DiagnosisEvaluateRequest(BaseModel):
    embeddings: ModalityEmbeddings
    idempotency_key: Optional[str] = Field(default=None, max_length=128)


class DiagnosisEvaluateResponse(BaseModel):
    status: str
    prediction_id: uuid.UUID
    patient_id: uuid.UUID
    target_definition: str
    prediction_type: str
    model_version_tag: str
    confidence_uncalibrated: float
    predicted_class_index: int
    research_only: bool = True
    proxy_task: bool = True
    disclaimer: str

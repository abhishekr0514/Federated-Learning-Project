"""Schemas for patient question workflow."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class PatientQuestionSubmitRequest(BaseModel):
    question_text: str = Field(..., min_length=1, max_length=2000)
    assigned_clinician_id: Optional[uuid.UUID] = None


class ClinicianQuestionResponseRequest(BaseModel):
    mark_reviewed: bool = False
    response_text: Optional[str] = Field(default=None, max_length=4000)

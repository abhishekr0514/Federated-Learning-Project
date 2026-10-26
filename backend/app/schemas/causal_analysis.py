"""Schemas for cohort-level DoWhy causal analysis API."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


class CausalAnalysisRequest(BaseModel):
    cohort_id: str = Field(default="gat_ecg_report_ptbxl_research_v1")
    treatment: str = Field(..., description="Exposure / treatment column name")
    outcome: str = Field(..., description="Outcome column name")
    confounders: List[str] = Field(..., min_length=1)
    effect_modifiers: List[str] = Field(default_factory=list)
    causal_graph_dot: Optional[str] = Field(
        default=None,
        description="Optional DOT/Graphviz causal graph; default smoke-cardio DAG used when omitted.",
    )

    @field_validator("confounders")
    @classmethod
    def confounders_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one confounder is required.")
        return v


class CausalReadinessResponse(BaseModel):
    summary: str
    reason: str
    blockers: List[str] = Field(default_factory=list)
    cohort_id: str
    cohort_available: bool = False
    cohort_mode: str = "UNAVAILABLE"
    metadata_aligned: bool = False
    causal_pipeline_available: bool = False
    causal_estimand_identifiable: bool = False
    causal_claim_validated: bool = False
    causal_estimation_supported: bool = False
    causal_effect_estimation_allowed: bool = False
    gat_checkpoint_ready: bool = False
    dowhy_installed: bool
    analysis_enabled: bool
    descriptive_variables: List[str] = Field(default_factory=list)
    metadata_variables: List[str] = Field(default_factory=list)
    approved_variables: List[str] = Field(default_factory=list)
    disclaimer: str


class CausalAnalysisResponse(BaseModel):
    status: str
    disclaimer: str
    cohort_id: Optional[str] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    confounders: Optional[List[str]] = None
    effect_modifiers: Optional[List[str]] = None
    estimand: Optional[str] = None
    effect_estimate: Optional[float] = None
    estimation_method: Optional[str] = None
    sample_size: Optional[int] = None
    refutation: Optional[List[Any]] = None
    refutation_passed: Optional[bool] = None
    message: Optional[str] = None
    evaluated_at: Optional[str] = None

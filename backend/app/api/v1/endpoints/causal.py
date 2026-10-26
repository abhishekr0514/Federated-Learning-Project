"""Cohort-level causal analysis (DoWhy) — research observational workflow."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, status

from backend.app.api.dependencies import get_current_user, require_role
from backend.app.core.errors import ValidationAppError
from backend.app.core.security import UserRole
from backend.app.models.identity import User
from backend.app.schemas.causal_analysis import (
    CausalAnalysisRequest,
    CausalAnalysisResponse,
    CausalReadinessResponse,
)
from backend.app.services.causal_analysis_service import CausalAnalysisService
from backend.app.services.gat_causal_cohort_service import COHORT_ID

router = APIRouter(prefix="/causal", tags=["Causal"])


@router.get("/readiness", response_model=CausalReadinessResponse)
async def get_causal_readiness() -> Dict[str, Any]:
    """Dependency, dataset, and configuration readiness for DoWhy cohort analysis."""
    return CausalAnalysisService.get_readiness(build_cohort=True)


@router.get("/cohorts")
async def list_approved_cohorts(
    _current_user: User = Depends(
        require_role(UserRole.CLINICIAN, UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN)
    ),
) -> Dict[str, Any]:
    """List approved research cohorts and variable contracts (no row-level data)."""
    return CausalAnalysisService.list_cohorts()


@router.post(
    "/analyze",
    response_model=CausalAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_cohort_causal_effect(
    body: CausalAnalysisRequest,
    current_user: User = Depends(
        require_role(UserRole.CLINICIAN, UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN)
    ),
) -> CausalAnalysisResponse:
    """
    Run model → identify → estimate → refute on an approved observational cohort.
    Does not use GAT embeddings or individual patient records from the clinical DB.
    """
    if body.cohort_id != COHORT_ID:
        raise ValidationAppError(f"Unsupported cohort_id: {body.cohort_id}")

    try:
        result = CausalAnalysisService.run_cohort_analysis(
            treatment=body.treatment,
            outcome=body.outcome,
            confounders=body.confounders,
            effect_modifiers=body.effect_modifiers,
            causal_graph=body.causal_graph_dot,
        )
    except ValueError as exc:
        raise ValidationAppError(str(exc)) from exc
    except RuntimeError as exc:
        raise ValidationAppError(str(exc)) from exc

    _ = current_user
    return CausalAnalysisResponse(**result)

"""Explainable AI (DiCE Counterfactuals) router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/explanations", tags=["Explanations"])


@router.post("/dice")
async def generate_dice_counterfactuals():
    """Generate DiCE counterfactual feature perturbations."""
    raise FeatureNotImplementedError("DiCE counterfactual explainability module pending ML implementation.")

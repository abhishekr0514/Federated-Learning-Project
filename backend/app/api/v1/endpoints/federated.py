"""Federated learning campaign orchestration router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/federated", tags=["Federated"])


@router.get("/rounds")
async def list_fl_rounds():
    """List historical federated training rounds."""
    raise FeatureNotImplementedError("Flower federated learning orchestration pending FL phase implementation.")


@router.post("/start")
async def start_fl_campaign():
    """Initiate a new federated training campaign."""
    raise FeatureNotImplementedError("FL campaign trigger pending implementation.")

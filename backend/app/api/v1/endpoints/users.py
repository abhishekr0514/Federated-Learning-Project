"""User management router skeleton."""

from fastapi import APIRouter
from backend.app.core.errors import FeatureNotImplementedError

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/")
async def list_users():
    """List system users."""
    raise FeatureNotImplementedError("User directory management pending User/Role contract implementation.")


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Retrieve specific user profile."""
    raise FeatureNotImplementedError("User profile lookup pending implementation.")

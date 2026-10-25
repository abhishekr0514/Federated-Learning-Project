"""Platform capability status (truthful subsystem readiness)."""

from typing import Any, Dict

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_current_user
from backend.app.models.identity import User
from backend.app.services.capability_registry import get_platform_capability_report

router = APIRouter(prefix="/capabilities", tags=["Capabilities"])


@router.get("")
async def get_capabilities(
    _current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return centralized subsystem capability states and non-secret diagnostics."""
    return get_platform_capability_report()

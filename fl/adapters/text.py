"""Clinical text / NBME federation interface."""

from __future__ import annotations

from fl.errors import FederationBlockedError
from fl.registry import TEXT_NBME_V1


def assert_nbme_federation_allowed() -> None:
    TEXT_NBME_V1.assert_eligible()


def nbme_federation_readiness() -> dict[str, str | bool]:
    return {
        "federation_id": TEXT_NBME_V1.federation_id,
        "ready": False,
        "blocked": True,
        "reason": TEXT_NBME_V1.block_reason or "blocked",
    }


class NBMEFederationAdapterStub:
    def __init__(self) -> None:
        raise FederationBlockedError(TEXT_NBME_V1.block_reason or "NBME federation blocked")

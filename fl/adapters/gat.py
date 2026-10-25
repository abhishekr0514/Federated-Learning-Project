"""Multimodal GAT federation — explicitly blocked."""

from __future__ import annotations

from fl.errors import FederationBlockedError
from fl.registry import GAT_MULTIMODAL_V1


def assert_gat_federation_allowed() -> None:
    GAT_MULTIMODAL_V1.assert_eligible()


def gat_federation_readiness() -> dict[str, str | bool]:
    return {
        "federation_id": GAT_MULTIMODAL_V1.federation_id,
        "ready": False,
        "blocked": True,
        "reason": GAT_MULTIMODAL_V1.block_reason or "blocked",
    }


class GATFederationAdapterStub:
    def __init__(self) -> None:
        raise FederationBlockedError(GAT_MULTIMODAL_V1.block_reason or "GAT federation blocked")

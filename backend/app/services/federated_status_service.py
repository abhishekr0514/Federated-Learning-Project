"""Truthful Flower / federated orchestration status."""

from __future__ import annotations

from typing import Any, Dict

from backend.app.services.flower_orchestration_service import FlowerOrchestrationService


def get_flower_capability_status() -> Dict[str, Any]:
    return FlowerOrchestrationService.get_status()

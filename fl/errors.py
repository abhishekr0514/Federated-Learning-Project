"""Typed errors for MedShield-FL Flower infrastructure."""

from __future__ import annotations


class FederatedLearningError(Exception):
    """Base class for FL infrastructure errors."""


class FederationConfigError(FederatedLearningError):
    """Invalid or incomplete federation configuration."""


class FederationCompatibilityError(FederatedLearningError):
    """Task, architecture, or contract mismatch across federation participants."""


class ParameterValidationError(FederatedLearningError):
    """Malformed, incompatible, or non-finite model parameters."""


class CheckpointError(FederatedLearningError):
    """Checkpoint metadata or payload incompatible or corrupted."""


class FederationBlockedError(FederatedLearningError):
    """Federation explicitly blocked (e.g. multimodal GAT without linked data)."""


class ClientLocalDataError(FederatedLearningError):
    """Client-side dataset or local configuration invalid."""


class FlowerUnavailableError(FederatedLearningError):
    """Flower (flwr) is not installed or failed to import."""

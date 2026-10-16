"""Neural Models Package (ECG 1D-CNN+BiLSTM, BioClinicalBERT Projection, Neural Tabular Encoder, GAT Fusion)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ECGEncoder",
    "ECGBinaryClassifier",
    "ClinicalTextEncoder",
    "ClinicalTextEncoderConfig",
    "NeuralTabularEncoder",
    "TabularBinaryClassifier",
    "MedShieldDiagnosticNet",
    "FusionModelConfig",
]


def __getattr__(name: str) -> Any:
    if name in ("ECGEncoder", "ECGBinaryClassifier"):
        from ml.models.ecg_encoder import ECGBinaryClassifier, ECGEncoder

        return ECGEncoder if name == "ECGEncoder" else ECGBinaryClassifier
    if name in ("ClinicalTextEncoder", "ClinicalTextEncoderConfig"):
        from ml.models.clinical_text_encoder import ClinicalTextEncoder, ClinicalTextEncoderConfig

        return ClinicalTextEncoder if name == "ClinicalTextEncoder" else ClinicalTextEncoderConfig
    if name in ("NeuralTabularEncoder", "TabularBinaryClassifier"):
        from ml.models.tabular_encoder import NeuralTabularEncoder, TabularBinaryClassifier

        return NeuralTabularEncoder if name == "NeuralTabularEncoder" else TabularBinaryClassifier
    if name in ("MedShieldDiagnosticNet", "FusionModelConfig"):
        from ml.models.gnn_fusion import FusionModelConfig, MedShieldDiagnosticNet

        return MedShieldDiagnosticNet if name == "MedShieldDiagnosticNet" else FusionModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)

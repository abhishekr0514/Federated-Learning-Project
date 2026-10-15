"""Training-fitted ECG normalization (per-lead z-score)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ECGNormalizationStats:
    mean: np.ndarray  # shape (12,)
    std: np.ndarray  # shape (12,)
    protocol_version: str = "1.0.0"

    def transform(self, signal_mv: np.ndarray) -> np.ndarray:
        """signal_mv: (12, 1000) float."""
        if signal_mv.shape != (12, 1000):
            raise ValueError(f"Expected (12, 1000), got {signal_mv.shape}")
        out = (signal_mv - self.mean[:, None]) / self.std[:, None]
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ECGNormalizationStats":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            protocol_version=data.get("protocol_version", "1.0.0"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ECGNormalizationStats":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def fit_normalization_from_signals(signals: list[np.ndarray]) -> ECGNormalizationStats:
    """Fit per-lead mean/std from list of (12, 1000) mV arrays."""
    if not signals:
        raise ValueError("Cannot fit normalization with zero signals")
    stacked = np.stack(signals, axis=0)  # (N, 12, 1000)
    mean = stacked.mean(axis=(0, 2))
    std = stacked.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return ECGNormalizationStats(mean=mean, std=std)

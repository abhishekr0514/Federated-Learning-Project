"""Multimodal Dataset Modules."""

from __future__ import annotations

from typing import Any

__all__ = ["PTBXLECGDataset", "load_ecg_mv"]


def __getattr__(name: str) -> Any:
    if name in ("PTBXLECGDataset", "load_ecg_mv"):
        from ml.datasets.ptbxl_ecg import PTBXLECGDataset, load_ecg_mv

        if name == "PTBXLECGDataset":
            return PTBXLECGDataset
        return load_ecg_mv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)

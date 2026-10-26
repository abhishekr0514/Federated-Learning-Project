"""Device selection for ECG+report federated experiments."""

from __future__ import annotations

import os

import torch


def resolve_fl_device() -> torch.device:
    pref = os.environ.get("MEDSHIELD_FL_DEVICE", "auto").strip().lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def assert_model_on_device(model: torch.nn.Module, device: torch.device) -> None:
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            raise RuntimeError(
                f"Parameter {name} on {param.device}, expected device type {device.type}"
            )

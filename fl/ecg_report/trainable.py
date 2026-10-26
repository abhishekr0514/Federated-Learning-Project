"""Trainable-parameter selection for ECG+report federated experiments."""

from __future__ import annotations

from typing import Any

import torch

FUSION_HEAD_KEYS = frozenset({"head.weight", "head.bias"})
ALLOWED_SCOPES = frozenset({"fusion_head", "fusion_head_and_ecg_encoder"})


def _is_ecg_encoder_param(name: str) -> bool:
    return name.startswith("ecg_encoder.")


def apply_trainable_scope(model: torch.nn.Module, scope: str) -> tuple[list[str], list[str]]:
    """
    Configure requires_grad and return (trainable_names, frozen_names) sorted.
    fusion_head: only linear head trainable; all encoders frozen.
    fusion_head_and_ecg_encoder: head + ecg_encoder; text_encoder (incl. BERT) frozen.
    """
    if scope not in ALLOWED_SCOPES:
        raise ValueError(f"Unsupported federated_trainable_scope: {scope}")

    for param in model.parameters():
        param.requires_grad = False

    trainable: list[str] = []
    frozen: list[str] = []
    for name, param in model.named_parameters():
        if scope == "fusion_head" and name in FUSION_HEAD_KEYS:
            param.requires_grad = True
            trainable.append(name)
        elif scope == "fusion_head_and_ecg_encoder" and (
            name in FUSION_HEAD_KEYS or _is_ecg_encoder_param(name)
        ):
            param.requires_grad = True
            trainable.append(name)
        else:
            frozen.append(name)
    return sorted(trainable), sorted(frozen)


def trainable_state_dict(model: torch.nn.Module, trainable_names: list[str]) -> dict[str, Any]:
    state = model.state_dict()
    return {name: state[name] for name in trainable_names}


def merge_trainable_into_model(
    model: torch.nn.Module,
    arrays: list[Any],
    names: list[str],
    trainable_names: list[str],
) -> None:
    if names != trainable_names:
        raise ValueError("Trainable parameter name mismatch")
    state = model.state_dict()
    for name, arr in zip(names, arrays):
        template = state[name]
        state[name] = torch.as_tensor(arr, dtype=template.dtype, device=template.device)
    model.load_state_dict(state, strict=True)

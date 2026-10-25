"""Modality embedding vector contracts for GAT fusion inference."""

from __future__ import annotations

from typing import Sequence

from ml.inference.constants import ECG_EMBED_DIM, TABULAR_EMBED_DIM, TEXT_EMBED_DIM


def validate_embedding_vector(name: str, values: Sequence[float], expected_dim: int) -> None:
    if len(values) != expected_dim:
        raise ValueError(f"{name} embedding must have length {expected_dim}, got {len(values)}")
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            raise ValueError(f"{name} embedding index {i} is not numeric")
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            raise ValueError(f"{name} embedding contains non-finite value at index {i}")


def validate_modality_embeddings(
    ecg: Sequence[float],
    clinical_text: Sequence[float],
    tabular: Sequence[float],
) -> None:
    validate_embedding_vector("ecg", ecg, ECG_EMBED_DIM)
    validate_embedding_vector("clinical_text", clinical_text, TEXT_EMBED_DIM)
    validate_embedding_vector("tabular", tabular, TABULAR_EMBED_DIM)

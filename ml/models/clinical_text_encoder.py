"""BioClinicalBERT clinical text encoder (Prompt 13)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn

from ml.text.errors import ClinicalTextEncoderOutputError

BIO_CLINICAL_BERT_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_MAX_SEQUENCE_LENGTH = 512
BERT_CLS_DIM = 768
TEXT_EMBEDDING_DIM = 128


class TokenizerLike(Protocol):
    def __call__(
        self,
        text: list[str] | str,
        *,
        padding: bool | str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Any]: ...


class BertModelLike(Protocol):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Any: ...


@dataclass(frozen=True)
class ClinicalTextEncoderConfig:
    model_id: str = BIO_CLINICAL_BERT_MODEL_ID
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    embedding_dim: int = TEXT_EMBEDDING_DIM

    def __post_init__(self) -> None:
        if self.max_sequence_length < 1 or self.max_sequence_length > 512:
            raise ValueError("max_sequence_length must be in [1, 512] for Bio_ClinicalBERT.")


class ClinicalTextEncoder(nn.Module):
    """
    BioClinicalBERT CLS → Linear(768, 128).

    Expects already-masked text tokenization via external tokenizer call.
    """

    def __init__(
        self,
        bert: nn.Module,
        *,
        config: ClinicalTextEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ClinicalTextEncoderConfig()
        self.bert = bert
        self.projection = nn.Linear(BERT_CLS_DIM, self.config.embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if not hasattr(outputs, "last_hidden_state"):
            raise ClinicalTextEncoderOutputError("BERT output missing last_hidden_state.")
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != BERT_CLS_DIM:
            raise ClinicalTextEncoderOutputError(
                f"Unexpected hidden state shape: {tuple(hidden.shape)}"
            )
        cls = hidden[:, 0, :]
        emb = self.projection(cls)
        if emb.shape != (input_ids.shape[0], self.config.embedding_dim):
            raise ClinicalTextEncoderOutputError(f"Unexpected embedding shape: {tuple(emb.shape)}")
        if not torch.isfinite(emb).all():
            raise ClinicalTextEncoderOutputError("Non-finite values in text embedding.")
        return emb


def try_load_pretrained_encoder(
    config: ClinicalTextEncoderConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> tuple[ClinicalTextEncoder, Any, dict[str, Any]]:
    """
    Load genuine pretrained tokenizer-backed encoder. Raises ClinicalTextModelUnavailableError on failure.
    """
    from transformers import AutoModel, AutoTokenizer

    from ml.text.errors import ClinicalTextModelUnavailableError

    cfg = config or ClinicalTextEncoderConfig()
    meta: dict[str, Any] = {
        "model_id": cfg.model_id,
        "max_sequence_length": cfg.max_sequence_length,
        "pretrained_weights_loaded": False,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        bert = AutoModel.from_pretrained(cfg.model_id)
        meta["pretrained_weights_loaded"] = True
        meta["tokenizer_class"] = tokenizer.__class__.__name__
        meta["model_class"] = bert.__class__.__name__
    except Exception as exc:
        raise ClinicalTextModelUnavailableError(
            "Failed to load BioClinicalBERT tokenizer or weights."
        ) from exc

    encoder = ClinicalTextEncoder(bert, config=cfg)
    if device is not None:
        encoder = encoder.to(device)
        meta["device"] = str(device)
    meta["trainable_projection"] = True
    return encoder, tokenizer, meta

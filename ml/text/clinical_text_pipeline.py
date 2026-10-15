"""End-to-end masked clinical text → 128-d embedding (research only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ml.models.clinical_text_encoder import (
    BIO_CLINICAL_BERT_MODEL_ID,
    ClinicalTextEncoder,
    ClinicalTextEncoderConfig,
    DEFAULT_MAX_SEQUENCE_LENGTH,
)
from ml.preprocessing.clinical_text_privacy import mask_clinical_text_for_encoding
from ml.text.errors import TextInputValidationError

__all__ = [
    "BIO_CLINICAL_BERT_MODEL_ID",
    "ClinicalTextEncoderConfig",
    "ClinicalTextEncodingPipeline",
]


def validate_text_input(text: str | None, *, index: int | None = None) -> str:
    if text is None:
        raise TextInputValidationError(_msg("Text is None.", index))
    if not isinstance(text, str):
        raise TextInputValidationError(_msg("Text must be a string.", index))
    if text == "":
        raise TextInputValidationError(_msg("Text is empty.", index))
    if not text.strip():
        raise TextInputValidationError(_msg("Text is whitespace-only.", index))
    return text


def _msg(base: str, index: int | None) -> str:
    if index is None:
        return base
    return f"{base} (batch index={index})"


def validate_text_batch(texts: Sequence[str | None]) -> list[str]:
    """All-or-nothing batch validation preserving order."""
    validated: list[str] = []
    for i, text in enumerate(texts):
        validated.append(validate_text_input(text, index=i))
    return validated


@dataclass
class ClinicalTextEncodingPipeline:
    """
    Privacy-first encoding: mask → tokenize → BioClinicalBERT → CLS → Linear(768,128).

    Does not call external inference APIs. Does not log clinical text.
    """

    encoder: ClinicalTextEncoder
    tokenizer: Any
    config: ClinicalTextEncoderConfig
    available: bool = True

    @classmethod
    def from_pretrained(
        cls,
        *,
        config: ClinicalTextEncoderConfig | None = None,
        device: torch.device | str | None = None,
    ) -> tuple["ClinicalTextEncodingPipeline", dict[str, Any]]:
        from ml.models.clinical_text_encoder import try_load_pretrained_encoder

        cfg = config or ClinicalTextEncoderConfig()
        encoder, tokenizer, meta = try_load_pretrained_encoder(cfg, device=device)
        pipe = cls(encoder=encoder, tokenizer=tokenizer, config=cfg, available=True)
        meta["pipeline"] = "ClinicalTextEncodingPipeline"
        return pipe, meta

    def encode_texts(
        self,
        texts: Sequence[str | None],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Returns (batch_size, 128) embeddings for masked inputs.
        Raises on any invalid text, masking failure, or model output error.
        """
        validated = validate_text_batch(texts)
        masked: list[str] = []
        for i, raw in enumerate(validated):
            try:
                masked.append(mask_clinical_text_for_encoding(raw))
            except Exception as exc:
                from ml.text.errors import MaskingFailedError

                if isinstance(exc, MaskingFailedError):
                    raise MaskingFailedError(f"Masking failed at batch index={i}.") from exc
                raise

        tokenized = self.tokenizer(
            masked,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            self.encoder.to(device)

        self.encoder.train(self.encoder.training)
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

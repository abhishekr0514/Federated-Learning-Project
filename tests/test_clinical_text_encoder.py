"""Prompt 13 — BioClinicalBERT text encoder tests (mocked; no weight download)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from ml.models.clinical_text_encoder import (
    BERT_CLS_DIM,
    ClinicalTextEncoder,
    ClinicalTextEncoderConfig,
    try_load_pretrained_encoder,
)
from ml.preprocessing.clinical_text_privacy import mask_clinical_text_for_encoding
from ml.text.clinical_text_pipeline import (
    ClinicalTextEncodingPipeline,
    validate_text_batch,
    validate_text_input,
)
from ml.text.errors import (
    ClinicalTextModelUnavailableError,
    MaskingFailedError,
    TextInputValidationError,
)


class MockTokenizer:
    def __init__(self, vocab_size: int = 1000, max_len: int = 512):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.last_texts: list[str] | None = None

    def __call__(self, text, *, padding, truncation, max_length, return_tensors):
        self.last_texts = list(text) if isinstance(text, (list, tuple)) else [text]
        batch = len(self.last_texts)
        seq = min(max_length, 16)
        return {
            "input_ids": torch.ones(batch, seq, dtype=torch.long),
            "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        }


class MockBert(nn.Module):
    def __init__(self, hidden=BERT_CLS_DIM):
        super().__init__()
        self.hidden = hidden
        self.last_attention_mask = None

    def forward(self, *, input_ids, attention_mask):
        self.last_attention_mask = attention_mask
        b, s = input_ids.shape
        h = torch.randn(b, s, self.hidden)
        h[:, 0, :] = torch.arange(b, dtype=torch.float32).unsqueeze(1).expand(b, self.hidden) * 0.01
        return SimpleNamespace(last_hidden_state=h)


def test_validate_text_inputs():
    with pytest.raises(TextInputValidationError):
        validate_text_input(None)
    with pytest.raises(TextInputValidationError):
        validate_text_input("")
    with pytest.raises(TextInputValidationError):
        validate_text_input("   ")
    assert validate_text_input("  chest pain  ") == "  chest pain  "


def test_batch_validation_preserves_order_and_fails_deterministically():
    with pytest.raises(TextInputValidationError) as exc:
        validate_text_batch(["ok", ""])
    assert "index=1" in str(exc.value)
    out = validate_text_batch(["a", "b"])
    assert out == ["a", "b"]


def test_cls_projection_shape_and_finite():
    bert = MockBert()
    enc = ClinicalTextEncoder(bert)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(ids)
    emb = enc(ids, mask)
    assert emb.shape == (2, 128)
    assert torch.isfinite(emb).all()


def test_attention_mask_propagated_to_bert():
    bert = MockBert()
    enc = ClinicalTextEncoder(bert)
    ids = torch.tensor([[1, 2]])
    mask = torch.tensor([[1, 0]])
    enc(ids, mask)
    assert torch.equal(bert.last_attention_mask, mask)


def test_tokenizer_contract_padding_and_max_length():
    tok = MockTokenizer()
    tok(["hello", "world"], padding=True, truncation=True, max_length=8, return_tensors="pt")
    assert tok.last_texts == ["hello", "world"]
    out = tok(["x"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    assert out["input_ids"].shape[1] <= 512
    assert "attention_mask" in out


def test_pipeline_masks_before_tokenization():
    bert = MockBert()
    enc = ClinicalTextEncoder(bert)
    tok = MockTokenizer()
    pipe = ClinicalTextEncodingPipeline(encoder=enc, tokenizer=tok, config=ClinicalTextEncoderConfig())
    note = "Patient Name: John Smith. LDL 160 mg/dL."
    with mock.patch(
        "ml.text.clinical_text_pipeline.mask_clinical_text_for_encoding",
        wraps=mask_clinical_text_for_encoding,
    ) as mask_fn:
        pipe.encode_texts([note])
        mask_fn.assert_called_once()
        assert tok.last_texts is not None
        assert "[PATIENT_NAME]" in tok.last_texts[0]
        assert "John Smith" not in tok.last_texts[0]


def test_masking_failure_blocks_encoding():
    bert = MockBert()
    enc = ClinicalTextEncoder(bert)
    tok = MockTokenizer()
    pipe = ClinicalTextEncodingPipeline(encoder=enc, tokenizer=tok, config=ClinicalTextEncoderConfig())
    with mock.patch(
        "ml.text.clinical_text_pipeline.mask_clinical_text_for_encoding",
        side_effect=MaskingFailedError("MASKING_VALIDATION_FAILED"),
    ):
        with pytest.raises(MaskingFailedError):
            pipe.encode_texts(["valid text"])


def test_invalid_bert_output_rejected():
    class BadBert(nn.Module):
        def forward(self, *, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=torch.randn(input_ids.shape[0], input_ids.shape[1], 64))

    enc = ClinicalTextEncoder(BadBert())
    with pytest.raises(Exception):
        enc(torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long))


def test_pretrained_load_failure_is_explicit():
    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("offline")):
        with pytest.raises(ClinicalTextModelUnavailableError):
            try_load_pretrained_encoder()


def test_error_messages_exclude_clinical_content():
    sample = "Patient Name: Jane Doe. Troponin elevated."
    try:
        validate_text_input(sample)
    except TextInputValidationError as exc:
        pytest.fail("unexpected")
    with pytest.raises(MaskingFailedError):
        with mock.patch(
            "ml.preprocessing.clinical_text_privacy.MaskingValidator.validate_masking",
            return_value=(False, "MASKING_VALIDATION_FAILED: test"),
        ):
            mask_clinical_text_for_encoding(sample)
    # Ensure we never put the note in raised validation errors from our layer
    with pytest.raises(TextInputValidationError) as exc:
        validate_text_input("")
    assert "Jane" not in str(exc.value)
    assert "Troponin" not in str(exc.value)


def test_nbme_schema_validator(tmp_path):
    from ml.datasets.nbme_text import validate_nbme_schema

    d = tmp_path
    (d / "patient_notes.csv").write_text("pn_num,case_num,pn_history\n1,0,note\n", encoding="utf-8")
    (d / "train.csv").write_text(
        "id,case_num,pn_num,feature_num,annotation,location\n1,0,1,000,a,l\n",
        encoding="utf-8",
    )
    (d / "features.csv").write_text("feature_num,case_num,feature_text\n000,0,f\n", encoding="utf-8")
    rep = validate_nbme_schema(d)
    assert all(rep["columns_ok"].values())

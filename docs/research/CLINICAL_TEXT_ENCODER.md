# Clinical Text Encoder (Prompt 13)

## Architecture (research phase)

```
Clinical text
  → PrivacyMasker + MaskingValidator (fail-closed)
  → AutoTokenizer (emilyalsentzer/Bio_ClinicalBERT)
  → input_ids + attention_mask
  → BioClinicalBERT
  → CLS (last_hidden_state[:, 0, :], 768-d)
  → Linear(768 → 128)
  → (batch_size, 128)
```

**Note:** `docs/MODEL_CONTRACT.md` describes a deeper projection head (768→256→128 with GELU/Dropout/LayerNorm). Prompt 13 implements a **single Linear(768,128)** for the research encoder; full contract head may be applied before multimodal fusion (see ADR-007).

## Privacy

- Reuses `backend.app.services.privacy_masking.PrivacyMasker` and `MaskingValidator`.
- Masking runs **before** tokenization (`ml/preprocessing/clinical_text_privacy.py`).
- No external inference APIs; no logging of raw/masked text or token ids.
- Heuristic regex masking — **not** guaranteed complete de-identification.

## Missing text policy

- `None`, empty, or whitespace-only inputs → `TextInputValidationError` with batch index.
- Batches are **all-or-nothing** (order preserved; no silent zero embeddings).

## Model loading

- `ClinicalTextEncodingPipeline.from_pretrained()` loads Hugging Face weights locally.
- Failure → `ClinicalTextModelUnavailableError` (no hash/random/zero fallback).

## NBME dataset

- `ml/datasets/nbme_text.py` validates schema only.
- Task: clinical phrase span ID — **not** heart-disease labels; **no** linkage to PTB-XL/cardio.

## Validation status

- Unit tests use **mocked** tokenizer/BERT (no weight download in CI).
- Genuine weight load is optional/local (`from_pretrained`); not required for pytest.

## Boundaries

- Multimodal GAT fusion: **not implemented**
- Production inference endpoints: **disabled**

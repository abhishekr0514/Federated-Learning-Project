# Multimodal GAT Fusion Architecture (Prompt 15)

Research-only implementation of **MedShieldDiagnosticNet** — architecture, masks, and checkpoints. **Not** a trained diagnostic model; **not** authorized for patient care or production inference.

## Scientific boundaries

| Cohort | Role |
|--------|------|
| PTB-XL | Separate ECG-only proxy task |
| cardio_train.csv | Separate survey target task |
| NBME | Concept/span extraction, not CVD diagnosis |

No patient-level linkage or shared supervised target exists across cohorts. This module accepts **precomputed embeddings only** (synthetic in unit tests). Do not join datasets or report diagnostic performance from random initialization.

## Architecture ID

- **ID:** `medshield.diagnostic_net.gat_fusion`
- **Version:** `1.0.0`
- **Code:** `ml/models/gnn_fusion.py`

## Tensor shape trace (batch size `B`)

| Step | Shape | Notes |
|------|-------|--------|
| ECG embedding | `[B, 128]` | From ECG encoder contract |
| Text embedding | `[B, 128]` | From clinical text encoder |
| Tabular embedding | `[B, 64]` | From tabular encoder |
| Project ECG | `Linear(128→128)` | Node 0 |
| Project text | `Linear(128→128)` | Node 1 |
| Project tabular | `Linear(64→128)` | Node 2 |
| Graph nodes | `[B, 3, 128]` | Order: ECG, text, tabular |
| GAT layer 1 | `in=128, out=128, heads=4, concat=False` | PyG `GATConv`, fully connected 3-node graph |
| GAT layer 2 | `in=128, out=128, heads=1, concat=False` | |
| After GAT | `[B, 3, 128]` | |
| Flatten | `[B, 384]` | |
| Hidden | `Linear(384→128)` → ReLU → Dropout(0.3) | |
| Logits | `[B, 2]` | Binary head (research only) |

GAT inter-layer activation: `ELU` on layer outputs (see ADR-009). GAT attention dropout: `0.0` (contract silent).

## Modality availability and quality

Typed input: `FusionModelInput` with optional embedding tensors and boolean masks `[B, 3]`:

0 = ECG, 1 = clinical text, 2 = tabular.

- **Availability** ≠ **quality**. A modality may be present but rejected (`quality_ok=0`).
- **`modality_used`** = `available ∧ quality_ok ∧ embedding_present`. Missing tensors are never marked used.
- **Full forward** requires all three modalities available, quality-accepted, and finite embeddings for every sample. Otherwise `FusionInputError` (no imputation, no zero-filled fake embeddings).
- **`describe_modality_status()`** returns completeness and reasons without running GAT when fusion is unsupported.

Outputs (`FusionResearchOutput`):

- Logits, softmax probabilities, argmax class, **uncalibrated** confidence (= max prob).
- Explicit `complete` flag and per-sample `incomplete_reasons`.

Softmax confidence is **not** clinical certainty.

## Checkpoints

`save_fusion_checkpoint` / `load_fusion_checkpoint`:

- Metadata: architecture id/version, config, modality order, input dims, `trained` flag (default **false** for init), PyTorch version.
- **No** raw ECG, text, or patient identifiers.
- Incompatible id/version → `FusionCheckpointError`.
- `strict=True` state dict load; no partial silent merge.

## Missing-modality policy

`MODEL_CONTRACT.md` does not define an approved imputation strategy. **Gap:** fusion forward is **all-or-nothing** per batch sample set; partial modalities must use status reporting only, not fabricated embeddings.

## Training and inference

- Fused model **not trained** in Prompt 15.
- `POST /api/v1/diagnosis/evaluate` remains **501 / not implemented** — unchanged.

## Tests

`tests/test_gnn_fusion.py` — synthetic embeddings only; not performance evidence.

## Blockers (unchanged)

- Verified cross-cohort patient keys and compatible multimodal label.
- Encoder contract upgrades (full text head, contract tabular block) before production fusion training.
- Calibration policy on a linked evaluation cohort.

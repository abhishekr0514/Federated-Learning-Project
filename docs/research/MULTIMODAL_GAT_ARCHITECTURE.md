# Multimodal GAT Fusion Architecture (Prompt 15 / Prompt 19 Audit)

Research-only implementation of **MedShieldDiagnosticNet** — architecture, masks, structural validation, and checkpoints. **Not** a trained diagnostic model; **not** authorized for patient care or production inference.

## Scientific boundaries

| Cohort | Role | Linkage Status |
|--------|------|----------------|
| PTB-XL | Separate ECG-only proxy task (`NORM` vs `MI` ∨ `STTC`) | Independent cohort; no cross-cohort linkage |
| cardio_train.csv | Separate survey target task (`cardio`) | Independent cohort; no cross-cohort linkage |
| NBME | Concept/span extraction, not CVD diagnosis | Independent cohort; no cross-cohort linkage |

No patient-level linkage or shared supervised target exists across cohorts. This module accepts **precomputed embeddings only** (synthetic in unit tests). Do not join datasets or report diagnostic performance from random initialization.

## Architecture ID & Configuration

- **ID:** `medshield.diagnostic_net.gat_fusion`
- **Version:** `1.0.0`
- **Code:** `ml/models/gnn_fusion.py`
- **Default Sub-Encoder Versions:**
  - ECG baseline protocol: `v1.0.0`
  - Tabular schema: `v1.0.0` (`medshield.tabular.cardio_survey`)
  - Text encoder: `v1.0.0` (`emilyalsentzer/Bio_ClinicalBERT`)

## Tensor shape trace (batch size `B`)

| Step | Shape | Notes |
|------|-------|--------|
| ECG embedding | `[B, 128]` | From ECG encoder contract (`ml/models/ecg_encoder.py`) |
| Text embedding | `[B, 128]` | From clinical text encoder (`ml/models/clinical_text_encoder.py`) |
| Tabular embedding | `[B, 64]` | From tabular encoder (`ml/models/tabular_encoder.py`) |
| Project ECG | `Linear(128→128)` | Node 0 projection |
| Project text | `Linear(128→128)` | Node 1 projection |
| Project tabular | `Linear(64→128)` | Node 2 projection |
| Graph nodes | `[B, 3, 128]` | Node order: ECG (0), clinical text (1), tabular (2) |
| GAT layer 1 | `in=128, out=128, heads=4, concat=False` | PyG `GATConv`, fully connected 3-node graph with self-loops |
| GAT layer 2 | `in=128, out=128, heads=1, concat=False` | PyG `GATConv`, fully connected 3-node graph with self-loops |
| After GAT | `[B, 3, 128]` | Nonlinearity: `ELU` on GATConv layer outputs (ADR-009) |
| Flatten | `[B, 384]` | Flattened along modality dimension `(0:128, 128:256, 256:384)` |
| Hidden | `Linear(384→128)` → ReLU → Dropout(0.3) | `classifier_hidden` |
| Logits | `[B, 2]` | `classifier_out` binary classification head (research only) |

GAT inter-layer activation: `ELU` on layer outputs (see ADR-009). GAT attention dropout: `0.0`.

## Modality availability, quality gates & input validation

Typed input: `FusionModelInput` with optional embedding tensors and boolean masks `[B, 3]`:

Index mapping: 0 = ECG, 1 = clinical text, 2 = tabular.

- **Availability** ≠ **quality**. A modality may be present but rejected (`quality_ok=0`).
- **`modality_used`** = `available ∧ quality_ok ∧ embedding_present`. Missing, non-floating, non-finite, or dimensionally incorrect tensors are never marked used.
- **Fail-Closed Validation**:
  - Rejects empty batches ($B < 1$).
  - Rejects wrong tensor ranks (must be 2D `[B, D]`).
  - Rejects wrong embedding dimensions (`ECG != 128`, `Text != 128`, `Tabular != 64`).
  - Rejects non-floating point dtypes.
  - Rejects NaN and infinity in embeddings.
  - Rejects invalid ECG quality status (`quality_ok=0` fails closed).
  - Rejects non-binary or non-finite mask values.
  - Rejects device mismatches among provided embeddings or between embeddings and model device.
- **Full forward** requires all three modalities available, quality-accepted, and finite embeddings for every sample. Otherwise raises `FusionInputError` (no imputation, zero-filling, or fabricated embeddings).
- **`describe_modality_status()`** returns completeness and typed reasons without running GAT when fusion is unsupported.

Outputs (`FusionResearchOutput`):
- Logits, softmax probabilities, argmax class, **uncalibrated** confidence (= max prob).
- Explicit `complete` flag and per-sample `incomplete_reasons`.
- Softmax confidence is **not** clinical certainty.

## Checkpoints

`save_fusion_checkpoint` / `load_fusion_checkpoint`:
- Metadata: architecture ID (`medshield.diagnostic_net.gat_fusion`), version (`1.0.0`), model config, modality order, input dims, `trained` flag (default **false** for init), PyTorch version, encoder protocol versions.
- **Privacy Boundary**: Zero serialization of raw ECG signals, clinical notes, patient identifiers, or tabular demographics.
- Incompatible ID/version or missing/incompatible state dict keys → raises `FusionCheckpointError`.
- `strict=True` state dict loading enforced.

## Structural Readiness vs Scientific & Training Status

| Dimension | Status | Notes |
| :--- | :--- | :--- |
| **GAT Architecture Implementation** | **COMPLETE** | Model matches `MODEL_CONTRACT.md` §4 and ADR-009 |
| **GAT Structural Unit Tests** | **VERIFIED** | 15 test requirements covered in `tests/test_gnn_fusion.py` |
| **Encoder Interface Alignment** | **VERIFIED** | ECG (128d), Text (128d), Tabular (64d) consumed via Linear projections |
| **Checkpoint Save/Load Integrity** | **VERIFIED** | Typed `FusionCheckpointError`, metadata preserved, `trained=false` default |
| **Pretrained BioClinicalBERT Weights** | **UNAVAILABLE** | No cached weights; weights download prohibited in sandbox |
| **Paired Multimodal Training Cohort** | **UNAVAILABLE** | MIMIC-IV-ECG / MIMIC-IV / MIMIC-IV-Note absent; local cohorts independent |
| **GAT Supervised Training** | **DEFERRED / BLOCKED**| Blocked on acquisition of legitimate linked multimodal dataset |
| **Federated Learning Infrastructure** | **PENDING** | To be implemented in next phase via Flower (`flwr`) |
| **Production Diagnosis Inference** | **DISABLED** | `POST /api/v1/diagnosis/evaluate` returns HTTP 501 Not Implemented |


# ECG Research Baseline Protocol v1.0.0

**Status:** Frozen for Prompt 12  
**Dataset:** PTB-XL 1.0.3 (`data/raw/ptb-xl/`)  
**Metadata SHA-256:** `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`  
**Seed:** 42  

## Task definition (authoritative for this experiment)

This experiment follows **`docs/DATA_CONTRACT.md`** (NORM vs pathological MI/STTC), **not** the historical sinus-rhythm SNOMED proxy described in `README.md` / `ModelVersion.target_definition`. See **ADR-006** in `docs/DECISIONS.md`.

| Class | Label | Meaning |
|------:|------:|---------|
| 0 | Negative | Diagnostic superclass set is exactly `{NORM}` |
| 1 | Positive | `(MI ∈ supers) ∨ (STTC ∈ supers)` and `NORM ∉ supers` |

**Limitations:** SCP-ECG statement aggregation is a **research proxy**, not validated clinical heart-disease diagnosis. Sinus rhythm is **not** used as a negative class.

## SCP → superclass mapping

1. Parse `scp_codes` column with `ast.literal_eval` (per-record dict of SCP keys → confidence).
2. Join keys to `scp_statements.csv` where `diagnostic == 1.0`.
3. Collect unique `diagnostic_class` values → superclass set `S`.

## Exclusion rules (no forced binary for ambiguous rows)

| Code | Rule |
|------|------|
| `empty_superclass` | `S` is empty (no mappable diagnostic superclass) |
| `norm_pathology_conflict` | `NORM ∈ S` and (`MI ∈ S` or `STTC ∈ S`) |
| `non_binary_superclass` | Row not matching class 0 or class 1 rules (e.g. only `CD`/`HYP`, mixed pathology without NORM, etc.) |

Multi-label rows that satisfy class 0 or class 1 **without conflict** are included (rare for class 0; class 1 requires no NORM).

## Signal preprocessing

- **Source:** `filename_lr` only (native **100 Hz**, **10 s**, **12 leads**).
- **Loader:** WFDB `rdsamp`; reject non-finite values or shape ≠ `(1000, 12)`.
- **Lead order (model input channels):**  
  `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6` → tensor shape `(12, 1000)`.
- **Units:** millivolts (mV) from WFDB; stored as `float32`.
- **Normalization:** per-lead z-score using **training-split** mean/std only; applied frozen to val/test.

## Signal quality (Prompt 12A)

- **Module:** `ml/preprocessing/ecg_signal_quality.py` — `assess_ecg_signal()` after WFDB load in `load_ecg_mv()`.
- **Hard failures:** wrong shape (not 12×1000), non-finite values, empty/malformed arrays, lead count mismatch vs `LEAD_ORDER` length.
- **Heuristic (optional):** saturation if any |sample| > **100 mV** (default, configurable); flatline if **all** leads have std below threshold (**disabled** by default).
- **Research gate:** `ml/inference/ecg_quality_gate.py` — not exposed via API; on failure probability is `None` with messages *ECG quality insufficient* and *Prediction requires review/retry*.
- **Calibration:** `ml/evaluation/calibration.py` — Brier score and ECE (10 uniform bins) appended to evaluation reports; does not change val-AUROC checkpoint selection.

## Patient-independent splits (PTB-XL `strat_fold`)

Verified in Prompt 11B: **0** patients span multiple folds.

| Partition | `strat_fold` values | Use |
|-----------|---------------------|-----|
| Train | 1–7 | Fit normalization; model training |
| Validation | 8 | Checkpoint selection, early stopping |
| Test | 10 | **One-time** final metrics only |
| Unused | 9 | Held unused in v1 (keeps official fold 10 isolated) |

**Model selection:** best checkpoint by **validation AUROC** (primary). **Test fold 10 is never used for selection.**

## Training hyperparameters (defaults)

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4  
- Batch size: 32  
- Max epochs: 30  
- Early stopping: patience 5 on validation AUROC  
- Loss: `BCEWithLogitsLoss` (single logit binary head)  
- Encoder: `MODEL_CONTRACT.md` 1D CNN + BiLSTM → **128-d embedding** + linear classification head  

## Metrics

**Primary:** Test AUROC  
**Secondary:** AUPRC, F1, precision, recall (sensitivity), specificity, confusion matrix  

## Artifacts

Written only under `data/processed/ecg_baseline/` (gitignored):

- `manifests/split_v1.json`
- `checkpoints/`
- `logs/run_*.json`
- `reports/`

No patient identifiers in logs.

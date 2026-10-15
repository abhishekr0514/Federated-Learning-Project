# MedShield-FL Dataset Audit Report (Prompts 11A / 11B)

**Audit time (UTC):** 2026-10-15T10:26:52Z  
**Repository:** `C:/Federated-Learning-Project`  
**Datasets directory:** `data/datasets/` (flat; **0** subdirectories)  
**PTB-XL directory:** `data/raw/ptb-xl/`  
**Training conducted:** No  
**Raw datasets modified:** No  

## Executive summary

| Capability | Verdict | Evidence | Blockers | Next action |
|---|---|---|---|---|
| ECG-only research | **GO** | PTB-XL 1.0.3 @ `data/raw/ptb-xl/`; 21,799 waveforms; wfdb 96/96 @ 12×1000@100Hz | Proxy labels only | ECG encoder baseline (no training yet) |
| Tabular-only research | **CONDITIONAL_GO** | `cardio_train.csv` 70,000×13; balanced `cardio` label | BP/outlier rows; unverified license; survey label | Clean outliers; patient-level split by `id`; tabular baseline |
| Clinical-text research | **CONDITIONAL_GO** | NBME 42,146 notes; 14,300 span annotations | Phrase ID task; 5-row local `test.csv` | Split by `pn_num`; BioClinicalBERT span model |
| Multimodal GAT | **BLOCKED** | No shared patient keys across cohorts | No crosswalk; incompatible labels | Restricted linked dataset (e.g. MIMIC-IV-ECG + Note) or single-site data |
| Centralized evaluation | **CONDITIONAL_GO** | Single-modality datasets usable separately | Multimodal eval blocked | Run per-modality baselines first |
| Federated training | **CONDITIONAL_GO** | `cardio_train.csv` partitionable by synthetic clients | No real hospital/site IDs | Simulated FL only; document as simulation |
| Clinical diagnosis claim | **BLOCKED** | No validated multimodal GT | All modalities | Do not claim clinical diagnosis performance |

## Phase A — Inventory

**Files present (7):**

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `data/datasets/cardio_train.csv` | 2,941,524 | `21a705d23381b0df…` |
| `data/datasets/ecg-ptb-xl.ipynb` | 2,189,151 | `773f2b98dfa748d2…` |
| `data/datasets/features.csv` | 4,354 | `ba021dbf8b601d1e…` |
| `data/datasets/patient_notes.csv` | 34,966,100 | `b0cea588f3ca8610…` |
| `data/datasets/sample_submission.csv` | 93 | `06539347d4293242…` |
| `data/datasets/test.csv` | 141 | `212a67b8711a0e88…` |
| `data/datasets/train.csv` | 761,246 | `364a046f54189ae6…` |

**PTB-XL status:** None — PTB-XL 1.0.3 install complete (see `ptbxl_ecg_audit.json`).

**Duplicates by checksum:** 0 groups.

## Phase B — Dataset identities

### `cardio_train.csv`

- **Rows / columns:** 70,000 / 13 (`;` delimiter, UTF-8)
- **Label:** `cardio` — 35,021 class 0, 34,979 class 1 (near-balanced)
- **IDs:** 70,000 unique `id` values (assume one row per subject unless external docs say otherwise)
- **Missingness:** 0 empty cells across all columns (verified full pass)
- **Quality flags (full pass):** ap_lo>ap_hi: 1234; ap_hi out of 80–250: 247; ap_lo out of 40–150: 1034; height/weight outliers: 30/7
- **Task supported:** Binary cardiovascular survey outcome prediction (tabular encoder baseline)
- **Not supported without evidence:** Clinically confirmed heart-disease diagnosis claim

### NBME (`patient_notes.csv`, `features.csv`, `train.csv`, `test.csv`)

- **Notes:** 42,146 rows; unique `pn_num` per row; 10 `case_num` values
- **Features:** 143 concept definitions
- **Train annotations:** 14,300 rows; 1,000 unique notes annotated; all `pn_num` ∈ notes; all `feature_num` ∈ features
- **Target:** Character-span `location` for concept phrases (1,650 rows with multi-span `;` separators)
- **`test.csv`:** 5 rows — **sample/stub only**, not full competition hidden set
- **Task supported:** Clinical concept / phrase identification (BioClinicalBERT fine-tuning for span extraction)
- **Not supported:** Cardiovascular diagnosis label for MedShield multimodal pipeline

### PTB-XL

See **`PTB-XL_ECG_AUDIT_REPORT.md`** and `ptbxl_ecg_audit.json` for full Prompt 11B evidence.

## Phase F — Multimodal linkage

- **NBME internal:** Valid joins on `pn_num` / `feature_num` (within NBME only) — **GO**
- **PTB-XL ↔ cardio_train ↔ NBME:** No shared patient identifier, no crosswalk, incompatible labels — **BLOCKED**
- **GAT trainable today:** **No** — no genuine cross-modal patient correspondence

## Phase G — Splits & leakage

See `split_strategy.json`: patient/`id` split for cardio; `pn_num` grouping for NBME; PTB-XL split **blocked** until data present.

## Phase H — Sample sufficiency

- **Tabular:** ~70k rows adequate for binary classification experiment with CV; clean BP outliers first
- **NBME NLP:** ~14k annotated (note, feature) pairs; ~1k notes — adequate for phrase-ID prototyping, not for rare concept stability without careful splits
- **ECG:** Not assessable locally (data absent)

## Phase J — Federated learning

Real multi-hospital silos: **not available**. Simulated clients on `cardio_train.csv`: **feasible** if labeled simulation.

## Limitations

- PTB-XL signal integrity unverified (files missing)
- Cardio & NBME source licenses not stored in repo
- No model metrics computed in this audit

## Machine-readable outputs

- `dataset_inventory.json`
- `dataset_profiles.json`
- `linkage_audit.json`
- `split_strategy.json`
- `dataset_audit_summary.json`
- `dataset_recommendations.md`
- `ptbxl_ecg_audit.json`
- `PTB-XL_ECG_AUDIT_REPORT.md`

# Multimodal Data Linkage & Shared-Target Feasibility Audit (Prompt 16)

**Audit time (UTC):** 2026-10-15T15:00:00Z  
**Repository:** `C:/Federated-Learning-Project`  
**Mode:** Read-only — no joins, no training, no dataset modifications  
**Machine-readable:** `multimodal_linkage_audit.json`

---

## Executive summary

MedShield-FL **does not** currently have a legitimate local dataset linking **PTB-XL ECG**, **NBME clinical notes**, and **`cardio_train.csv` tabular** records to the same patient or encounter with a defensible **shared** supervision target. **Three-modality GAT training remains BLOCKED.**

The smallest concrete next step is **authorized acquisition** of a credentialed linked clinical corpus (documented candidate: **MIMIC-IV-ECG** + **MIMIC-IV** + **MIMIC-IV-Note**), followed by a **local install linkage audit** (subject/encounter keys, timestamps, label protocol) — not synthetic pairing of existing public files.

---

## 1. Linkage requirements (minimum evidence)

| Requirement | Purpose |
|-------------|---------|
| **Documented linkage key** | Stable `patient_id` and/or `encounter_id` (e.g. `hadm_id`, `ecg_id`) defined in source schema |
| **Provenance & license** | Source, version, DUA, permitted research use on file |
| **Temporal alignment** | When labels depend on co-occurring care, document observation windows |
| **Shared target protocol** | Single audited label definition with inclusion/exclusion rules |
| **Patient-independent splits** | Train/val/test grouped by subject (or encounter policy documented) |
| **Privacy boundary** | No raw PHI in central DB; hospital-local raw data per SECURITY_CONTRACT |

### Patient-level vs encounter-level

- **Patient-level:** multiple visits/ECGs/notes grouped under one subject — splits must not leak subjects across partitions.
- **Encounter-level:** one hospital stay or one ECG record — linkage must state whether labels apply to encounter or longitudinal patient.

### What is **not** evidence of identity

- Similar demographics or feature distributions  
- Compatible column counts (age, sex, BP)  
- Row counts or class balance similarity  
- **Numeric ID overlap** across unrelated datasets (see §4)  
- Repository colocation or notebook references  

---

## 2. Local datasets inspected

### PTB-XL 1.0.3 (`data/raw/ptb-xl/`)

| Item | Finding |
|------|---------|
| ECG records | 21,799 |
| Unique `patient_id` | 18,869 |
| Linkage keys | `ecg_id`, `patient_id`, `recording_date` |
| Text | `report` field on same row as ECG (21,797 non-empty) — **in-cohort ECG–report only** |
| Labels | `scp_codes` / superclass mapping (ECG interpretation proxy per DATA_CONTRACT) |
| SHA-256 `ptbxl_database.csv` | Matches Prompt 11B manifest (`7600de9c…`) |

PTB-XL does **not** include NBME notes or `cardio_train` survey rows. Metadata tabular fields (age, sex, height, weight) are **not** the versioned MedShield 10-feature schema tied to `cardio_train.csv`.

### `cardio_train.csv` (`data/datasets/`)

| Item | Finding |
|------|---------|
| Rows / unique `id` | 70,000 / 70,000 |
| Target | `cardio` — survey-style binary outcome (**not** verified clinical GT) |
| Linkage keys | `id` only — no ECG file reference, no note key |
| SHA-256 | Unchanged vs `dataset_inventory.json` |

### NBME (`patient_notes.csv`, `features.csv`, `train.csv`)

| Item | Finding |
|------|---------|
| Notes | 42,146; unique `pn_num` |
| Internal joins | `pn_num`, `feature_num`, `case_num` — **valid within NBME only** |
| Target | Phrase **span location** — not cardiovascular diagnosis |
| SHA-256 | Unchanged vs inventory |

### MIMIC (candidate, not local)

No paths under `data/raw/mimic-iv-ecg`, `data/raw/mimic-iv`, or `data/datasets/mimic`. Prior docs cite official PhysioNet pages only — **access not verified** for this workspace.

---

## 3. Cross-cohort linkage findings

| Pair | Verdict | Evidence |
|------|---------|----------|
| **ECG ↔ text** (PTB-XL vs NBME) | **BLOCKED** | Different populations, tasks, keys; no crosswalk |
| **ECG ↔ tabular** (PTB-XL vs cardio) | **BLOCKED** | No crosswalk; incompatible labels |
| **Text ↔ tabular** (NBME vs cardio) | **BLOCKED** | `pn_num` vs `id`; unrelated sources |

### Numeric ID overlap (important negative result)

A Prompt 16 check found **13,181** string-equal values between `cardio_train.id` and PTB-XL `patient_id` namespaces. This is **not** linkage evidence: cardio uses sparse integers in **0–99,999**; PTB patient IDs occupy a overlapping numeric range by **coincidence**. There is no shared provenance, encounter time, or crosswalk file. **Joining on equal integers would be scientific misconduct** per DATA_CONTRACT and ADR-004.

---

## 4. Candidate sources evaluation

| Candidate | Local | Modalities | Linkage | Shared target | Verdict |
|-----------|-------|------------|---------|---------------|---------|
| PTB-XL + NBME + cardio (local triad) | Partial | All three files exist **separately** | None verified | None compatible | **BLOCKED** |
| PTB-XL ECG + PTB `report` (internal) | Yes | ECG + short report text | `ecg_id` | ECG SCP / protocol v1 | **CONDITIONAL_GO** (two-modality **research**; not MedShield triad) |
| MIMIC-IV-ECG + MIMIC-IV + MIMIC-IV-Note | No | ECG, structured data, notes (if credentialed) | Documented in PhysioNet specs — **unverified locally** | Feasible only after DUA + label audit | **BLOCKED** locally; **CONDITIONAL_GO** post-credentialing |

---

## 5. Research paths compared

| Path | Description | Evidence required before implementation |
|------|-------------|----------------------------------------|
| **A (recommended now)** | Separate ECG, tabular, text research | Existing per-modality audits; no cross joins |
| **B** | Authorized linked hospital dataset | PhysioNet/single-site DUA, local install, linkage JSON, shared target ADR |
| **C** | Two-modality experiment | Verified join for **that pair only** + compatible label; still not three-modality GAT |

**Do not** synthesize paired patients or impute missing modalities for supervised fusion.

---

## 6. Decision matrix (Prompt 16 verdicts)

| Dimension | Verdict |
|-----------|---------|
| ECG–text linkage (cross local cohorts) | **BLOCKED** |
| ECG–tabular linkage (cross local cohorts) | **BLOCKED** |
| Text–tabular linkage (cross local cohorts) | **BLOCKED** |
| Shared-target validity (multimodal local) | **BLOCKED** |
| Patient-independent evaluation (per modality) | **CONDITIONAL_GO** |
| Two-modality research readiness | **CONDITIONAL_GO** (within-cohort / single-modality only) |
| Three-modality GAT training readiness | **BLOCKED** |

---

## 7. Privacy & scope confirmation

- No patient identifiers, raw notes, or waveforms printed in this report.  
- No joins executed; checksums for `data/datasets/*` and PTB-XL metadata **unchanged** vs prior inventory.  
- GAT architecture, training, FL, and `POST /api/v1/diagnosis/evaluate` — **unchanged** (still not implemented for inference).

---

## 8. Related artifacts

- Prior: `linkage_audit.json`, `DATASET_AUDIT_REPORT.md`, `ptbxl_ecg_audit.json`  
- Research: `docs/research/MULTIMODAL_GAT_ARCHITECTURE.md` (architecture only; training blocked)  
- Acquisition plan: `MULTIMODAL_DATASET_ACQUISITION_PLAN.md`

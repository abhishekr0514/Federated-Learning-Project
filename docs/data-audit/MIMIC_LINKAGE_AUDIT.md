# MIMIC Multimodal Linkage Verification (Prompt 16B)

**Audit time (UTC):** 2026-10-15T16:40:00Z  
**Mode:** Read-only — no downloads, no merged dataset, no training  
**Machine-readable:** `mimic_linkage_audit.json`  
**Auditor:** `ml/datasets/mimic_linkage_audit.py`

---

## Executive summary

An authorized local installation of **MIMIC-IV-ECG**, **MIMIC-IV**, and **MIMIC-IV-Note** was **not found** at documented repository paths. Local schema, referential integrity, and aggregate join counts were **not computed**.

**Three-modality GAT training remains BLOCKED.**

No other disks or unauthorized locations were searched. Data were not downloaded.

---

## 1. Datasets sought (documented paths only)

| Module | Documented roots | Present |
|--------|------------------|---------|
| MIMIC-IV-ECG v1.0 | `data/raw/mimic-iv-ecg`, `data/raw/mimic-iv-ecg/1.0` | **No** |
| MIMIC-IV v2.2 | `data/raw/mimic-iv`, `data/raw/mimiciv`, `data/raw/mimic-iv/2.2` | **No** |
| MIMIC-IV-Note v2.2 | `data/raw/mimic-iv-note`, `data/raw/mimic-iv-note/2.2` | **No** |

`data/raw/` currently contains **PTB-XL only** (plus `.gitkeep`). There is **no** env var or config path for MIMIC in `.env.example`.

### Missing modules (stop condition)

1. `mimic_iv_ecg`  
2. `mimic_iv`  
3. `mimic_iv_note`  

---

## 2. Provenance from official PhysioNet pages (not locally verified)

Fetched 2026-10-15 for key semantics only (restricted files were not accessed).

| Dataset | Version | License / DUA | Official page |
|---------|---------|---------------|---------------|
| MIMIC-IV-ECG | 1.0 | PhysioNet Credentialed Health Data License **1.5.0** + DUA 1.5.0; CITI training | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| MIMIC-IV | 2.2 | Same credentialed license/DUA | https://physionet.org/content/mimiciv/2.2/ |
| MIMIC-IV-Note | 2.2 | Same credentialed license/DUA | https://physionet.org/content/mimic-iv-note/2.2/ |

Local LICENSE/DUA files: **absent** (install missing). Workspace PhysioNet credential status: **unverified**.

### Expected metadata (when installed; not present)

**ECG:** `record_list.csv`, `machine_measurements.csv`, `waveform_note_links.csv`, WFDB `.hea`/`.dat` under `files/`.  
**Clinical:** `hosp/patients`, `hosp/admissions`, plus candidate tabular tables `omr`, `labevents`, `diagnoses_icd`; ICU `chartevents` / `icustays`.  
**Notes:** `discharge`, `discharge_detail`, `radiology`, `radiology_detail`.

---

## 3. Linkage key semantics (documentation-level)

Patient- vs encounter-level must not be conflated.

| Link | Documented key | Meaning | Local status |
|------|----------------|---------|--------------|
| **ECG → patient** | `subject_id` | Deidentified person ID; ECG `record_list.csv` maps waveform path + `study_id` → `subject_id`; MIMIC-IV `patients.subject_id` is the person table | **UNVERIFIED** (files absent) |
| **ECG → study** | `study_id` | One diagnostic ECG record | **UNVERIFIED** |
| **ECG → encounter** | **Not a native ECG `hadm_id`.** Official method: compare ECG time (`ecg_time` / WFDB `base_date`+`base_time`) with `admissions.admittime`/`dischtime` or ED stays for the **same** `subject_id` | **UNVERIFIED** |
| **ECG → note** | `waveform_note_links.csv.note_id` | Cardiologist ECG reports; `note_id` uses domain **EK**. MIMIC-IV-Note **v2.2** documents **discharge** and **radiology** tables. Official ECG page states EK free-text “will be made available as part of MIMIC-IV-Note **at a later time**.” | **UNVERIFIED**; possible **schema gap** even after install |
| **ECG → tabular** | `subject_id` plus optional temporal `hadm_id` window into `omr` / `labevents` / `chartevents` | **UNVERIFIED** |

Official ECG page (usage notes): machine clocks **may be unsynchronized** with EHR time; many ECGs are **outside** ED/ICU visits. Approximate published overlap: **~55%** hospital admission, **~25%** ED visit — **not recomputed here**.

### What was not done

- No join of PTB-XL, NBME, or `cardio_train.csv` to MIMIC.  
- Numeric ID coincidence is **not** identity (ADR-004 / ADR-010).  
- No merged patient-level table was written.

---

## 4. Shared-target feasibility

**Verdict: BLOCKED** — no local labels.

Documentation-only candidates (see `MIMIC_RESEARCH_TASK_OPTIONS.md`) are **not** approved targets:

- In-hospital mortality (`hospital_expire_flag`) — notes can leak the outcome.  
- ICD-coded MI/HF — billing codes, stay-level, not ECG ground truth.  
- ECG machine `report_*` lines — vendor interpretation, not clinician diagnosis.  
- Date of death (`patients.dod`) — censored ~1 year after last discharge.

None of these is a confirmed “heart disease diagnosis” equivalent to the MedShield product claim.

---

## 5. Patient-split feasibility

**BLOCKED** locally: no `subject_id` universe. When data exist, splits **must** be by `subject_id` (multiple `study_id` / `hadm_id` per person). Manifests should store hashed IDs only if checked into git.

---

## 6. Decision matrix

| Dimension | Verdict | Evidence |
|-----------|---------|----------|
| ECG-to-patient linkage | **BLOCKED** | ECG module not installed |
| ECG-to-note linkage | **BLOCKED** | ECG + Note absent; EK notes may not be in Note v2.2 |
| ECG-to-tabular linkage | **BLOCKED** | MIMIC-IV hosp/icu tables absent |
| Shared-target validity | **BLOCKED** | No audited label protocol |
| Temporal alignment | **BLOCKED** | No local timestamps; official clock-sync limitations |
| Leakage-safe patient splits | **BLOCKED** | No subject list |
| Two-modality research readiness | **BLOCKED** | Modules missing |
| Three-modality GAT training readiness | **BLOCKED** | Authorized linked cohort not available |

---

## 7. Privacy and scope

- No patient identifiers, notes, or waveforms printed.  
- Source datasets unmodified (none present to modify).  
- GAT architecture, FL, and `POST /api/v1/diagnosis/evaluate` unchanged.  
- Tests use **synthetic** IDs (`900001`…) only.

---

## 8. Next phase

Complete credentialing and install the three modules under the documented `data/raw/` paths (gitignored). Re-run this auditor to compute **aggregate** referential counts and timestamp overlap **without** writing a merged training table. Then draft a shared-target ADR **before** any GAT training.

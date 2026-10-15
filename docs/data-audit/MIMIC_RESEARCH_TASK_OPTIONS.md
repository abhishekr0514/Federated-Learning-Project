# MIMIC Research Task Options (Prompt 16B)

These options are **planning notes** from official PhysioNet documentation. They are **not** approved MedShield targets. Local MIMIC files are **absent**; no option is GO.

Do not treat ECG machine text, ICD codes, or mortality flags as confirmed heart-disease diagnosis.

---

## Option 0 — Do not train multimodal GAT (current)

**Status:** selected  
**Why:** MIMIC-IV-ECG, MIMIC-IV, and MIMIC-IV-Note are not installed. Continue Path A (PTB-XL ECG-only, cardio tabular-only, NBME span-only).

---

## Option 1 — ECG + structured data (two-modality), encounter-windowed

**Modalities:** MIMIC-IV-ECG waveforms + MIMIC-IV `omr` / `labevents` (and optionally ICU `chartevents`).  
**Linkage:** `subject_id`; encounter via ECG time inside `[admittime, dischtime]` (not a native ECG `hadm_id`).  
**Candidate labels (none approved):**

| Label | Source | Limitation |
|-------|--------|------------|
| In-hospital death | `admissions.hospital_expire_flag` | Outcome known at discharge; notes/labs may leak; not a diagnosis of CAD |
| Stay-level ICD (e.g. MI, HF) | `diagnoses_icd` | Billing/coding, stay-level, possible post-ECG assignment |
| Machine ECG statement | `machine_measurements.report_*` | Vendor algorithm; circular if used as both input and label |

**Temporal risk:** Official docs: unsynchronized ECG clocks; ~45% of ECGs may **not** overlap a hospital admission.  
**Readiness after install:** at best **CONDITIONAL_GO** pending overlap counts and a written inclusion window.  
**GAT three-node contract:** **not satisfied** (text node missing unless notes join).

---

## Option 2 — ECG + discharge summary (two-modality NLP)

**Modalities:** waveform + MIMIC-IV-Note `discharge.text` (deidentified).  
**Linkage:** same `subject_id` and temporal/`hadm_id` window. Direct `waveform_note_links.note_id` targets **EK** cardiologist reports, which Note **v2.2** does **not** list (discharge + radiology only). Official ECG page: EK notes “at a later time.”  
**Label risk:** Discharge summaries often **state** diagnoses and hospital course — high leakage if the text is both feature and label source.  
**Readiness:** **BLOCKED** until (a) local files exist and (b) EK vs discharge join rates are measured, and (c) a leakage protocol exists.

---

## Option 3 — Three-modality GAT (ECG + note + tabular)

**Requires:** all three modules; documented keys locally verified; a **single** shared target with a pre-ECG or otherwise non-leaky time origin; patient-level (`subject_id`) splits.  
**Current verdict:** **BLOCKED**.  
**Even after install, remaining gates:**

1. Referential integrity: ECG `subject_id` ⊆ `patients.subject_id`.  
2. Note coverage: fraction of studies with resolvable `note_id` in available note tables.  
3. Tabular coverage: observations inside a declared window **before** label time.  
4. Shared-target ADR signed; not using rhythm categories as “heart disease.”  
5. No merge committed to git; hospital-local storage only.

---

## Option 4 — Single-site / future MIMIC-IV-Ext

Only if a PhysioNet or IRB-linked extension provides **explicit** ECG–note–vitals keys and a protocolized label. Same audit must be repeated. Using “Ext” naming if publishing derivatives (PhysioNet MIMIC guidelines).

---

## Explicitly rejected shortcuts

- Pairing PTB-XL records with MIMIC notes by `patient_id` / `subject_id` numeric equality.  
- Using NBME exam notes as MIMIC text.  
- Using `cardio_train.cardio` as a MIMIC label.  
- Zero-filling missing modalities to force a 3-node graph sample.  
- Training GAT on synthetic paired embeddings as if they were patients.

---

## Recommended next research action

1. Credential and install the three MIMIC modules under documented `data/raw/` paths.  
2. Re-run `run_local_install_audit` / Prompt 16B for **aggregate** counts.  
3. If EK notes are still missing, consider **Option 1** (ECG+tabular) as the only two-modality path — still not production diagnosis.  
4. Do **not** open GAT training or diagnosis APIs until a shared-target ADR exists.

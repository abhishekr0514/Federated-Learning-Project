# PTB-XL ECG Audit Report (Prompt 11B)

**Generated:** 2026-10-15T10:26:43Z  
**Path:** `C:\Federated-Learning-Project\data\raw\ptb-xl`  
**Official source:** [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)  
**Version evidence:** 1.0.3  
**License:** LICENSE.txt (Creative Commons Attribution 4.0)

## Installation completeness — **GO**

| Check | Result |
|---|---|
| `ptbxl_database.csv` rows | 21799 |
| Missing LR waveforms | 0 |
| Missing HR waveforms | 0 |
| `records100` .hea / .dat | 21799 / 21799 |
| `records500` .hea / .dat | 21799 / 21799 |
| SHA256 manifest match (root files) | OK |

## Signal loading — **GO**

- **wfdb available:** True
- **Sample loads (records100):** 96/96 OK
- **Lead order:** `['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']`
- **Units:** mV (from wfdb meta on sample ecg_id=1)
- **NaN in sample:** 0

### 12 × 1000 @ 100 Hz — **GO**

- **filename_lr:** Already 12 leads × 1000 samples at 100 Hz — no resampling required.
- **filename_hr:** 500 Hz × 5000 samples — downsample by factor 5 (e.g. decimation or anti-alias filter) to 100 Hz × 1000.

## Labels and targets — **CONDITIONAL_GO**

**Diagnostic superclass counts (record-level, may be multi-label):**

{
  "NORM": 9514,
  "MI": 5469,
  "STTC": 5235,
  "HYP": 2649,
  "CD": 4898
}

- Records with >1 superclass: 5144
- Records with no mapped superclass: 411
- Records with `SR` in scp_codes: 16748

### MedShield sinus-rhythm proxy (repository contract — **not** clinical diagnosis)

Repository contract: normal sinus rhythm (SNOMED 426783006) vs pathological superclasses MI/STTC. Operational mapping for audit counts: SR present in scp_codes AND no MI/STTC superclass vs MI or STTC superclass present. NOT validated heart-disease diagnosis.

**Audit mapping counts:** {
  "proxy_positive_sr_without_mi_sttc": 10168,
  "proxy_negative_mi_or_sttc_superclass": 9365,
  "proxy_ambiguous_other": 2266
}

SNOMED 426783006 appears in application docs; PTB-XL uses SCP statement **SR** (rhythm=1.0 in `scp_statements.csv`).

## Patient-independent splits — **CONDITIONAL_GO**

- Unique `patient_id`: 18869
- ECG records: 21799
- Patients spanning multiple `strat_fold` values: 0
- Fold-10 vs other-folds patient overlap: 0

**Recommended:** {
  "primary": "Patient-level split by patient_id with stratification on aggregated labels (deterministic seed 42).",
  "official_baseline": "strat_fold != 10 train/val, strat_fold == 10 test (matches example_physionet.py) \u2014 document patient leakage.",
  "test_fold_for_model_selection": "Forbidden \u2014 hold out fold 10 (or patient test partition) for final evaluation only.",
  "manifest_path": "data/manifests/ptbxl_patient_split_seed42.json"
}

## Verdict summary

| Item | Verdict |
|---|---|
| Installation | GO |
| Signal loading | GO |
| 12×1000 preprocessing | GO |
| Target validity | CONDITIONAL_GO |
| Patient-independent eval | CONDITIONAL_GO |
| ECG-only research | GO |
| Multimodal GAT | BLOCKED |

**Training conducted:** No  
**Raw PTB-XL files modified:** No  

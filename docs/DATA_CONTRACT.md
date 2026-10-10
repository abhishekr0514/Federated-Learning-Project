# MedShield-FL Data Contract & Scientific Integrity Specification

## Data Governance & Privacy Boundaries

### Non-Negotiable Boundary Rules
1. **Hospital Boundary Isolation**: All raw patient records, raw ECG arrays, clinical notes, and PII must reside exclusively inside the client/hospital infrastructure.
2. **Central Database Prohibition**: The central PostgreSQL database MUST NEVER store raw clinical text, raw ECG signals, or direct patient identifiers.
3. **Transmission Policy**: Only approved model weight updates, differential privacy parameters, and aggregated evaluation metrics may be transmitted to the central server.

---

## Datasets Scope & Linkage Constraints

### Referenced Datasets
1. **PTB-XL ECG Dataset**:
   - 12-lead ECG records with clinical diagnostic labels and superclass metadata.
   - Purpose: ECG modality feature extraction and diagnostic pretraining/fine-tuning.
2. **Cardiovascular Disease Dataset (`cardio_train.csv`)**:
   - Tabular patient records containing 10 clinical parameters.
   - Purpose: Structured modality training for cardiovascular risk prediction.
3. **NBME Clinical Patient Notes (`patient_notes.csv`, `features.csv`, `train.csv`, `test.csv`, `sample_submission.csv`)**:
   - Clinical text notes and feature mapping.
   - Purpose: Clinical text representation learning with BioClinicalBERT.

### Scientific Integrity & Linkage Rules
- **Independent Datasets**: The referenced datasets originate from distinct patient populations and protocols. They MUST be treated as independent datasets unless verified patient-level alignment is documented with verifiable metadata.
- **Prohibition of Synthetic Joins**: Creating synthetic joins, randomly assigning patient IDs to pair ECGs with unrelated text notes, or fabricating multi-modal records across unrelated datasets is STRICTLY PROHIBITED.
- **Multimodal Evaluation Integrity**: If patient-linked multi-modal data is unavailable, individual modalities must be evaluated independently or on verified linked subsets without claiming fabricated multi-modal co-occurrence.

---

## Label Audit & Target Definition Contract

### Diagnostic Superclass Mapping (PTB-XL)
- PTB-XL diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`) do NOT automatically equate to a generic binary "heart disease" target.
- **Target Definition**:
  - **Positive Class (1)**: Pathological cardiac conditions (e.g., Myocardial Infarction `MI`, ST/T Change `STTC`).
  - **Negative Class (0)**: Normal ECG (`NORM`).
  - Uncertain or ambiguous secondary diagnoses must be explicitly handled via documented inclusion/exclusion filters.

### Patient Leakage & Data Splitting
1. **Patient-Independent Splitting**: Data splits (Train / Validation / Test) MUST be partitioned strictly by `patient_id`. Multiple recordings or notes from the same patient must never span across training and test sets.
2. **Preprocessing Isolation**: Normalization parameters, scalers, and tokenizers MUST be fit exclusively on training splits. Fitting preprocessing transformations on validation/test splits is prohibited.
3. **Feature Hygiene**: Subject identifiers, record IDs, hospital codes, and timestamps must be excluded from feature vectors to prevent shortcut learning.

---

## Metric Integrity Contract

- All reported metrics (Accuracy, ROC-AUC, PR-AUC, Sensitivity, Specificity, F1-Score) must be derived strictly from actual execution logs on independent test splits.
- Metric fabrication or reporting validation metrics as final test performance is forbidden.

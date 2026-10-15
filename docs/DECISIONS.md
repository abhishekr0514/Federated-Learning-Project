# MedShield-FL Architectural Decision Records (ADR)

## ADR-001: Sole Authoritative GitHub Repository

### Context & Problem
Multiple workspace environments or repository clones can introduce code fragmentation, diverged implementations, and lost development work across prompts.

### Decision
The single authoritative GitHub repository for all MedShield-FL source code, documentation, backend, frontend, models, migrations, configurations, and assets is strictly:
`https://github.com/abhishekr0514/Federated-Learning-Project`

All operations must execute inside the verified local clone located at `C:\Federated-Learning-Project`. No alternate repository or secondary directory shall be created or treated as the main project base.

### Status
**ACCEPTED**

---

## ADR-002: Mandatory Graph Attention Network (GAT) Multimodal Fusion Architecture

### Context & Problem
Multimodal fusion across 1D ECG signals, BioClinicalBERT text representations, and Neural Tabular Encoder features requires dynamic inter-modality attention to weigh diagnostic signals effectively.

### Decision
The core multimodal fusion model MUST be implemented using Graph Attention Networks (GAT), treating each modality representation (ECG 128d, Text 128d, Tabular 64d) as a graph node. Modality fusion via XGBoost, DenseNet, simple concatenation, or arbitrary MLPs is prohibited as the primary architecture and permitted only as documented baseline comparisons.

### Status
**ACCEPTED**

---

## ADR-003: Strict Hospital Boundary Isolation & Zero Central Raw Data Storage

### Context & Problem
Centralizing healthcare data creates severe HIPAA/GDPR privacy risks and single-point-of-failure vulnerabilities for patient confidentiality.

### Decision
All raw ECG waveforms, unstructured clinical text, patient medical records, and direct PII must remain permanently inside local hospital/client boundaries. The central PostgreSQL database shall store zero raw clinical text, zero raw signal vectors, and zero direct patient identifiers. Centralized FL components store only pseudonymous node IDs and aggregated model parameters.

### Status
**ACCEPTED**

---

## ADR-004: Scientific Data Integrity & Dataset Separation Policy

### Context & Problem
The project specification references PTB-XL, Cardiovascular `cardio_train.csv`, and NBME datasets, which originate from independent cohorts.

### Decision
Datasets must be treated as independent unless verified patient-level linkage metadata is documented. Fabricating synthetic patient joins, assigning random cross-dataset IDs, or inventing label targets is strictly forbidden. Data splitting must be strictly patient-independent, and target definitions must undergo scientific auditing prior to model training.

### Status
**ACCEPTED**

---

## ADR-005: Flower Framework for Federated Learning & Differential Privacy

### Context & Problem
Federated learning requires a standardized orchestration framework capable of managing asynchronous client updates, parameter aggregation, and differential privacy noise insertion.

### Decision
Flower (`flwr`) is selected as the standard federated learning framework. Client nodes implement differential privacy gradient clipping ($L_2$) and Gaussian noise addition prior to broadcasting updates to the central aggregator.

### Status
**ACCEPTED**

---

## ADR-006: Prompt 12 ECG Baseline Target vs Historical Sinus-Rhythm Proxy

### Context & Problem
`README.md` and `ModelVersion.target_definition` describe a **sinus rhythm (SNOMED 426783006) vs MI/STTC** proxy. Prompt 11B showed **2,266** ambiguous records under that mapping. `docs/DATA_CONTRACT.md` defines a separate binary task: **`NORM` vs (`MI` ∨ `STTC`)** with explicit exclusions.

### Decision
Prompt 12 ECG-only research baseline **v1.0.0** follows **`DATA_CONTRACT.md`** (protocol `docs/research/ECG_BASELINE_PROTOCOL_v1.md`). The application registry string remains unchanged until a dedicated model-version migration; new training artifacts record `protocol_version` and must not be reported as clinical heart-disease diagnosis.

### Status
**ACCEPTED** (Prompt 12)

---

## ADR-007: Prompt 13 Clinical Text Projection vs Full MODEL_CONTRACT Head

### Context & Problem
`MODEL_CONTRACT.md` specifies BioClinicalBERT → Linear(768,256) → GELU → Dropout → Linear(256,128) → LayerNorm. Prompt 13 requires CLS → Linear(768,128) for the research encoder.

### Decision
Implement **Linear(768,128)** in Prompt 13; document upgrade path to the full contract head before GAT fusion (`docs/research/CLINICAL_TEXT_ENCODER.md`).

### Status
**ACCEPTED** (Prompt 13)

---

## ADR-008: Prompt 14 Tabular Schema & Encoder vs MODEL_CONTRACT Block 2

### Context & Problem
`MODEL_CONTRACT.md` specifies a 10d tabular vector but does not name individual features. `docs/ARCHITECTURE.md` names ten clinical/survey fields including BMI. `cardio_train.csv` provides height/weight instead of BMI. Prompt 14 specifies a fixed MLP encoder (BatchNorm, Dropout 0.2, no residual) distinct from the contract’s residual + LayerNorm block.

### Decision
- Register schema **`medshield.tabular.cardio_survey` v1.0.0** with explicit feature order in `ml/tabular/feature_schema.py`, mapping ARCHITECTURE fields to cardio columns (BMI derived from height/weight).
- Implement Prompt 14 encoder architecture in `ml/models/tabular_encoder.py`; defer full contract block until pre-GAT integration.
- Keep `id` and `cardio` out of model inputs; no cross-dataset linkage (ADR-004).

### Status
**ACCEPTED** (Prompt 14)

---

## ADR-009: Prompt 15 GAT Layer Activation & Missing-Modality Policy

### Context & Problem
`MODEL_CONTRACT.md` specifies GAT dimensions and heads but not the nonlinearity applied between GAT layers or an approved missing-modality imputation strategy. Prompt 15 forbids silent zero imputation and unvalidated partial fusion.

### Decision
- Apply **ELU** to PyG `GATConv` outputs between the two contracted layers; GAT attention **dropout = 0.0**.
- **Full fusion forward** requires all three modalities available, quality-accepted, and finite; otherwise raise `FusionInputError`. Use `describe_modality_status()` for explicit incomplete reporting without GAT execution.
- Register architecture **`medshield.diagnostic_net.gat_fusion` v1.0.0** in `ml/models/gnn_fusion.py`. Checkpoints default `trained=false`.

### Status
**ACCEPTED** (Prompt 15)

---

## ADR-010: Prompt 16 Multimodal Linkage — No Local Cross-Cohort Joins

### Context & Problem
PTB-XL, `cardio_train.csv`, and NBME coexist locally but originate from independent populations with incompatible labels. Prompt 16 found large **numeric ID overlap** between `cardio_train.id` and PTB-XL `patient_id` without provenance crosswalk — a high-risk false-linkage pattern.

### Decision
- Treat all **cross-cohort** ECG–text, ECG–tabular, and text–tabular joins as **BLOCKED** until a credentialed linked dataset passes a documented install audit (`docs/data-audit/multimodal_linkage_audit.json`).
- **Numeric ID equality is not linkage evidence** (ADR-004 reinforcement).
- Continue **Path A** (separate modality research); smallest next step: PhysioNet credentialing for MIMIC-IV-ECG + MIMIC-IV + MIMIC-IV-Note per acquisition plan.
- **Three-modality GAT supervised training** remains blocked; architecture-only module unchanged.

### Status
**ACCEPTED** (Prompt 16)

---

## ADR-011: Prompt 16B — MIMIC Install Absent; GAT Training Remains Blocked

### Context & Problem
Prompt 16B requires a local audit of authorized MIMIC-IV-ECG, MIMIC-IV, and MIMIC-IV-Note. Documented roots (`data/raw/mimic-iv-ecg`, `data/raw/mimic-iv`, `data/raw/mimic-iv-note`) are empty. Official PhysioNet pages document `subject_id` / `study_id` / `waveform_note_links.note_id` and warn that ECG clocks may not align with EHR time; MIMIC-IV-Note v2.2 lists discharge and radiology tables, while EK cardiologist reports are described as forthcoming.

### Decision
- Do **not** search unauthorized paths or download credentialed data from automation.
- Treat all MIMIC linkage and shared-target verdicts as **BLOCKED** until a gitignored local install passes `ml/datasets/mimic_linkage_audit.py`.
- Do **not** join PTB-XL, NBME, or `cardio_train.csv` to MIMIC identifiers.
- Do **not** open three-modality GAT training, merged cohort construction, or production diagnosis endpoints.
- Documentation-level keys are recorded in `docs/data-audit/MIMIC_LINKAGE_AUDIT.md` as **unverified locally**.

### Status
**ACCEPTED** (Prompt 16B)

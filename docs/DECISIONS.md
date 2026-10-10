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

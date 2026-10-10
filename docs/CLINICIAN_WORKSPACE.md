# Clinician Workspace & Safe Clinical Shell Specification

**Status**: Authoritative Architectural Contract  
**Applies to**: `/clinician` Route Hierarchy, `/api/v1/clinician/*` Endpoints, and Clinician Security Context  
**Governing Prompts**: Prompt 00, Prompt 01, Prompt 02, Prompt 03, Prompt 04, Prompt 05, Prompt 07, Prompt 08  

---

## 1. Executive Summary & Purpose

The **Clinician Workspace** provides the clinical diagnostic interface for authorized clinicians within MedShield-FL.
Its primary objective is to enable safe clinical inquiry, document review, and patient diagnostic exploration while strictly safeguarding patient privacy and scientific integrity.

### Current Scientific & Implementation Status
The clinical artificial intelligence and federated learning pipelines are currently in an offline or experimental baseline state:
- **ECG Inference**: `NOT_IMPLEMENTED` (1D CNN-BiLSTM neural modules not exposed for runtime scoring)
- **BioClinicalBERT**: `NOT_IMPLEMENTED` (PII masking and text embedding neural networks offline)
- **Neural Tabular Encoder**: `NOT_IMPLEMENTED`
- **Graph Attention Network (GAT) Fusion**: `NOT_IMPLEMENTED`
- **Clinical Diagnosis Inference**: `NOT_ENABLED`
- **DiCE Counterfactual Explanations**: `NOT_IMPLEMENTED`
- **DoWhy Causal Inference & Interventions**: `NOT_IMPLEMENTED`
- **Flower Federated Training**: `NOT_ENABLED`

### Hard Architectural Rule: Zero Mock Clinical Data
Because the runtime inference subsystems are not implemented or enabled, **the clinician workspace must NEVER manufacture or simulate a diagnosis, risk score, confidence level, explanation, causal effect, or clinical recommendation.**

Whenever data or capabilities are absent, the application must explicitly halt and display the canonical operational state:
- `AVAILABLE`
- `NO_DATA`
- `NOT_IMPLEMENTED`
- `NOT_ENABLED`
- `FORBIDDEN`
- `ERROR`

---

## 2. Clinician Authorization & Scope Model

### 2.1 Role & Profile Requirements
1. **Role Enforcement**: Access to `/clinician` and `/api/v1/clinician/*` is restricted to authenticated users holding the `CLINICIAN` role (or `SUPER_ADMIN`).
2. **Active Clinician Profile**: The user must possess an active `ClinicianProfile` in the database linked to an active `Hospital` tenant.
3. **Role Segregation**: Users with only `PATIENT` or `HOSPITAL_ADMIN` roles without clinician privileges receive HTTP `403 Forbidden`.

### 2.2 Explicit Patient Assignment Scoping
1. **Assignment Boundary**: A clinician can ONLY access patient references for which an explicit `PatientClinicianAssignment` record exists in the database.
2. **Institutional Parity**: The assignment must match the clinician's authorized hospital (`assignment.hospital_id == clinician.hospital_id`).
3. **Unassigned / Cross-Hospital Patients**: Attempting to query an unassigned patient or a patient belonging to a different hospital returns HTTP `404 Not Found`. Returning `404` instead of `403` prevents cross-tenant and cross-clinician patient resource enumeration.
4. **No URL Tampering**: A clinician cannot broaden their access scope simply by changing a `patient_id` query or path parameter.

---

## 3. Privacy Boundary & Data Visibility Contract

```
┌────────────────────────────────────────────────────────┐
│              LOCAL HOSPITAL BOUNDARY                   │
│  - Raw Patient Records (Name, MRN, Address, Phone)     │
│  - Raw Unmasked Clinical Notes & Full-Text Reports     │
│  - Raw 12-Lead ECG Voltage Waveform Arrays (500Hz)     │
└──────────────────────────┬─────────────────────────────┘
                           │ (Never cross central boundary)
                           ▼
┌────────────────────────────────────────────────────────┐
│          CENTRAL DATABASE & OBSERVABILITY              │
│  - Opaque SHA-256 Hashes (opaque_patient_id)           │
│  - Document Metadata (type, status, version checksum)  │
│  - ECG Metadata (sampling_rate, lead_count, duration)  │
│  - Processing Job & Privacy Masking Metadata           │
│  - Model Architecture Registries & Version Tags        │
└────────────────────────────────────────────────────────┘
```

### 3.1 Prohibited Data in API & UI
The clinician API and frontend MUST NEVER serialize, transmit, or render:
- Patient Names (First name, Last name)
- Medical Record Numbers (MRN)
- Social Security Numbers (SSN) / Government Identifiers
- Contact Information (Address, Phone number, Email)
- Raw Unmasked Clinical Notes
- Raw ECG Signal Voltage Arrays / Float Matrices
- System Secrets (Passwords, Hashed passwords, JWT access/refresh tokens, API keys)

### 3.2 Permitted Patient Representation
- Opaque SHA-256 identifier (`opaque_patient_id`)
- Hospital node association reference
- Assignment timestamp
- Aggregated metadata counts (number of documents, number of ECG records)

### 3.3 Medical Documents Visibility
- Document UUID and type (`clinical_note`, `discharge_summary`, `lab_report`)
- Storage status (`pending`, `completed`, `failed`)
- Version numbers and SHA-256 checksums
- Privacy masking execution status (`sanitized`, `pending`, `failed`) and entities detected count
- **Raw document text is strictly excluded.**

### 3.4 ECG Telemetry Visibility
- Sampling frequency (`sampling_rate_hz`, e.g., 500 Hz)
- Recording duration (`duration_seconds`, e.g., 10.0 s)
- Lead count (`lead_count`, e.g., 12 leads)
- Dataset provenance (`dataset_source`, e.g., `PTB-XL-Proxy`)
- Acquisition timestamp
- **Raw ECG waveform voltage samples are strictly excluded.**

---

## 4. Clinician Workflow Hierarchy

The clinician workspace implements the following logical navigation hierarchy:

```
1. Patient Dossier Selection (Opaque Hash)
   │
   ├── 2. Available Clinical Metadata & Observations
   │
   ├── 3. Medical Documents (Metadata & Privacy Checksums)
   │
   ├── 4. ECG Telemetry (Acquisition Parameters; Zero Raw Arrays)
   │
   └── 5. AI Diagnostic Intelligence (Safely Halted)
         ├── Clinical AI Inference ──────► NOT_IMPLEMENTED / NO_DATA
         ├── Predicted Risk & Confidence ─► Displayed ONLY if genuine DB record exists
         ├── Why this prediction? (DiCE) ─► NOT_IMPLEMENTED
         ├── What could change? (CFs) ───► NOT_IMPLEMENTED
         ├── Causal Insights (DoWhy) ────► NOT_IMPLEMENTED
         ├── Decision Support ───────────► NOT_IMPLEMENTED (Disabled)
         ├── Model Information ──────────► Architecture metadata (Zero weights)
         └── Technical Details ──────────► Safe run metadata (Zero secrets)
```

---

## 5. Diagnostic Semantics & Provenance Rules

### 5.1 Prediction Confidence vs. Model Accuracy
- **Individual Prediction Confidence**: Represents the predicted class probability output by a neural classifier for an individual patient instance (e.g., $P(\text{Pathological} \mid x) = 0.88$).
- **Model Accuracy / Macro F1**: Represents an aggregate evaluation metric computed over a held-out test split of a dataset (e.g., PTB-XL test set).
- **CRITICAL RULE**: The UI must NEVER display an individual prediction confidence and label it as "Accuracy" (e.g., labeling an 88% confidence score as "88% Accuracy" is strictly prohibited).
- When a genuine prediction exists:
  - Display: `Model Confidence: <value>%`
  - Never display: `Accuracy: <value>%` for a single patient instance.

### 5.2 Absence of Predictions
- If no prediction record exists in `diagnosis_predictions`:
  - Status must be `NO_DATA`.
  - Capability must be `NOT_IMPLEMENTED`.
  - UI Text: *"Prediction unavailable — clinical inference is not implemented."*
  - Do NOT calculate a risk score in JavaScript.
  - Do NOT assign risk categories ("HIGH", "MEDIUM", "LOW") based on arbitrary formulas.
  - Do NOT display empty placeholders like `0.00%` or `0%` pretending to be calculated metrics.

### 5.3 Explainable AI (DiCE) Semantics
- Counterfactual explanation generation is `NOT_IMPLEMENTED`.
- The UI section *"Why this prediction?"* must explicitly display:
  *"Explanation generation is not yet available."*
- Fabricating feature importance bars, SHAP values, or artificial contributing findings is strictly prohibited.

### 5.4 Causal AI (DoWhy) Semantics
- Causal inference and DAG intervention analysis is `NOT_IMPLEMENTED`.
- The UI section *"Causal insights"* must explicitly state:
  *"Causal inference is not yet available."*
- Presenting correlational weights as causal treatments or generating fake intervention effects is strictly prohibited.

### 5.5 Clinical Decision Support Semantics
- Automated clinical decision support and treatment recommendations are `NOT_IMPLEMENTED`.
- The capability card must remain visibly disabled with the notice:
  *"Clinical decision support will become available after the validated inference and causal-analysis pipeline is implemented."*

---

## 6. Endpoints Specification

| Method | Route | Description | Auth / Scoping Guard |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/clinician/overview` | Clinician profile, assigned patient counts, subsystem status | `CLINICIAN` role + Active Profile |
| `GET` | `/api/v1/clinician/patients` | Paginated assigned patient references (`limit`, `offset`) | Scoped to `PatientClinicianAssignment` |
| `GET` | `/api/v1/clinician/patients/{id}` | Clinical dossier metadata for assigned patient | Returns `404` if unassigned / cross-tenant |
| `GET` | `/api/v1/clinician/documents` | Paginated documents for assigned patients | Scoped to assigned patient IDs |
| `GET` | `/api/v1/clinician/documents/{id}` | Detailed document metadata, checksums, masking jobs | Returns `404` if document not assigned |
| `GET` | `/api/v1/clinician/diagnosis` | Diagnosis prediction records for assigned patients | Scoped to assigned patient IDs |
| `GET` | `/api/v1/clinician/diagnosis/{id}`| Diagnosis prediction record for assigned patient | Returns `404` if unassigned; `NO_DATA` if empty |
| `GET` | `/api/v1/clinician/explanations/{id}`| DiCE explanations for genuine prediction | Returns `404` if unassigned; `NOT_IMPLEMENTED` |
| `GET` | `/api/v1/clinician/causal/{id}` | DoWhy causal analysis for genuine prediction | Returns `404` if unassigned; `NOT_IMPLEMENTED` |

---

## 7. Compliance Verification Checklist

- [x] Unauthenticated request returns HTTP `401`.
- [x] Non-clinician request (Patient / unprivileged Hospital Admin) returns HTTP `403`.
- [x] Unassigned patient query returns HTTP `404`.
- [x] Cross-hospital patient query returns HTTP `404`.
- [x] All patient directory endpoints employ server-side pagination.
- [x] Zero PII (name, email, phone, SSN, MRN) exposed in payloads.
- [x] Zero raw clinical notes or unmasked text exposed in payloads.
- [x] Zero raw ECG waveform sample arrays exposed in payloads.
- [x] Subsystems return canonical honest states (`NO_DATA`, `NOT_IMPLEMENTED`, `NOT_ENABLED`).
- [x] Zero simulated diagnoses, risk scores, or treatment recommendations rendered.

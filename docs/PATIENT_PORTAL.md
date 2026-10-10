# MedShield-FL — Patient Portal Specification & Implementation Contract

**Authoritative Reference**: Prompt 09 (Authoritative over Prompt 00–08 for Patient UI & Access Boundary)  
**Status**: ACTIVE / PRODUCTION CONTRACT  
**Scope**: Patient Portal (`/patient`), API (`/api/v1/patient/*`), Patient Ownership, Privacy & Presentation

---

## 1. Executive Summary

The **Patient Portal** is a privacy-safe, non-technical presentation layer designed exclusively for patients. It provides individuals with transparent, secure access to their own authorized health records, diagnostic test reports, and assessment results without exposing internal engineering internals, institutional administration data, or other patients' records.

The Patient Portal operates under two non-negotiable principles:
1. **Strict Patient Ownership Boundary**: A patient may access strictly and exclusively their own health information. Backend authorization derives patient identity from the authenticated session link (`User.id == Patient.user_id`). Route parameters (`patient_id`) are never trusted as proof of ownership; querying an out-of-scope patient reference returns HTTP `404 Not Found` to prevent resource enumeration.
2. **Zero Mock Data & Honest Clinical States**: Missing, unmeasured, or offline clinical and AI services are never coerced into numeric zeros, "Normal", or "Healthy". AI diagnostic capabilities (`GAT`, `DiCE`, `DoWhy`, `Flower`) currently remain offline and are explicitly surfaced as `NOT_ENABLED` or `NOT_IMPLEMENTED`.

---

## 2. Patient Ownership & Authorization Model

```
+-------------------------------------------------------------+
|                     Client Request                          |
|             Authorization: Bearer <JWT_TOKEN>               |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|              FastAPI Dependency: get_current_user           |
|                Validates JWT & Extracts User                |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|        FastAPI Dependency: get_current_patient_context      |
|  1. Checks UserRole.PATIENT in user.roles                   |
|     (Non-patient users receive HTTP 403 Forbidden)          |
|  2. Queries Patient WHERE Patient.user_id == user.id        |
|     (Unlinked accounts receive HTTP 404 Not Found)          |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|        FastAPI Dependency: get_authorized_patient_self      |
|  Enforces: route param patient_id == current_patient.id     |
|  - Matching ID: Returns Patient entity (HTTP 200)           |
|  - Mismatched or Cross-Tenant ID: HTTP 404 Not Found        |
+-------------------------------------------------------------+
```

### Authorization Rules:
- **Patient Self-Access**: Patient A querying Patient A records $\rightarrow$ HTTP `200 OK`.
- **Cross-Patient Access**: Patient A querying Patient B records $\rightarrow$ HTTP `404 Not Found`.
- **Arbitrary Identifier**: Patient A querying non-existent UUID $\rightarrow$ HTTP `404 Not Found`.
- **Cross-Hospital Isolation**: Patient A querying a patient from another hospital node $\rightarrow$ HTTP `404 Not Found`.
- **Role Parity**: Clinicians or Hospital Administrators accessing `/api/v1/patient/*` without an authorized `PATIENT` role receive HTTP `403 Forbidden`. Administrative users cannot inherit patient portal ownership context.

---

## 3. Privacy Boundaries & Data Scrubbing

The patient portal implements strict privacy filtering. Responses serialized across the network to patient clients must satisfy the following boundaries:

| Data Category | Patient Portal Permitted? | Serialization Policy |
| :--- | :--- | :--- |
| **Patient Identifiable Information (PII)** | **PROHIBITED** | Plaintext government IDs, Social Security Numbers, Medical Record Numbers (MRNs), and residential addresses are excluded. Patients are identified by opaque reference hashes. |
| **Raw Clinical Notes** | **PROHIBITED** | Unmasked, raw physician clinical notes are never exposed in patient endpoints. Only privacy-sanitized metadata and status are permitted. |
| **Raw ECG Waveforms** | **PROHIBITED** | 12-lead raw microvolt sample arrays (e.g., 5,000 voltage floats per lead) are never transmitted. Only acquisition parameters (sampling rate, duration, lead count, date) are visible. |
| **Privacy Processing Internals** | **PROHIBITED** | Masking character spans, NER token logits, and sanitization dictionary dumps are omitted. |
| **Security Secrets & Credentials** | **PROHIBITED** | JWT secrets, refresh tokens, database connection strings, and server hashes are excluded from all responses. |
| **Engineering Diagnostics** | **PROHIBITED** | Graph Attention Network (GAT) adjacency matrices, model gradients, embedding vectors, FL training rounds, and backend stack traces are strictly scrubbed. |

---

## 4. Current Scientific Status & Honest Operational States

MedShield-FL's multimodal AI pipeline is currently under development. The patient portal strictly reflects this ground truth:

| Subsystem | Operational State | Patient Presentation |
| :--- | :--- | :--- |
| **12-Lead ECG Feature Extraction** | `NOT_IMPLEMENTED` | Offline. Only test acquisition date and lead count are displayed. |
| **Clinical Text Processing** | `NOT_IMPLEMENTED` | Offline. Sanitized report metadata only. |
| **Tabular Clinical Encoder** | `NOT_IMPLEMENTED` | Offline. Vitals observation record metadata only. |
| **Multimodal GAT Fusion** | `NOT_IMPLEMENTED` | *"AI-assisted assessment is not currently available."* |
| **Clinical Diagnosis Inference** | `NOT_ENABLED` | Offline. No risk scores or diagnostic predictions manufactured. |
| **DiCE Counterfactual Explanations** | `NOT_IMPLEMENTED` | Offline. No simulated feature changes. |
| **DoWhy Causal AI DAG** | `NOT_IMPLEMENTED` | Offline. No treatment suggestions. |
| **Clinical Decision Support** | `NOT_IMPLEMENTED` | Disabled. |
| **Flower Federated Learning** | `NOT_ENABLED` | Background training node idle. |

### Operational State Canonical Vocabulary:
- `AVAILABLE`: Genuine records exist in database and are authorized for this patient.
- `NO_DATA`: No records exist in database for this patient query (e.g., *"No health records are currently available."*). Missing data is never coerced to "Healthy" or "0%".
- `NOT_ENABLED`: Subsystem is intentionally disabled in this deployment (e.g., *"AI-assisted assessment is not currently available."*).
- `NOT_IMPLEMENTED`: Feature is still in development (e.g., *"Online messaging is still being developed."*).
- `FORBIDDEN`: Request lacks valid patient authorization.
- `ERROR`: Backend operational error encountered.

---

## 5. Medical Safety Wording & Provenance Invariants

### Mandatory Safety Disclaimer
Whenever an actual AI-assisted assessment or diagnostic prediction is eventually surfaced to a patient, the system MUST display the following mandatory safety disclaimer:

> *"This result was generated with AI-assisted clinical decision support. It is intended to support, not replace, professional clinical judgment. Please discuss your result with your qualified healthcare professional."*

For the current deployment, because inference is offline, the portal displays:
> *"AI-assisted assessment is not currently available."*

### Prediction Confidence vs. Model Accuracy Separation
- **Model Confidence**: Refers to the model's calibrated probability output for an individual patient instance (e.g., *"Model confidence: 94.7%"*). May only be displayed if derived from an actual `diagnosis_predictions` record.
- **Model Accuracy**: Refers to aggregate validation metrics on held-out test datasets. **Model accuracy must NEVER be displayed as an individual patient's confidence score or diagnostic result.**

---

## 6. Information Architecture & Non-Technical Terminology

The Patient Portal uses plain, empathetic language understandable to non-technical individuals:

### Main Portal Sections:
1. **My Health**: High-level health profile showing available record counts, latest test dates, and assessment availability.
2. **My Records**: Overview of medical document metadata, laboratory/vitals observations, and ECG telemetry test dates.
3. **My Reports**: List of finalized diagnostic reports with privacy-sanitized metadata.
4. **My Assessment**: Future AI-assisted analysis section clearly communicating current `NOT_ENABLED` status.
5. **My Results**: Dedicated view for verified diagnosis predictions (when genuine records exist) or honest `NO_DATA`.
6. **Questions for My Clinician**: Guidance section offering suggested questions patients can ask their physician during appointments (online messaging placeholder is `NOT_IMPLEMENTED`).

### Terminology Translation Guide:

| Forbidden Technical Jargon | Approved Patient-Facing Plain Term |
| :--- | :--- |
| `GAT Multimodal Neural Fusion` | `AI-assisted assessment` |
| `Prediction Probability / Logits` | `Model confidence` |
| `Held-out Test Set ROC-AUC / Accuracy` | *Do not display to patient* |
| `BioClinicalBERT Token Sanitization` | `Privacy-protected health report` |
| `1D CNN-BiLSTM 500Hz Lead Array` | `12-Lead ECG recording` |
| `DiCE Perturbation Bounds` | `Explanation summary` |
| `DoWhy Backdoor Adjustment Estimator` | `Clinical insight` |
| `Flower Decentralized FedAvg Round` | *Do not display to patient* |

---

## 7. API Endpoint Specifications

All endpoints require `Authorization: Bearer <JWT_TOKEN>` with `PATIENT` role:

| HTTP Verb | Path | Description | Response Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/patient/overview` | High-level summary of patient's health records and AI capability states | 200 OK |
| `GET` | `/api/v1/patient/records` | List of patient's own clinical records and test metadata | 200 OK / NO_DATA |
| `GET` | `/api/v1/patient/records/{patient_id}` | Patient self-scoped records query (validates `patient_id == self`) | 200 OK / 404 |
| `GET` | `/api/v1/patient/reports` | List of patient's own diagnostic test reports | 200 OK / NO_DATA |
| `GET` | `/api/v1/patient/diagnosis` | Current assessment status and genuine prediction (if present) | 200 OK / NO_DATA |
| `GET` | `/api/v1/patient/diagnosis/{patient_id}` | Patient self-scoped diagnosis query (validates `patient_id == self`) | 200 OK / 404 |

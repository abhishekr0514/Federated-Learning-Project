# MedShield-FL Hospital Admin Dashboard & Observability Specification

## 1. Overview & Architectural Boundaries

The Hospital Admin Command Center (`/hospital`) provides institutional-level observability, operational oversight, and local compliance monitoring for hospital administrators.

### Core Architectural Principles
1. **Strict Tenant Isolation**: An administrator may only access operational metadata for hospitals where they possess an explicit, authorized `HospitalMembership`. All repository queries enforce the authorized hospital scope server-side.
2. **Zero Mock Data Contract**: Synthetic metrics, simulated loss curves, fake validation accuracies, and placeholder patient rows are strictly prohibited. Missing records are never coerced to zero.
3. **Explicit Operational States**: Every subsystem state is explicitly typed and surfaced directly to the API consumer and frontend UI.
4. **Clinical & Privacy Protection**: Zero Protected Health Information (PHI) or Personally Identifiable Information (PII) is exposed centrally. Opaque patient references (`opaque_patient_id`) and append-only event ledgers are used exclusively.

---

## 2. Canonical Operational States

All observability indicators and endpoint responses must distinguish between the following operational states:

| State Code | Semantic Definition | UI Rendering Behaviour |
| :--- | :--- | :--- |
| `AVAILABLE` | The subsystem is operational and valid database records or real-time infrastructure measurements exist. | Render numeric metrics, tables, and details. |
| `NO_DATA` | The subsystem is operational, but currently zero database records exist for this hospital node. | Display explicit empty state ("No records available"). Never display "0" unless 0 is mathematically verified. |
| `NOT_ENABLED` | The subsystem is intentionally disabled in this deployment (e.g. live Flower federated training, local training engine, automatic alert generation). | Display disabled badge ("Subsystem is not enabled"). |
| `NOT_MEASURED` | The metric is unmeasured by the current backend configuration (e.g. CPU, RAM, Disk, GPU usage). | Display unmeasured tag ("Not Measured"). |
| `NOT_IMPLEMENTED` | The endpoint or feature capability is intentionally deferred/unimplemented (e.g. queue depth, inference latency). | Display unsupported tag ("Not Implemented"). |
| `ERROR` | An exception occurred during query or infrastructure check. | Display error alert with correlation ID. |

---

## 3. Server-Side Tenant Scoping Model

### Authorization & Scope Derivation
The server derives the authorized hospital context from the authenticated actor's verified credentials:
- **`require_role(UserRole.HOSPITAL_ADMIN)`**: Ensures caller is an active Hospital Admin (or Super Admin).
- **Single-Hospital Admin**: If the administrator is associated with exactly one hospital membership, that hospital is automatically bound to the request context.
- **Multi-Hospital Admin**: If the administrator holds memberships across multiple hospitals, the client may pass a `hospital_id` parameter. The backend strictly validates that `hospital_id` against the caller's active memberships.
- **Tenant Boundary Violation**: If an administrator attempts to query an unauthorized hospital ID, the API returns **`404 Not Found`** to prevent cross-tenant resource enumeration.

---

## 4. API Endpoints & Provenance Specification

All endpoints reside under the `/api/v1/hospital` namespace and require `HOSPITAL_ADMIN` role and relevant scoped permissions.

### 4.1 Overview (`GET /api/v1/hospital/overview`)
- **Provenance**: Server-side aggregations over `hospitals`, `hospital_memberships`, `patient_hospital_associations`, `federated_clients`, `model_versions`, `hospital_metrics`, `audit_logs`, and `alerts`.
- **Fields**:
  - `hospital_id`, `name`, `code`, `is_active`
  - `clinicians`: Count of authorized clinician profiles or `NO_DATA`
  - `patients`: Count of opaque patient reference hashes or `NO_DATA`
  - `federated_clients`: Registered local client count or `NO_DATA`
  - `latest_model_version`: Latest global model version tag or `NO_DATA`
  - `latest_metric`: Most recent hospital evaluation metric metadata or `NO_DATA`
  - `latest_audit_timestamp`: Timestamp of most recent hospital audit event or `NO_DATA`
  - `latest_alert_timestamp`: Timestamp of most recent hospital alert or `NO_DATA`

### 4.2 System Health (`GET /api/v1/hospital/system-health`)
- **API Status**: `AVAILABLE`
- **Database Status**: `AVAILABLE` with measured round-trip latency in milliseconds (`SELECT 1`), or `ERROR`.
- **Hardware Telemetry**: `cpu_usage: NOT_MEASURED`, `memory_usage: NOT_MEASURED`, `disk_usage: NOT_MEASURED`, `worker_health: NOT_IMPLEMENTED`, `queue_depth: NOT_IMPLEMENTED`.

### 4.3 Clinicians (`GET /api/v1/hospital/clinicians`)
- **Provenance**: `clinician_profiles` and `hospital_memberships` scoped to `hospital_id`.
- **Fields**: `id`, `user_id`, `department`, `license_number_hash`, `user_status`, `created_at`.
- **Privacy Boundary**: Zero exposure of passwords, hashed passwords, JWTs, refresh tokens, or clinical access secrets.
- **Pagination**: Server-side `limit` and `offset`.

### 4.4 Patients (`GET /api/v1/hospital/patients`)
- **Provenance**: `patient_hospital_associations` joined with `patients` scoped to `hospital_id`.
- **Fields**: `patient_id` (UUID), `opaque_patient_id` (SHA-256 hash), `created_at`.
- **Privacy Boundary**: Zero exposure of patient names, phone numbers, email addresses, government IDs, medical records numbers (MRNs), raw clinical notes, or ECG signal waveforms.
- **Pagination**: Server-side `limit` and `offset`.

### 4.5 Models (`GET /api/v1/hospital/models`)
- **Provenance**: `model_versions` and `hospital_metrics` for this hospital node.
- **Fields**: `version_tag`, `status`, `target_definition`, `created_at`.
- **Honest State**: If no model records exist, returns `status: "NO_DATA"`.

### 4.6 Metrics (`GET /api/v1/hospital/metrics`)
- **Provenance**: `hospital_metrics` where `hospital_id == authorized_hospital_id`.
- **Scientific Integrity Mandate**:
  - The historical PTB-XL proxy experiment (`Normal Sinus Rhythm vs Pathological Superclasses`) is an offline research baseline, NOT a clinical diagnosis model.
  - Every metric record must display its exact `target_definition` alongside metric values.
  - If no metrics are stored, returns `status: "NO_DATA"`.

### 4.7 Federated Learning (`GET /api/v1/hospital/federated`)
- **Provenance**: `federated_clients` and `federated_client_updates` scoped to `hospital_id`.
- **Status**: `NOT_ENABLED` (Live Flower aggregation engine is disabled).
- **Historical Records**: Any persisted client registrations or previous updates are explicitly tagged as `"HISTORICAL"`.

### 4.8 Privacy Masking (`GET /api/v1/hospital/privacy`)
- **Provenance**: `privacy_masking_results` joined with `medical_documents` where `medical_documents.hospital_id == authorized_hospital_id`.
- **Fields**: Aggregate sanitized document count, entities detected count, sanitization status, timestamps.
- **Privacy Boundary**: Zero exposure of raw note text, PII entity spans, or replacement maps.
- **Honest State**: If no masking events exist, returns `status: "NO_DATA"`.

### 4.9 Data Quality (`GET /api/v1/hospital/data-quality`)
- **Provenance**: `data_quality_events` where `hospital_id == authorized_hospital_id`.
- **Fields**: `id`, `severity`, `status`, `category`, `created_at`, `resolved_at`.
- **Privacy Boundary**: Zero clinical text in event payloads.
- **Honest State**: If no events exist, returns `status: "NO_DATA"`.

### 4.10 Node Audit Log (`GET /api/v1/hospital/audit`)
- **Provenance**: Append-only `audit_logs` where `hospital_id == authorized_hospital_id`.
- **Fields**: `id`, `event_type`, `resource_type`, `outcome`, `request_id`, `created_at`.
- **Immutability & Safety**: No update/delete operations exist. Zero tokens, passwords, or PHI payloads serialized.
- **Pagination**: Server-side `limit` and `offset`.

### 4.11 Node Alerts (`GET /api/v1/hospital/alerts`)
- **Provenance**: `alerts` where `hospital_id == authorized_hospital_id`.
- **Fields**: `id`, `severity`, `status`, `category`, `title`, `created_at`, `resolved_at`.
- **Subsystem State**: `automatic_generation: "NOT_ENABLED"`.
- **Honest State**: Returns `status: "NO_DATA"` when zero records exist.

---

## 5. Summary of What is Intentionally NOT Implemented

1. **Flower Federated Training**: Flower clients and central federated server are not enabled.
2. **Local Weight Training**: Local PyTorch/Flower training routines are not executed.
3. **Clinical Diagnostics**: GAT multimodal inference, ECG prediction, and BERT diagnosis are disabled.
4. **Counterfactual & Causal Explanations**: DiCE counterfactuals and DoWhy DAG causal inference are not computed.
5. **Cross-Hospital Aggregation**: No hospital administrator can view metrics, patients, or logs across other institutions.

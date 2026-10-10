# MedShield-FL Database Contract & Privacy Boundary Specification

## Overview

This document specifies the PostgreSQL domain schema, privacy boundaries, entity relationships, indexing strategies, and migration controls for the **MedShield-FL** platform.

---

## 1. Central vs. Local Privacy Boundary Matrix

> **AUTHORITATIVE ARCHITECTURAL PRINCIPLE**:
> The central PostgreSQL database is **NOT** a repository for raw patient data, raw ECG signal waveforms, or unmasked clinical text reports. Raw clinical data remains isolated strictly inside hospital/client boundaries.

| Data Category | Central PostgreSQL Storage | Hospital-Local Node Storage | Privacy Rationale |
| :--- | :--- | :--- | :--- |
| **Patient Identity** | Opaque UUID & Reference Hash (`opaque_patient_id`) | Local MRN, Name, SSN, DOB, Address, Phone | Prevents central PII aggregation |
| **ECG Signal Data** | Acquisition metadata, duration, local URI (`local_storage_ref`) | 12-lead raw waveform arrays / HDF5 / WFDB | Protects raw biometrics |
| **Clinical Reports** | Document URI, document type, processing status | Unmasked raw clinical report text / PDFs | Protects unmasked PHI |
| **Privacy Sanitization** | Sanitization status, entity count, masking method | Sanitized text cache & local entity mapping | Verifies sanitization telemetry |
| **Federated Updates** | Round metadata, sample count, weight storage URI | Local dataset, gradients, raw mini-batches | Preserves federated privacy |
| **Audit Logs** | Event type, actor ID, request ID, IP, status | Detailed local audit trail | Excludes secrets & PHI from logs |

---

## 2. Entity Ownership & Tenant Isolation

Every clinical reference (`MedicalDocument`, `ECGRecord`, `ClinicalObservation`, `DiagnosisPrediction`) is explicitly tied to a `hospital_id` FK.

```
Hospital (Institution Boundary)
├── Memberships (Hospital Admin, Clinician)
├── Clinician Profiles
├── Patient Associations (Hospital-Scoped Patient Hashes)
├── Document Metadata References
├── ECG Record Metadata References
└── Federated Client Nodes
```

---

## 3. Entity Relationship Diagram (Text Schema)

```
[Users] ◄─── (Many-to-Many via user_roles) ───► [Roles] ◄─── (Many-to-Many) ───► [Permissions]
   │
   ├──► [HospitalMemberships] ◄─── [Hospitals]
   ├──► [ClinicianProfiles]    ◄───┘
   │                                 │
   │                                 ├──► [PatientHospitalAssociations] ◄─── [Patients]
   │                                 │                                          │
   │                                 ├──► [MedicalDocuments] ◄──────────────────┤
   │                                 │       ├──► [DocumentVersions]            │
   │                                 │       ├──► [DocumentProcessingJobs]      │
   │                                 │       └──► [PrivacyMaskingResults]       │
   │                                 │                                          │
   │                                 ├──► [ECGRecords] ◄────────────────────────┤
   │                                 ├──► [ClinicalObservations] ◄──────────────┤
   │                                 │                                          │
   │                                 └──► [DiagnosisPredictions] ◄──────────────┘
   │                                         ├──► [PredictionExplanations]
   │                                         │       └──► [CounterfactualExplanations]
   │                                         └──► [CausalAnalyses]
   │
   └──► [AuditLogs]
```

---

## 4. Key Schema Strategies

### Identifiers
- **Primary Keys**: All tables use 128-bit UUIDv4 (`postgresql.UUID(as_uuid=True)`).
- **Public API References**: Exposed identifiers are exclusively UUID strings; sequential auto-incrementing integers are prohibited.

### Timestamps
- All timestamp columns use **UTC Timezone-Aware** types (`DateTime(timezone=True)`).
- Lifecycle tracking: `created_at` (default UTC now), `updated_at` (onupdate UTC now).

### Soft Deletion
- **Soft Delete Applied**: `users.deleted_at` (allows account deactivation without breaking historical references).
- **Hard Delete / Cascade**: Parent-child document versions, masking results, and client updates cascade on delete.
- **Immutable Entities**: `audit_logs` and `security_events` are append-only; soft deletion is disabled.

### Indexing Strategy
- Single-column indexes on high-cardinality lookups: `users.email`, `roles.name`, `hospitals.code`, `patients.opaque_patient_id`, `model_versions.version_tag`, `federated_rounds.round_number`.
- Foreign key indexes: `hospital_memberships.user_id`, `hospital_memberships.hospital_id`, `medical_documents.patient_id`, `ecg_records.patient_id`, `diagnosis_predictions.patient_id`, `audit_logs.actor_id`.
- Composite unique indexes: `(user_id, hospital_id)` on `hospital_memberships`, `(patient_id, hospital_id)` on `patient_hospital_associations`, `(round_id, client_id)` on `federated_client_updates`.

---

## 5. Unresolved Schema Decisions

1. **Local Weight Storage S3/GCS URI Schema**: Model update URI formats will be locked when Flower server aggregation is implemented.
2. **Tabular Feature Parquet Parity**: Tabular schema versioning (`ClinicalObservation.feature_schema_version`) will be extended upon dataset loading implementation.

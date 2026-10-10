# MedShield-FL Super Admin Command Center Contract

## Overview

The Super Admin Command Center provides global system observability, institutional node oversight, security monitoring, and audit log inspection for the MedShield-FL platform.

---

## 1. Zero Mock Data & State Categorization Semantics

Every displayed metric originates from an actual database query, server execution status, or infrastructure check. The Command Center strictly avoids mock values (such as fake 95% accuracy scores or synthetic federated round numbers).

All metric indicators are categorized into four explicit states:

| State Code | UI Display Tag | Semantic Meaning |
| :--- | :--- | :--- |
| `AVAILABLE` | **Active / Measured** | Valid database records or system measurements exist and are actively returned. |
| `NO_DATA` | **No Records Recorded** | The backend subsystem is operational, but currently 0 database records exist. |
| `NOT_ENABLED` | **Not Enabled** | The feature/subsystem is intentionally disabled (e.g. Flower federated training engine). |
| `NOT_MEASURED` | **Not Measured** | The infrastructure measurement (e.g. CPU/Memory usage) is unmeasured by current backend code. |

---

## 2. API Endpoint & Metric Provenance Specification

All endpoints are prefixed under `/api/v1/admin` and require `SUPER_ADMIN` role and `global.admin` permission.

### Endpoint Matrix

#### `GET /api/v1/admin/overview`
- **Purpose**: System-wide summary cards for the main Command Center dashboard.
- **Source Tables**: `hospitals`, `users`, `security_events`, `audit_logs`.
- **Calculations**: `COUNT(hospitals.id)`, `COUNT(users.id WHERE is_active=True)`, `COUNT(security_events.id WHERE event_type='LOGIN_SUCCESS')`, `COUNT(security_events.id WHERE event_type='LOGIN_FAILED')`.
- **Freshness**: Real-time server-side query.

#### `GET /api/v1/admin/system-health`
- **Purpose**: Real API connectivity, DB latency check, active user sessions, and hardware measurement flags.
- **Source**: FastAPI engine, SQLAlchemy `SELECT 1` ping, `user_sessions`.
- **Calculations**: Database response time in milliseconds. Hardware metrics (CPU/Memory/Disk) return `status: "NOT_MEASURED"`.
- **Freshness**: Real-time.

#### `GET /api/v1/admin/hospitals`
- **Purpose**: List of institutional nodes registered in PostgreSQL.
- **Source Tables**: `hospitals`, `hospital_memberships`.
- **Calculations**: SQL Join counting active members per hospital.
- **Freshness**: Real-time.

#### `GET /api/v1/admin/federated`
- **Purpose**: Federated training status overview.
- **Response**: `{ status: "NOT_ENABLED", message: "Federated training is not enabled.", data: { current_round: null, active_clients: 0 } }`.
- **Freshness**: Immutable contract state.

#### `GET /api/v1/admin/models`
- **Purpose**: Model version artifacts registry.
- **Source Table**: `model_versions`.
- **Freshness**: Real-time. (Returns `status: "NO_DATA"` when 0 records exist).

#### `GET /api/v1/admin/metrics`
- **Purpose**: Model performance metrics (accuracy, ROC-AUC, sensitivity).
- **Source Table**: `model_metrics`.
- **Freshness**: Real-time. (Returns `status: "NO_DATA"` when 0 evaluation records exist).

#### `GET /api/v1/admin/privacy`
- **Purpose**: PII/PHI privacy masking result statistics.
- **Source Table**: `privacy_masking_results`.
- **Freshness**: Real-time. (Returns `status: "NO_DATA"` when 0 records exist).

#### `GET /api/v1/admin/security`
- **Purpose**: Aggregated security event counts.
- **Source Table**: `security_events`.
- **Calculations**: Server-side SQL grouping by `event_type`.
- **Freshness**: Real-time.

#### `GET /api/v1/admin/audit`
- **Purpose**: Paginated append-only security audit log ledger.
- **Source Table**: `audit_logs`.
- **Calculations**: Server-side pagination with `limit` and `offset`.
- **Freshness**: Real-time.

#### `GET /api/v1/admin/alerts`
- **Purpose**: Active system alerts.
- **Source Table**: `system_metrics` / `alerts`.
- **Freshness**: Real-time. (Returns `status: "NO_DATA"` when 0 active alerts exist).

---

## 3. Security & Privacy Boundaries

- **Access Guard**: Requests from roles other than `SUPER_ADMIN` (`HOSPITAL_ADMIN`, `CLINICIAN`, `PATIENT`) return HTTP `403 Forbidden`. Unauthenticated requests return `401 Unauthorized`.
- **No Sensitive Credential Serializing**: API responses never contain JWT secrets, password hashes, refresh token hashes, database credentials, PII/PHI patient data, or unencrypted ECG telemetry.

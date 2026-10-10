# MedShield-FL Authentication, RBAC, and Tenant Authorization Contract

## Overview

This contract documents the JWT authentication strategy, token rotation policy, session revocation mechanisms, canonical role-based access control (RBAC), permission capabilities, and multi-tenant isolation rules for **MedShield-FL**.

---

## 1. Token Lifetimes & Cryptographic Signing

| Property | Configuration Setting | Default Value | Claims Included |
| :--- | :--- | :--- | :--- |
| **Access Token** | `ACCESS_TOKEN_EXPIRE_MINUTES` | 15 minutes | `sub` (User ID), `jti` (UUID), `iat`, `exp`, `iss`, `type: "access"` |
| **Refresh Token** | `REFRESH_TOKEN_EXPIRE_DAYS` | 7 days | `sub` (User ID), `jti` (UUID), `iat`, `exp`, `iss`, `type: "refresh"` |
| **Algorithm** | `JWT_ALGORITHM` | `HS256` | Signed via `JWT_SECRET` |

> **SECURITY INVARIANT**: JWT tokens contain strictly **ZERO** PHI, PII, raw clinical notes, or ECG signal values.

---

## 2. Refresh Sessions, Token Rotation & Compromise Protection

### Database Storage
Raw refresh tokens are **NEVER** stored in PostgreSQL. The `user_sessions` table stores SHA-256 hashes of token secrets alongside `token_jti` references and revocation status flags.

### Token Rotation
Each refresh request (`POST /api/v1/auth/refresh`) enforces strict single-use token rotation:
1. Validates the presented refresh token JWT signature and expiration.
2. Checks the `user_sessions` record corresponding to `token_jti`.
3. Marks the old session as `is_revoked = True`.
4. Creates a new `UserSession` entry with a fresh `jti` and issues a new access and refresh token pair.

### Reuse Detection & Compromise Mitigation
If a revoked refresh token is presented:
1. A `CRITICAL` severity `SecurityEvent` (`REFRESH_REUSE_DETECTED`) is immediately recorded.
2. **ALL** active sessions for the associated user account are forcefully revoked (`is_revoked = True`).
3. Request is rejected with `400 Bad Request / 401 Unauthorized`.

---

## 3. Canonical Roles & Permission Mapping Matrix

Canonical role names:
- `SUPER_ADMIN`
- `HOSPITAL_ADMIN`
- `CLINICIAN`
- `PATIENT`

```
System Permissions:
- global:admin
- hospital:manage
- federated:monitor
- global_model:manage
- security:monitor
- audit:read
- global_metrics:read
- hospital:config
- clinician:manage
- local_patient:manage
- local_training:monitor
- local_metrics:read
- local_audit:read
- clinical:read
- clinical:write
- diagnosis:read
- explanation:read
- causal:read
- own_profile:read
- own_records:read
- own_diagnosis:read
```

---

## 4. Tenant & Patient Ownership Isolation

1. **Hospital Tenant Scoping**:
   - `HospitalAdmin` actors are constrained to institutions where an active `HospitalMembership` exists.
   - Cross-hospital attempts return `404 Not Found` to prevent revealing resource existence.

2. **Patient Ownership Scoping**:
   - `PATIENT` actors can access only resources where `patient.user_id == current_user.id`.
   - `CLINICIAN` actors can access patient references assigned via `PatientClinicianAssignment` or belonging to their affiliated hospital node.

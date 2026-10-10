# MedShield-FL Frontend Security Architecture Contract

## Overview

This contract establishes the security boundaries, session management strategy, and role-based UI architecture for the MedShield-FL web application.

---

## 1. Backend Authority Security Model

- **Authoritative Security Boundary**: All authorization decisions are strictly enforced by the backend FastAPI API guards (`get_current_user`, `require_role`, `require_permission`, `verify_hospital_access`, `verify_patient_access`).
- **UI Route Protection**: Frontend route guards (`ProtectedRoute.jsx`) and permission-aware navigation components serve exclusively to improve user experience (UX) and guide valid user navigation.
- **Manual Navigation Safety**: If an unprivileged user manually enters a protected URL (e.g., `/hospital` or `/admin`), the client API calls to backend endpoints will return `401 Unauthorized`, `403 Forbidden`, or `404 Not Found`, ensuring zero data leakage even if frontend checks are bypassed.

---

## 2. Authentication Flow & Token Handling

### Login & Session Bootstrap
1. **Credentials Submission**: Client sends user email and password via `POST /api/v1/auth/login`.
2. **Token Issuance**: Backend returns signed short-lived JWT `access_token` (15-min expiration) and an opaque `refresh_token` (7-day expiration).
3. **Identity Resolution**: Client immediately calls `GET /api/v1/auth/me` with the `Authorization: Bearer <access_token>` header.
4. **Context Initialization**: `AuthContext` populates:
   - `user`: Account profile (`id`, `email`, `full_name`, `is_super_admin`, `is_active`).
   - `roles`: Assigned canonical roles (`SUPER_ADMIN`, `HOSPITAL_ADMIN`, `CLINICIAN`, `PATIENT`).
   - `permissions`: Effective permission capabilities derived from backend user state.
   - `hospitalMemberships`: Authorized hospital institutional scopes for `HOSPITAL_ADMIN` and `CLINICIAN` actors.
   - `patientRecord`: Linked opaque patient entity for `PATIENT` actors.

---

## 3. Session Expiration & Single-Flight Token Refresh

- **Access Token Expiration**: Access tokens expire after 15 minutes.
- **Automatic Interception**: `services/api.js` intercepts HTTP `401 Unauthorized` responses on protected requests.
- **Single-Flight Refresh Queue**: When a 401 is encountered, incoming API requests pause while a single `POST /api/v1/auth/refresh` request executes with the current `refresh_token`.
- **Token Rotation Handling**:
  - Upon successful refresh, both the new `access_token` and rotated `refresh_token` update in session memory.
  - Suspended requests retry automatically with the new Bearer token.
- **Session Termination**: If token refresh fails (due to expiration, session revocation, or refresh-token reuse detection), `AuthContext.logout()` clears local storage and redirects the user to `/login`.

---

## 4. Token Storage Security Tradeoffs

- **Storage Strategy**: Access and refresh tokens are stored in browser `sessionStorage` (scoped to the active browser tab) to maintain state across page refreshes while ensuring automatic clearance upon tab closure.
- **XSS Mitigation**:
  - Tokens are never exposed in URLs, query strings, error messages, or console logs.
  - HTML content rendered across pages is sanitized, preventing XSS injection.
  - Request headers scrub sensitive tokens before logging.
- **Tradeoff Notice**: Storing tokens in `sessionStorage` allows client-side script access within the same tab context. When cookie support (`HttpOnly`, `SameSite=Strict`) is enabled in future backend releases, the frontend API client will transition seamlessly to cookie-based session handling.

---

## 5. Multi-Tenant Hospital Context & Patient Scoping

### Hospital Admin Context Selection
- Hospital Admins with memberships across multiple institutions select an active hospital context via `HospitalContextSelector.jsx`.
- The active `hospital_id` filters UI views; however, every API request sending `hospital_id` parameters is validated against backend membership tables (`HospitalMembership`). Switching client context manually without DB membership returns `404 Not Found`.

### Patient Ownership Isolation
- Patient users can access only their own linked patient profile and records (`/patient/*`).
- Patient ownership checks compare `user.id` against the server-authenticated patient record.
- Manual manipulation of patient IDs in URL paths returns `404 Not Found` from backend endpoints.

---

## 6. Zero Mock Data & Honest 501 Error States

- **No Fake Data**: In strict compliance with Prompt 00 privacy and model contracts, the frontend contains zero mock clinical records, fake diagnostic outputs, synthetic ROC-AUC curves, or fabricated ECG signals.
- **Standard 501 Handling**: Unimplemented backend clinical endpoints (GAT inference, BioClinicalBERT, DiCE, DoWhy, Flower FL) return HTTP status `501 Not Implemented`.
- **Standardized 501 Component**: The UI renders an explicit `FeatureNotImplemented` component displaying `"Feature Pending ML Implementation"` to maintain transparency.

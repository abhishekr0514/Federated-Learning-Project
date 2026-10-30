# Clinician Portal — Page-by-Page Audit

**Primary files:** `frontend/src/pages/clinician/ClinicianDashboard.jsx`, `frontend/src/components/DiagnosticWorkspace.jsx`, `frontend/src/layouts/ClinicianLayout.jsx`, `backend/app/api/v1/endpoints/clinician.py`, clinician services under `backend/app/services/`.

---

## Navigation tabs

From `Sidebar.jsx` (CLINICIAN): `dashboard`, `inference` (Diagnostic Studio), `patients`, `documents`, `questions`, `federated`, `diagnosis`, `causal`.  
(DiCE explanations nav item commented out.)

---

## Tab: Clinician Workspace (`dashboard`)

| Aspect | Detail |
|--------|--------|
| **API** | `GET /api/v1/clinician/overview` |
| **Data** | Assigned patient counts, document stats, capability hints |
| **Errors** | `ClinicalForbidden`, `ClinicalError`, separate `overviewError` state |

**Tests:** `tests/test_clinician_workspace.py`, `tests/test_clinician_overview_latency.py`.

---

## Tab: Assigned Patients (`patients`)

| Action | API | Backend authorization |
|--------|-----|------------------------|
| List assigned | `GET /api/v1/clinician/patients?limit&offset` | Clinician profile + hospital scope |
| List eligible for assignment | `GET /api/v1/clinician/patients/eligible` | Same |
| Assign | `POST /api/v1/clinician/patients/{id}/assignment` | Creates assignment row |
| Dossier | `GET /api/v1/clinician/patients/{id}` | Must be assigned or authorized |

**UI:** Search, pagination (`patientLimit=20`), drill-down dossier with accordions (prediction explanation placeholders, causal sections gated on capabilities).

**Evidence:** `ClinicianDashboard.jsx` lines 60–77, 313+; `api.js` clinician helpers.

**Why clinician sees a patient:** Active `ClinicianPatientAssignment` (and hospital membership alignment) — verified in `verify_patient_access` / clinician services (see `dependencies.py` patient access branch).

---

## Tab: Clinical Reports (`documents`)

| Action | API | Effect |
|--------|-----|--------|
| List | `GET /api/v1/clinician/documents` | Metadata list |
| Detail | `GET /api/v1/clinician/documents/{id}` | Includes sanitization metadata when processed |
| Upload | `POST /api/v1/clinician/documents/upload` multipart | Stores document linked to patient |
| Process | `POST /api/v1/clinician/documents/{id}/process` | Privacy pipeline job |

**Terms in implementation:**

- **Document status / metadata:** Stored on `MedicalDocument` (type, patient link, hospital).
- **Privacy masking / sanitization:** Processing populates masking results; patient reports tab shows “Privacy Sanitized” when `sanitization_status == "sanitized"`.
- **Sanitized entities:** From privacy pipeline (see `test_document_pipeline.py`) — not re-listed here to avoid stale field names; trace `document` services.

**Patient visibility:** Patient `GET /patient/reports` sees summaries only — not full clinician document viewer content.

---

## Tab: Diagnostic Studio (`inference`)

**Component:** `DiagnosticWorkspace.jsx` (embedded when `activeTab === 'inference'`).

### Workflow modes

- **ECG + report fusion** (encoder classifier path) vs **GAT fusion** (two 128-d embeddings) — toggled in UI (`inferenceMode` / gating via `isGatDiagnosticWorkflowEnabled`).

### Patient linkage

- Dropdown loads `GET /api/v1/clinician/patients?limit=100` (or hospital patients for hospital admin context in same component).
- **`patient_id` query param** on evaluate calls when selected — triggers persistence.

### End-to-end GAT path (VERIFIED IN CODE)

1. Optional: `apiValidatePrivacyMasking` → `POST /api/v1/privacy/mask-text`
2. `apiEncodeEcgReportEmbeddings` → `POST /api/v1/ecg-report/embed`
3. `apiEvaluateGatFusion` → `POST /api/v1/gat-fusion/evaluate?patient_id=...`
4. Backend: `authorize_patient_linked_inference` → `GATFusionService.evaluate` → DB commit

### Idempotency

- UI generates UUID (`generateUUID()`), refresh button regenerates (`DiagnosticWorkspace.jsx` ~958).
- Server matches `DiagnosisPrediction.provenance == f"idempotency:{key}"` per patient — **replay returns stored prediction** without re-running model (`gat_fusion_service.py` 117–122).

### ECG/report fusion path

- `apiEvaluateEcgReport` → single-stack inference; confidence semantics **P(MI/STTC)** for binary head (`ecg_report_runtime.py` ~387–393).

### Failure modes

| Case | Behavior |
|------|----------|
| Readiness false | UI disables run; readiness endpoints describe missing checkpoint/deps |
| Inference exception | Error surfaced in workspace panel |
| No patient selected | Evaluate may still run **without** persistence (research-only) |
| Unauthorized patient | 403 from auth layer |
| Repeat idempotency key | Stored row returned (GAT class from stored index) |

---

## Tab: Patient Questions (`questions`)

| Action | API |
|--------|-----|
| Load inbox | `GET /api/v1/clinician/questions?limit=50` |
| Respond | `PATCH /api/v1/clinician/questions/{question_id}` `{ response_text, status? }` |

**UI:** Textarea per question, “Send response” disabled until non-empty trim.

**Status:** VERIFIED IN CODE — **not** backend-only; inbox exists (`ClinicianDashboard.jsx` 656–705).

**Scope:** Questions filtered to clinician’s hospital (`PatientQuestionService` join on hospital_id).

---

## Tab: GAT Diagnosis (`diagnosis`)

- Fetches clinician diagnosis listing endpoints (`GET /api/v1/clinician/diagnosis`, per-patient variants).
- Shows prediction history / capability-gated content — parallel to diagnostic studio results but read-focused.

---

## Tab: Federated Node (`federated`)

- Wraps `FederatedDemoShell` / capability gates — monitors Flower status via `/api/v1/federated/*` (requires user with federated monitor permission; clinician may see read-only or gated UI depending on role permissions).

**Note:** Starting FL demo is typically **Super Admin** + `federated:monitor`; clinician tab may be informational.

---

## Tab: DoWhy Causal AI (`causal`)

- `CausalAnalysisPanel.jsx` — cohort selection and analyze against stored predictions; `GET /api/v1/causal/readiness`, `POST /api/v1/causal/analyze`.

---

## Clinician interaction table (major controls)

| Page | Control | Handler | API | Backend | DB/model | Status |
|------|---------|---------|-----|---------|----------|--------|
| Patients | Assign | POST assignment | `/clinician/patients/{id}/assignment` | assignment service | insert assignment | VERIFIED |
| Documents | Upload | `apiUploadClinicalDocument` | POST upload | document service | insert blob/metadata | VERIFIED |
| Documents | Process | `apiProcessClinicalDocument` | POST process | privacy pipeline | masking results | VERIFIED |
| Diagnostic | Load sample ECG | `handleLoadSampleEcg` | — | — | in-memory | VERIFIED |
| Diagnostic | Drop ECG file | `handleEcgDrop` | — | — | parse client-side | VERIFIED |
| Diagnostic | Validate privacy | `handleValidatePrivacy` | POST `/privacy/mask-text` | privacy | — | VERIFIED |
| Diagnostic | Run inference | `handleRunInference` | embed + evaluate | GAT or ECG services | optional insert prediction | VERIFIED |
| Diagnostic | Reset form | `handleResetForm` | — | — | — | VERIFIED |
| Diagnostic | New idempotency key | `setIdempotencyKey(generateUUID())` | — | — | affects provenance on next save | VERIFIED |
| Questions | Send response | PATCH | `/clinician/questions/{id}` | `PatientQuestionService` | update response | VERIFIED |
| Login/Logout | Auth | AuthContext | `/auth/*` | — | — | VERIFIED |

**Placeholder / not verified in UI:** DiCE explanations route commented out in sidebar.

---

## Authorization helpers (reference)

| Helper | Use |
|--------|-----|
| `authorize_patient_linked_inference` | GAT/ECG evaluate with `patient_id` |
| `verify_patient_access` | Generic patient resource access |
| `require_role(CLINICIAN, ...)` | Clinician endpoints |

**Tests:** `test_gat_fusion_auth_integration.py`, `test_gat_fusion_patient_persistence.py`, `test_diagnostic_studio_patient_list.py`, `test_clinician_profile_guard.py`.

---

## Patient visibility after clinician save

When `patient_id` provided and inference succeeds → row in `diagnosis_predictions` → patient `GET /patient/diagnosis` includes it in history/charts on next load.

**Delay / cache:** No WebSocket — patient must refresh or re-open tab.

---

## Gaps and misleading labels

1. “Live ML” badge on Diagnostic Studio — actual readiness depends on env flags and checkpoints (`FUSION_INFERENCE_ENABLED`, files on disk).
2. Federated tab for clinician ≠ full FL orchestration rights.
3. Causal and GAT research outputs labeled proxy in ML layer; ensure viva language matches `target_definition` strings in DB.

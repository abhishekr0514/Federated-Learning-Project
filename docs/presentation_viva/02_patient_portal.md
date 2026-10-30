# Patient Portal — Page-by-Page Audit

**Primary files:** `frontend/src/pages/patient/PatientDashboard.jsx`, `frontend/src/layouts/PatientLayout.jsx`, `frontend/src/components/Sidebar.jsx` (role PATIENT), `frontend/src/components/PatientAssessmentCharts.jsx`, `backend/app/api/v1/endpoints/patient.py`, `backend/app/services/patient_service.py`, `backend/app/services/patient_question_service.py`.

**Verification legend:** VERIFIED IN CODE | TESTED | OBSERVED IN UI | PARTIAL | NOT IMPLEMENTED

---

## Navigation model

Tabs (not URLs): `health`, `records`, `reports`, `assessment`, `results`, `questions`.  
Badges from `patientPortalNav.js` — e.g. messaging may show **“Not built”** when capability is `NOT_IMPLEMENTED` even if backend exists.

---

## Tab: My Health (`health`)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Landing summary for the authenticated patient |
| **API** | `GET /api/v1/patient/overview` |
| **Backend** | `PatientService.get_health_overview` |
| **Auth** | `get_current_patient_context` |
| **Data shown** | Opaque patient id, institution/hospital association, counts (records, reports), assessment/results **status enums**, capabilities (including messaging) |
| **Persistence** | Aggregates from DB (documents, ECG metadata, predictions count) — not static mock in service layer |
| **States** | Loading via `PatientLoadingState`; 403/404 via `PatientForbidden` / `PatientNotFound`; refresh re-fetches overview |
| **Patient actions** | Navigate via cards to other tabs; refresh |
| **Cannot do** | Run ML inference; edit clinical documents |

**Evidence:** `PatientDashboard.jsx` lines 64–75, 232+; `patient.py` lines 30–38.

**Assessment status semantics:** Overview exposes `assessment.status` and `results.status` (e.g. `NOT_ENABLED`, `NO_DATA`, `RESULTS_ON_FILE`, `AVAILABLE`) computed in `PatientService` — patient does **not** trigger assessment from this tab.

---

## Tab: My Records (`records`)

| Aspect | Detail |
|--------|--------|
| **API** | `GET /api/v1/patient/records` |
| **Backend** | `PatientService.get_records` |
| **DB** | Patient-linked documents, ECG study metadata, observations (see service queries) |
| **UI** | Search filter (client-side on fetched list), record detail selection |
| **Empty** | `status: NO_DATA` → `PatientEmptyState` |
| **Audit** | Audit log on access (service) |

**Evidence:** `PatientDashboard.jsx` 78–88, 373+; `patient_service.py` records section ~280–295.

**Data type:** Persisted seeded/demo or clinician-uploaded data — **not** fabricated vitals in service when empty.

---

## Tab: My Reports (`reports`)

| Aspect | Detail |
|--------|--------|
| **API** | `GET /api/v1/patient/reports` |
| **Backend** | `PatientService.get_reports` |
| **DB** | `MedicalDocument` + `masking_results` |
| **Displayed** | Title from `document_type`, date, protection status (“Privacy Protected” / “Privacy Sanitized”) |
| **Not shown** | Full raw report text in patient API response (privacy-safe summary string) |

**Evidence:** `patient_service.py` 298–358.

---

## Tab: My Assessment (`assessment`)

| Aspect | Detail |
|--------|--------|
| **APIs** | `GET /api/v1/patient/overview` + `GET /api/v1/patient/diagnosis` |
| **Purpose** | Explain AI-assisted research proxy; link to results and questions |
| **Inference** | **Patients cannot start inference** — no patient-scoped evaluate endpoint |
| **Disclaimers** | `PatientSafetyDisclaimer`, `PatientStates` mandatory unavailable phrasing (also enforced in frontend tests `frontend/tests/patient_portal.test.js`) |

**Evidence:** `PatientDashboard.jsx` 157–159, 546+.

---

## Tab: My Results (`results`)

| Aspect | Detail |
|--------|--------|
| **API** | `GET /api/v1/patient/diagnosis` |
| **Backend** | `PatientService.get_diagnosis` → `_load_prediction_history`, `_build_chart_payload` |
| **DB** | `diagnosis_predictions` joined with `model_versions` |
| **Charts** | `PatientAssessmentCharts` — timeline and confidence (needs ≥2 points for trend message) |
| **Class display** | `resolve_patient_portal_proxy_class` (`prediction_proxy_class.py`) |

### Result state machine (patient-facing)

| Condition | What patient sees | Mechanism |
|-----------|-------------------|-----------|
| No rows | `NO_DATA`, empty states | No predictions for patient |
| GAT row with `predicted_class_index` 0/1 | NORM / MI/STTC + confidence % | STORED |
| GAT legacy row (null class index) | **“Legacy (class not stored)”** — class **not** inferred from confidence | `proxy_class_status: LEGACY_UNAVAILABLE` |
| ECG fusion row without index | Class derived from confidence as P(MI/STTC) threshold 0.5 | `DERIVED_FROM_ECG_PROBABILITY` |
| Model unavailable / not enabled | Overview assessment status `NOT_ENABLED` / `UNAVAILABLE` | Capability + empty history |

**Confidence vs class (GAT):** Confidence stored is **max(softmax)** (`fusion_runtime.py`); displayed class uses **argmax index** when persisted. Evaluator question: “Why not use confidence ≥ 0.5 for GAT?” — because max-softmax is not P(MI/STTC).

**Evidence:** `patient_service.py` 362–401; `prediction_proxy_class.py` 36–71.

---

## Tab: Questions for My Clinician (`questions`)

| Aspect | Detail |
|--------|--------|
| **GET** | `GET /api/v1/patient/questions` → list + templates + emergency notice |
| **POST** | `POST /api/v1/patient/questions` body `{ question_text, optional assigned_clinician_id }` |
| **Backend** | `PatientQuestionService` → `PatientClinicianQuestion` table |
| **UI** | Draft textarea (max 2000), suggested templates, submission history, clinician responses when present |
| **Badge** | May still show “Not built” in sidebar if `capabilities.patient_messaging === 'NOT_IMPLEMENTED'` while feature works — **UI inconsistency** |

**Production risk:** Missing PostgreSQL GRANTs → 500 on GET (see `01_project_architecture.md`).

**Evidence:** `PatientDashboard.jsx` 675–752; `patient.py` 108–132; clinician response visible when `response_text` populated.

---

## Patient inference capability

| Question | Answer |
|----------|--------|
| Can patient run GAT/ECG inference? | **No** — evaluate endpoints require clinician/hospital admin/super admin roles |
| Can patient view clinician-run results? | **Yes**, when rows exist in `diagnosis_predictions` for their `patient_id` |
| Real-time inference status on patient UI? | **Partial** — status enums on overview, not live job polling |

**Status:** VERIFIED IN CODE.

---

## Interaction table (Patient Portal)

| Page | Control | Handler | API | Backend | DB / model | UI outcome | Status |
|------|---------|---------|-----|---------|------------|------------|--------|
| All | Sidebar tab | `setActiveTab` | On tab change | respective GET | read | Load section | VERIFIED |
| Health | Refresh | `fetchOverview` | GET `/patient/overview` | `get_health_overview` | aggregate | Updated cards | VERIFIED |
| Records | Search | local filter | — | — | — | Filters list | VERIFIED |
| Records | Refresh | `fetchRecords` | GET `/patient/records` | `get_records` | read | List | VERIFIED |
| Reports | Refresh | `fetchReports` | GET `/patient/reports` | `get_reports` | documents | List | VERIFIED |
| Assessment | “Go to questions” | `setActiveTab('questions')` | — | — | — | Nav | VERIFIED |
| Results | Refresh | `fetchDiagnosis` | GET `/patient/diagnosis` | `get_diagnosis` | predictions | History + charts | VERIFIED |
| Questions | Submit | POST handler | POST `/patient/questions` | `submit_question` | insert question | Clears draft, reload | VERIFIED IN CODE; TESTED (`test_patient_questions_api.py`) |
| Questions | Template chip | sets draft | — | — | — | Prefill | VERIFIED |
| Login | (Auth layout) | `AuthContext.login` | POST `/auth/login` | auth service | session tokens | Dashboard | VERIFIED |
| Logout | Navbar | clears tokens | — | — | — | Login page | VERIFIED IN CODE |

**No handlers found for:** patient-side document upload, patient-side inference, patient-side FL controls.

---

## Automated tests (patient)

| Test file | Scope |
|-----------|--------|
| `tests/test_patient_portal.py` | API overview, records, isolation |
| `tests/test_patient_questions_api.py` | Questions CRUD |
| `frontend/tests/patient_portal.test.js` | Disclaimer copy |

**Inspection run (2026-10-27):** `pytest` on patient + questions + clinician + gat auth: **34 passed, 1 failed** — `test_patient_can_access_own_overview_and_records` (likely seed/count drift vs assertion `stored_count == 0`). Treat as **environment/seed sensitivity**, not necessarily production bug.

---

## Evaluator challenges

1. Sidebar “Privacy Boundary / zero PII across hospitals” vs actual patient data staying in one hospital DB.
2. “Not built” badge on Questions while POST/GET exist.
3. Legacy GAT rows without class index — honest labeling vs old conflated confidence threshold.
4. Research proxy labels (NORM vs MI/STTC) vs clinical diagnosis language in UI.

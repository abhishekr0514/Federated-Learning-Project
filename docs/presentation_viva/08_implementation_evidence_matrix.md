# Implementation Evidence Matrix

**Legend:** ✅ VERIFIED IN CODE | 🧪 TESTED | ⚠️ PARTIAL | ❌ NOT IMPLEMENTED / 501

| Feature | Frontend | API | Backend service | DB / model | Tests | Status |
|---------|----------|-----|-----------------|------------|-------|--------|
| Role login & dashboard routing | `App.jsx`, `AuthContext.jsx` | `/auth/login`, `/auth/me` | auth services | users, roles | `test_auth_rbac.py` | ✅ 🧪 |
| Patient overview | `PatientDashboard.jsx` | `GET /patient/overview` | `PatientService.get_health_overview` | aggregates | `test_patient_portal.py` | ✅ 🧪 |
| Patient records | same | `GET /patient/records` | `get_records` | documents, ECG meta | `test_patient_portal.py` | ✅ 🧪 |
| Patient reports | same | `GET /patient/reports` | `get_reports` | `medical_documents` | `test_patient_portal.py` | ✅ |
| Patient results & charts | `PatientAssessmentCharts.jsx` | `GET /patient/diagnosis` | `get_diagnosis`, `_load_prediction_history` | `diagnosis_predictions` | `test_patient_portal.py`, GAT persistence tests | ✅ 🧪 |
| Patient submit question | `PatientDashboard.jsx` | `POST /patient/questions` | `PatientQuestionService.submit_question` | `patient_clinician_questions` | `test_patient_questions_api.py` | ✅ 🧪 |
| Patient list questions | same | `GET /patient/questions` | `list_patient_questions` | same | `test_patient_questions_api.py` | ✅ 🧪 (⚠️ DB grants) |
| Patient run inference | — | — | — | — | — | ❌ by design |
| Clinician overview | `ClinicianDashboard.jsx` | `GET /clinician/overview` | clinician service | assignments | `test_clinician_workspace.py` | ✅ 🧪 |
| Clinician patients | same | `GET /clinician/patients` | scoped list | assignments | `test_clinician_workspace.py` | ✅ |
| Assign patient | same | `POST .../assignment` | assignment | insert | demo alignment tests | ✅ |
| Document upload/process | same | upload, process | document pipeline | documents, masking | `test_document_pipeline.py` | ✅ 🧪 |
| Diagnostic Studio ECG/GAT | `DiagnosticWorkspace.jsx` | `/ecg-report/*`, `/gat-fusion/*` | runtime + services | optional prediction insert | `test_gat_fusion_api.py`, auth integration | ✅ 🧪 |
| GAT class index persist | — | evaluate | `gat_fusion_service.py` | `predicted_class_index` | `test_gat_predicted_class_persistence.py` | ✅ 🧪 |
| Idempotency | UI UUID | evaluate body | `_find_idempotent_prediction` | `provenance` | fusion persistence tests | ✅ 🧪 |
| Clinician question inbox | `ClinicianDashboard.jsx` | GET/PATCH `/clinician/questions` | `PatientQuestionService` | messaging table | partial API tests | ✅ |
| Privacy mask demo | DiagnosticWorkspace | `POST /privacy/mask-text` | privacy router | — | privacy tests | ✅ |
| Flower start/stop | `FederatedLearningPanel.jsx` | `/federated/start` | `FlowerSupervisor` | — | `test_federated_orchestration_rbac.py`, flower e2e | ✅ 🧪 |
| FedAvg GAT FL | admin/clinician shell | — | `medshield_fl_app.py` | global checkpoint file | `test_gat_ecg_report_fl.py` | ✅ |
| Offline evaluation report | admin research panels (if wired) | **501** `/metrics/evaluation` | — | — | `test_model_evaluation_report.py` | ⚠️ static files only |
| Causal analysis | `CausalAnalysisPanel.jsx` | `/causal/*` | causal service | predictions | `test_causal_analysis_service.py` | ✅ 🧪 |
| Audit API | — | `GET /audit/` | — | — | — | ❌ 501 |
| Generic `/patients` API | — | 501 | — | — | — | ❌ |

---

## End-to-end chains (short)

**GAT → patient visibility:**  
`DiagnosticWorkspace` → `apiEncodeEcgReportEmbeddings` → `apiEvaluateGatFusion(patientId)` → `GATFusionService` → `DiagnosisPrediction` → `PatientService.get_diagnosis` → `PatientDashboard` results tab.

**Question round-trip:**  
Patient `POST /patient/questions` → `PatientClinicianQuestion` → Clinician `GET /clinician/questions` → `PATCH` response → Patient `GET /patient/questions` shows `response_text`.

---

## Inspection test run (2026-10-27)

Command: `pytest tests/test_patient_portal.py tests/test_patient_questions_api.py tests/test_clinician_workspace.py tests/test_gat_fusion_auth_integration.py`

Result: **34 passed, 1 failed** — `test_patient_can_access_own_overview_and_records` failed at `test_patient_portal.py:304`: expected `diag["assessment_status"] == "NOT_ENABLED"` but API returned **`NO_DATA`** (seed patient A has no stored prediction; diagnosis payload status semantics changed or test is stale). **Not a runtime portal crash** — adjust test or align `PatientService.get_diagnosis` status enums if you want a green CI before viva.

---

## Extended interaction audit (Task 4 — buttons & controls)

| Page | Control/label | Frontend handler | API | Backend | DB/model effect | UI outcome | Status |
|------|---------------|------------------|-----|---------|-----------------|------------|--------|
| Login | Sign in | `AuthContext.login` | POST `/auth/login` | auth | session tokens | Dashboard by role | ✅ |
| Login | Logout | token clear in Navbar/Auth | — | — | — | Login page | ✅ |
| Patient Health | Refresh / retry | `fetchOverview` etc. | GET overview | PatientService | read | Reload | ✅ |
| Patient Health | Quick links | `setActiveTab` | — | — | — | Tab change | ✅ |
| Patient Records | Select record | `setSelectedRecord` | — | — | — | Detail pane | ✅ |
| Patient Assessment | Validation toggle | `setValidationOpen` | — | — | — | Expand copy | ✅ |
| Patient Questions | Template | `setQuestionDraft` | — | — | — | Prefill | ✅ |
| Clinician Dashboard | Open Diagnostic Studio | `setActiveTab('inference')` | — | — | — | Embed workspace | ✅ |
| Clinician Patients | Select patient | `handleSelectPatient` | GET `/clinician/patients/{id}` | dossier | read | Dossier view | ✅ |
| Clinician Patients | Assign | `handleAssignPatient` | POST assignment | assignment svc | insert | List update | ✅ |
| Clinician Patients | Pagination | `setPatientPage` | GET with offset | list | read | Page | ✅ |
| Clinician Patients | Back | `handleBackToPatients` | — | — | — | List view | ✅ |
| Clinician Dossier | Accordions | `toggleSection` | — | — | — | Expand sections | ✅ |
| Clinician Documents | Upload modal | upload handler | POST upload | documents | insert | New doc | ✅ |
| Clinician Documents | Process | `handleProcessDocument` | POST process | pipeline | masking | Status update | ✅ |
| Clinician Documents | View metadata | `fetchDocumentDetail` | GET doc by id | — | read | Modal/detail | ✅ |
| Clinician Documents | Pagination | `setDocumentPage` | GET list | — | read | Page | ✅ |
| Diagnostic Studio | Mode toggle | set inference mode | — | — | — | GAT vs ECG path | ✅ |
| Diagnostic Studio | File upload ECG | file input + drop | — | — | client parse | Waveform | ✅ |
| Diagnostic Studio | Run inference | `handleRunInference` | embed + evaluate | GAT/ECG svc | optional insert | Result card | ✅ |
| Diagnostic Studio | Refresh idempotency | `generateUUID` | — | — | next save provenance | New key | ✅ |
| Clinician Questions | Mark resolved / respond | PATCH handlers | PATCH question | question svc | update row | Inbox update | ✅ |
| Clinician GAT tab | Load diagnosis list | fetch in useEffect | GET `/clinician/diagnosis` | — | read | Table | ✅ |
| Clinician Federated | Shell controls | FederatedDemoShell | `/federated/*` | orchestration | — | Status | ⚠️ permission |
| Super Admin FL | Start/stop | FederatedLearningPanel | POST start/stop | FlowerSupervisor | processes | Demo rounds | ✅ 🧪 |
| Hospital Admin | (parallel tabs) | HospitalAdminDashboard | `/hospital/*` | hospital svc | read | Not assigned module | ✅ |

**Placeholder / no-op:** DiCE sidebar entry commented out; `/audit/` and generic `/patients` API unused by portals.

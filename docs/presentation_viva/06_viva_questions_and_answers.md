# Viva Questions and Answers (50+)

Use these with file references from other docs. Follow-ups marked **F/U**.

---

## Architecture & stack

**Q1. What is the frontend stack?**  
React (Vite), single-page app without React Router; role picked in `App.jsx` after JWT login.

**F/U:** Why no URL routes? — Scope choice for demo dashboards; deep linking not implemented.

**Q2. What is the backend?**  
FastAPI, `/api/v1` router, async PostgreSQL (`backend/app/main.py`).

**Q3. How do frontend and backend authenticate?**  
JWT in `sessionStorage`, `apiRequest` attaches Bearer token, refresh via `/api/v1/auth/refresh` (`api.js`, `AuthContext.jsx`).

**F/U:** Is RBAC only on server? — Yes; frontend permission set is for UX hints only.

**Q4. Name the four portal roles.**  
SUPER_ADMIN, HOSPITAL_ADMIN, CLINICIAN, PATIENT.

**Q5. Which API returns 501 by design?**  
`/api/v1/metrics/evaluation` — avoids fabricated metrics (`metrics.py`).

---

## Patient portal

**Q6. Can a patient run GAT inference?**  
No. `/gat-fusion/evaluate` requires clinician, hospital admin, or super admin (`gat_fusion.py`).

**Q7. How does a patient see results?**  
Clinician runs inference with `patient_id` → row in `diagnosis_predictions` → `GET /api/v1/patient/diagnosis`.

**Q8. What endpoints power My Health?**  
`GET /api/v1/patient/overview`.

**Q9. What is shown in My Reports?**  
Metadata from `MedicalDocument` — not full raw text (`PatientService.get_reports`).

**Q10. What happens if no predictions exist?**  
`NO_DATA` status and empty states in UI.

**Q11. Explain legacy GAT results.**  
If `predicted_class_index` is null, portal shows “Legacy (class not stored)” and does **not** infer class from max-softmax confidence (`prediction_proxy_class.py`).

**F/U:** Why was confidence misleading before? — Max softmax for NORM can exceed 0.5; threshold logic falsely implied MI/STTC.

**Q12. Can patients upload ECG?**  
Not in patient portal — clinician Diagnostic Studio only.

**Q13. Patient questions — which APIs?**  
`GET/POST /api/v1/patient/questions`.

**F/U:** Why might GET return 500 in dev? — PostgreSQL table privileges for app role; run grant scripts.

**Q14. What disclaimer is mandatory in patient UI?**  
Wording enforced in tests: discuss with qualified healthcare professional (`frontend/tests/patient_portal.test.js`).

**Q15. What do assessment chart trends require?**  
At least two stored assessments (`patient_service.py` `_build_chart_payload`).

---

## Clinician portal

**Q16. How is patient list scoped?**  
Hospital membership + clinician assignments (`/api/v1/clinician/patients`).

**Q17. Document upload flow?**  
POST upload → optional POST process for sanitization (`ClinicianDashboard`, document endpoints).

**Q18. What is Diagnostic Studio?**  
`DiagnosticWorkspace.jsx` — ECG samples, report text, privacy validate, embed, GAT or ECG evaluate.

**Q19. What is idempotency key for?**  
Prevents duplicate predictions for same patient when key repeats; stored in `provenance` field (`gat_fusion_service.py`).

**F/U:** Does idempotency skip model on replay? — Returns existing DB row via `_find_idempotent_prediction`.

**Q20. Can clinician run inference without patient?**  
Yes — evaluate without `patient_id` is research-only, no persistence (`gat_fusion.py` docstring).

**Q21. Clinician question inbox — implemented?**  
Yes — tab in `ClinicianDashboard`, PATCH response endpoint.

**Q22. Difference between `/ecg-report/evaluate` and `/gat-fusion/evaluate`?**  
Full encoder classifier vs GAT on precomputed 128-d embeddings.

---

## Authorization & security

**Q23. Who can persist predictions?**  
Roles on evaluate endpoints + `authorize_patient_linked_inference` for patient linkage.

**Q24. How is cross-hospital leakage prevented?**  
Hospital context on documents/questions; unauthorized hospital IDs may 404 (`dependencies.py`).

**Q25. Does sidebar “zero PII across boundaries” mean FL sends no patient data?**  
FL demo sends **model weights**, not Postgres patient rows; UI statement is aspirational marketing — explain honestly.

**Q26. Are audit logs written for patient reads?**  
Yes for reports/records paths in `PatientService` (e.g. `PATIENT_REPORTS_ACCESS`).

**Q27. Flower start permission?**  
`SystemPermission.FEDERATED_MONITOR` on federated routes.

---

## Database

**Q28. Table for predictions?**  
`diagnosis_predictions` with `confidence`, `predicted_class_index`, `model_version_id`, `provenance`.

**Q29. What migration added class index?**  
`006_diagnosis_prediction_class_index.py`.

**Q30. Patient-clinician messages table?**  
`patient_clinician_questions` (`messaging.py`).

**Q31. Foreign keys on predictions?**  
`patient_id`, `hospital_id`, `model_version_id` (`ml_models.py`).

---

## ML — encoders & GAT

**Q32. ECG encoder input shape?**  
(12, 1000) (`ecg_encoder.py`).

**Q33. Clinical text model ID?**  
`emilyalsentzer/Bio_ClinicalBERT`.

**Q34. Embedding dimensions before GAT?**  
128 + 128.

**Q35. How many GAT nodes?**  
Two — ECG and clinical text (`gat_ecg_report_fusion.py`).

**Q36. GAT edge structure?**  
Fully connected with self-loops (4 directed edges).

**Q37. How is predicted class computed at inference?**  
argmax(softmax(logits)); confidence = max(softmax) (`fusion_runtime.py`).

**Q38. Class 0 vs 1 meaning?**  
0 = NORM, 1 = MI/STTC (manifest + schemas).

**F/U:** Is this clinical MI diagnosis? — No; PTB-XL **proxy** label for research.

**Q39. Which layers train in FL?**  
GAT head only; embeddings frozen in bundle.

**Q40. Where is centralized GAT checkpoint?**  
`models/gat_ecg_report/best_gat_checkpoint.pt`.

---

## Federated learning

**Q41. What is FedAvg aggregating here?**  
GAT head NumPy weight vectors from two simulated clients (`medshield_fl_app.py`).

**Q42. Default FL rounds in demo?**  
2 (`context_run_config_rounds` / `_RUN_CFG_ROUNDS`).

**Q43. What data never leaves the “hospital” in FL code path?**  
Raw ECG and raw reports in DB — FL uses precomputed embedding tensors in bundle file.

**Q44. SuperLink role?**  
Flower 1.30 orchestration via backend `FlowerSupervisor` (see `flower_supervisor.py` in repo).

**Q45. Is differential privacy applied to gradients?**  
Not implemented in codebase — honest answer: **no formal DP**.

---

## Metrics & evaluation

**Q46. Current test set size for report?**  
1838 (`evaluation_manifest.json`).

**Q47. Centralized vs federated AUROC?**  
Both ≈ 0.9651 on same embeddings (CSV).

**Q48. Why two AUROC numbers (~0.96 vs 0.965) in conversation?**  
200-sample smoke bundle vs full test — different runs; do not combine.

**Q49. AUROC score input?**  
softmax class-1 probability, not max-softmax (`classification_metrics.py`).

**Q50. Confusion matrix orientation?**  
Rows actual, columns predicted, labels [NORM, MI/STTC].

**Q51. What is MCC for centralized model?**  
≈ 0.7903 (CSV).

**Q52. Does high AUROC imply calibrated confidence?**  
No — separate uncalibrated max-softmax in API.

---

## Testing & quality

**Q53. Tests for patient questions?**  
`tests/test_patient_questions_api.py`.

**Q54. Tests for GAT persistence?**  
`tests/test_gat_predicted_class_persistence.py`, `test_gat_fusion_patient_persistence.py`.

**Q55. Inspection pytest result?**  
34 passed, 1 failed in `test_patient_portal.py` overview counts — check seed data.

---

## Hard cross-questions

**Q56. If confidence is 0.9 and class NORM, could old UI show MI/STTC?**  
Yes if thresholding max-softmax as P(MI/STTC) — fixed by storing `predicted_class_index`.

**Q57. Idempotency with new key but same inputs?**  
New prediction row — new provenance string.

**Q58. ECG fusion legacy without index?**  
Class derived from confidence ≥ 0.5 as P(MI/STTC) (`ecg_idempotent_replay_class`).

**Q59. Overfitting concern?**  
Val fold 8 for selection; test fold 10 held out per manifest — still in-distribution PTB-XL.

**Q60. Why only 1500 train embeddings in bundle?**  
Reused embedding cache for demo/train speed — limitation for “full FL on all PTB-XL train.”

**Q61. Simulated clients vs real hospitals?**  
Partitions of same bundle file — research demo, not production multi-tenant FL.

**Q62. Can hospital A clinician see hospital B patient?**  
Should fail assignment/hospital checks — cite `verify_patient_access` tests (`test_tenant_scoping.py`).

**Q63. What proves patient isolation in API tests?**  
`test_patient_portal.py` cross-patient denial cases (inspect file for scenarios).

**Q64. Is DoWhy causal analysis clinically validated?**  
Research/explainability add-on — readiness gated (`causal/readiness`).

**Q65. Where are evaluation HTML reports?**  
`reports/model_evaluation/` — safe to show offline if live API 501.

---

## Closing traps

**Q66. “MedShield diagnoses heart disease” — correct?**  
Incorrect wording — **proxy research classification** with disclaimers.

**Q67. Does FL automatically protect patient privacy in production?**  
No — only specific data types are excluded from Flower messages; governance and legal review still required.

**Q68. Patient messaging badge says Not built — is it broken?**  
Capability enum may lag; backend and clinician inbox can still work — **UI inconsistency**.

**Q69. What would you demo if inference readiness is false?**  
Static reports, architecture slides, pre-seeded patient results, offline confusion matrix PNGs.

**Q70. Strongest verified story for your modules?**  
Role-separated portals, clinician-triggered inference with persistence, patient read-only results with honest legacy handling, question submit + clinician reply path in code and tests.

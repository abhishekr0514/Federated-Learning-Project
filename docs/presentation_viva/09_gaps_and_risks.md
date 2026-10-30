# Gaps, Risks, and Honest Limitations

For viva defense — acknowledge proactively.

---

## 1. Feature gaps

| Item | Detail | Severity |
|------|--------|----------|
| Live metrics API | `/api/v1/metrics/evaluation` returns 501 | Medium — use static `reports/model_evaluation/` |
| Patient deep links | Tab state only, no URLs | Low |
| Patient inference | Not implemented (by design) | Explain as safety/governance |
| Audit log REST | `/audit/` 501 | Medium for “governance story” |
| Generic patients/hospitals routers | 501 stubs | Low — real flows use role-specific routes |
| DiCE explanations nav | Commented out in clinician sidebar | Low |

---

## 2. UI vs backend inconsistencies

| Issue | Evidence |
|-------|----------|
| Questions sidebar badge “Not built” while API works | `patientPortalNav.js` maps `NOT_IMPLEMENTED`; capabilities may disagree with deployed DB |
| Sidebar “Zero raw PII across hospital boundaries” | Aspirational copy; FL demo uses local bundle, not live cross-hospital EHR |
| “Live ML” badge | Depends on env readiness flags and checkpoint files |

---

## 3. Data & evaluation risks

| Risk | Mitigation narrative |
|------|---------------------|
| PTB-XL proxy ≠ clinical gold standard | Cite manifest disclaimer and patient disclaimers |
| n=200 vs n=1838 metrics | Always state which bundle/run; current repo artifacts are **n=1838** |
| Train bundle 1500 vs 12898 eligible | FL trained on subset; full test eval is still 1838 |
| Uncalibrated confidence in portal | Distinguish from AUROC score (class-1 prob) |
| Legacy rows without `predicted_class_index` | UI shows legacy label — do not backfill with wrong heuristic for GAT |

---

## 4. Operational / database

| Risk | Symptom | Fix class |
|------|---------|-----------|
| Missing GRANT on `patient_clinician_questions` | Patient questions HTTP 500 | DBA grant scripts in repo |
| Inference not ready | Readiness false, buttons ineffective | Deploy checkpoints, env flags |
| Seed / test drift | `test_patient_can_access_own_overview_and_records` expects `assessment_status=NOT_ENABLED`, API returns `NO_DATA` (line 304) | Update test expectation or service enum mapping |

---

## 5. Security & privacy (examiner angles)

- **JWT in sessionStorage** — XSS risk class; mention HTTPS and short-lived tokens in production hardening (not all mitigations verified in repo).
- **Flower TLS/auth** — inspect `flower_supervisor` / deploy configs before claiming mTLS; default may be local demo.
- **Document sanitization** — pipeline exists; **verify** on uploaded sample rather than assuming all docs processed.
- **No formal DP on FL gradients** — do not overclaim.

---

## 6. Testing holes

- One failing test in patient portal suite at inspection time — investigate seed vs assertion before claiming “all green.”
- End-to-end browser E2E for full clinician→patient flow may rely on manual demo, not single Playwright spec (check `frontend/tests/` — mostly unit/copy tests).

---

## 7. Misleading phrases to avoid in viva

- “Diagnoses myocardial infarction” → use **research proxy NORM vs MI/STTC**.
- “Federated learning trains on hospital EHR live” → **simulated clients, embedding bundle**.
- “Confidence is probability of disease” → **GAT: max softmax; ECG path: P(MI/STTC)**.
- “API serves evaluation metrics” → **501; files on disk**.

---

## 8. Strongest verified claims (use these)

1. Server-enforced RBAC on inference and patient data APIs.
2. Clinician-only inference with optional persistence and idempotency.
3. Patient read-only history with GAT legacy class handling fixed in code.
4. Patient questions + clinician inbox PATCH flow implemented and API-tested.
5. Reproducible offline evaluation artifact with documented n=1838 confusion matrices and AUROC ~0.965 for both centralized and federated GAT heads on the same embeddings.

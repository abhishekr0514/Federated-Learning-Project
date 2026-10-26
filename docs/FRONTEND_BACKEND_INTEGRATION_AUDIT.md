# MedShield-FL Frontend ↔ Backend Integration Audit

Generated: 2026-10-26 (local repository state)  
**Final E2E verification:** 2026-10-26 (live stack: Vite `:5173`, FastAPI `127.0.0.1:8000`, PostgreSQL `medshield_fl_db`)

## Final E2E pass / fail summary

| Area | Result | Evidence / cause | Next action if failing |
|------|--------|------------------|------------------------|
| **Login (API + UI)** | **PASS** | `scripts/e2e_verify_integration.py`: superadmin + hospital admin HTTP 200, envelope `data.access_token`, invalid password 401. Browser: Super Admin quick-login → role nav + Logout visible. | Re-run `scripts/seed_postgres_dev.py --non-interactive` if admin rows missing; use `ValidPassword123!` for seeded admins only. |
| **Database / schema** | **PASS** | `/health` → `database: operational`. Alembic at `004_alerts_dq`. `scripts/db_verify_dev.py`: 4 users, roles `CLINICIAN`, `HOSPITAL_ADMIN`, `PATIENT`, `SUPER_ADMIN`, 35 public tables, required `users`/`roles`/`hospitals`/`alerts`/`data_quality_events` present. | If `hospital/overview` 500: `py -m alembic -c alembic.ini upgrade head` (revision ids must be ≤32 chars). |
| **Frontend proxy / API base** | **PASS** | `frontend/vite.config.js` proxies `/api` and `/health` → `http://127.0.0.1:8000`. `API_BASE_URL` empty → same-origin proxy. `GET http://localhost:5173/health` returns backend JSON. | Start backend on `:8000` and `npm run dev` in `frontend/`. |
| **ECG/report inference (API)** | **PASS** | Readiness `ready: true`. `POST /api/v1/ecg-report/evaluate` with 12×1000 leads + report text → 200, probability + research disclaimer. | Set `ECG_REPORT_INFERENCE_ENABLED=true` and deploy encoder checkpoint per readiness blockers. |
| **ECG/report inference (UI)** | **PASS** | Diagnostic Studio → Run Model Inference → “Model Prediction Evaluation” + “EXPERIMENTAL RESEARCH PROXY OUTPUT ONLY” disclaimer rendered. | Ensure logged-in role with evaluate permission (clinician/admin/superadmin). |
| **GAT inference (API)** | **PASS** | Readiness `ready: true`, checkpoint in `models/gat_ecg_report/`. `POST /api/v1/gat-fusion/evaluate` with `ecg_embedding` + `text_embedding` (128-d each) → 200. | Copy trained `best_gat_checkpoint.pt` + manifest; `FUSION_INFERENCE_ENABLED=true`; restart backend if readiness cached. |
| **GAT inference (UI workflow)** | **PASS** | GAT tab uses separate handler: `POST /ecg-report/embed` (server-side privacy mask + encoder) → `POST /gat-fusion/evaluate`. Does **not** call `/ecg-report/evaluate`. Live chain: embed 200 → GAT 200 + disclaimer. | Both encoder + GAT readiness must be true; ECG quality gate applies on embed. |
| **Dashboard data (hospital overview)** | **PASS** | `GET /api/v1/hospital/overview?hospital_id=…` → 200 after migration `004_alerts_dq` (was 500: missing `alerts`). | Pass `hospital_id` from `/hospital/accessible-nodes`. |
| **Privacy masking** | **PASS** | `POST /api/v1/privacy/mask-text` → 200 (API). UI exposes “Verify Privacy Boundary & Sanitize” on Diagnostic Studio. | — |
| **Automated tests** | **PASS** | See [Tests run (final)](#tests-run-final) below. | — |

---

## Capability audit & status corrections (2026-10-26)

### Root causes (contradictory UI)

| UI surface | Was wrong | Root cause | Fix |
|------------|-----------|------------|-----|
| Clinician **GAT Diagnosis** tab | Always “NOT IMPLEMENTED” | Hardcoded `ClinicalNotImplemented` in `ClinicianDashboard.jsx` while capability matrix used live registry | Tab now mounts `DiagnosticWorkspace` when ECG/GAT capabilities are `AVAILABLE`; otherwise shows registry `capability_details.*.reason` |
| Capability matrix vs GAT tab | Matrix AVAILABLE, tab offline | Different UI paths (registry vs static banner) | Single backend source: `get_capability_details()` + shared badges |
| Hospital **Active User Sessions** | “AVAILABLE” + large count (e.g. 32) | Counted all non-revoked `user_sessions` without `expires_at` filter; label implied live UI sessions | Query: non-revoked **and** `expires_at > now()`; card renamed **Active Refresh-Token Sessions** with explicit definition metadata |
| DoWhy card | NOT IMPLEMENTED while package installed | Registry only checked `import dowhy`; no service/dataset readiness | `CausalAnalysisService.get_readiness()` drives `dowhy_causal` status; real API at `/api/v1/causal/*` |
| Flower client | “READY” read as live training | `INFRASTRUCTURE_READY` conflated with aggregation running | `federated_status_service.py` exposes separate flags (`training_enabled=false`, `server_process_available=false`) |
| DiCE | Mixed badges | No backend pipeline | Stays `NOT_IMPLEMENTED` (or `MISSING_DEPENDENCY` if `dice_ml` absent) — no counterfactual API |

### Status pipeline (authoritative)

- **Backend:** `capability_registry.get_capability_details()` → `/api/v1/capabilities` + clinician `overview.capability_details`
- **Inference readiness:** `/api/v1/ecg-report/readiness`, `/api/v1/gat-fusion/readiness` (unchanged contracts)
- **Frontend badges:** `ClinicalStatusBadge` + `integrationStatus.js` (no fake OPERATIONAL)

### Active session metric (evidence)

- **Definition:** rows in `user_sessions` where `is_revoked = false` and `expires_at` is in the future (refresh-token sessions).
- **Not measured:** concurrent browser tabs, WebSocket sessions, hospital-scoped counts (`hospital_scoped: false` in API).
- **Previously inflated counts:** expired sessions still counted → fixed in `hospital_service.get_system_health`.

### DoWhy causal AI (GAT-grounded cohort) — estimability review (2026-10-26)

| Item | Detail |
|------|--------|
| Endpoints | `GET /api/v1/causal/readiness`, `GET /api/v1/causal/cohorts`, `POST /api/v1/causal/analyze` (analyze **disabled by default**) |
| Three concepts | **`causal_pipeline_available`** (DoWhy + GAT + cohort built); **`causal_estimand_identifiable`** (platform scientific review — default **false**); **`causal_claim_validated`** (always **false** from automation; refutation ≠ causation) |
| Default readiness | **`CAUSAL_DESIGN_REQUIRED`** when descriptive cohort is built but sex→proxy-label is **not** approved as a defensible causal question |
| Cohort | `gat_ecg_report_ptbxl_research_v1` from `embedding_cache_train.pt` + GAT inference; **train split only** (folds 1–7); test fold excluded; labels from **SCP superclasses** (`ml.ptbxl.labels`), verified against cache tensor vs PTB-XL train rows |
| Cap | Deterministic prefix: `deterministic_prefix_after_train_alignment` then first `CAUSAL_ANALYSIS_MAX_ROWS` rows |
| Descriptive API | `GET /causal/cohorts` includes `descriptive_statistics` (class balance, missingness, label mismatch count, GAT vs proxy accuracy) |
| Candidate question (not approved) | Cross-sectional sex vs SCP proxy label adjusting age — documented in `causal_design_review.py` with limitations (selection, no temporal ordering, same-ECG outcome, unmeasured confounding) |
| Analyze gate | `CAUSAL_EFFECT_ESTIMATION_ALLOWED=false` (default) — lab override only; responses still set `causal_claim_validated=false` |
| Config | `CAUSAL_ANALYSIS_ENABLED`, `CAUSAL_GAT_EMBEDDING_CACHE_PATH`, `CAUSAL_PTBNXL_ROOT`, `CAUSAL_EFFECT_ESTIMATION_ALLOWED` |
| Blocked | Variation in sex/label as proof of identifiability; GAT fields as treatment/outcome; `cardio_train.csv` DoWhy default |

### Flower federated learning (orchestration slice)

| Item | Detail |
|------|--------|
| Endpoints | `GET /api/v1/federated/status`, `POST /api/v1/federated/start` (202), `POST /api/v1/federated/stop`, `GET /api/v1/federated/rounds` |
| Default flags | `FLOWER_ORCHESTRATION_ENABLED=false`, `FLOWER_TRAINING_ENABLED=false` (production does not auto-train) |
| Runtime | In-process **synthetic** clients (`SyntheticInfrastructureAdapter`) + `weighted_fedavg` — real parameter aggregation, non-clinical data only |
| Status semantics | Separate `summary`, `lifecycle`, `flower_version`, registered/selected/successful client counts, round progress, last aggregation time, `last_error` |
| Auth | `FEDERATED_MONITOR` permission on status/start/stop |
| UI | `FederatedLearningPanel.jsx` polls `/federated/status`; start disabled unless flags + auth satisfied |
| Not claimed | Secure aggregation, DP, multi-hospital clinical FL, live Ray/Flower server unless separately deployed |
| In-process smoke | Proves FedAvg over synthetic adapters in the **API process** — **not** evidence of separate client processes, TLS, or hospital nodes |
| Tabular experiment read API | Unchanged (`GET /experiments/tabular/latest`) — historical `cardio_train.csv` simulation artifacts only |

#### Flower deployment-level test plan (not automated here)

| # | Scenario | Pass criteria |
|---|----------|----------------|
| 1 | Two+ client processes connect to Flower server with TLS + auth | Registered client count matches; unauthorized connect rejected |
| 2 | Round start with `min_fit_clients` | Selected/successful counts in `/federated/status`; parameters change after aggregation |
| 3 | Client dropout mid-round | Run → `FAILED` or partial round with documented `last_error`; no silent success |
| 4 | Stop/cancel | `POST /federated/stop` → lifecycle `STOPPING`/`SERVER_STOPPED` |
| 5 | Duplicate start | Second start rejected while `RUNNING` |
| 6 | Observability | Round history matches aggregated checkpoints; audit event for start/stop |
| 7 | Data boundary | Only weight tensors/metrics cross wire; no raw ECG/report payloads |

### Tests (estimability review pass — 2026-10-26 evening)

| Suite | Result |
|-------|--------|
| `tests/test_causal_design_review.py` | 4 pass |
| `tests/test_causal_analysis_service.py` | 4 pass |
| `tests/test_gat_cohort_validation.py` | 1 pass |
| `tests/test_flower_orchestration_service.py` | 3 pass (`flwr` smoke skipped if package missing) |
| `tests/test_capability_status_consistency.py` | 5 pass |
| `tests/test_capability_details_registry.py` | 1 pass |
| `tests/test_gat_fusion_api.py` + embed + permission | 25 pass |
| `frontend` `npm run build` | pass (this pass) |

**Not run in this pass:** `scripts/e2e_verify_integration.py`, full pytest suite, frontend unit tests beyond build, live multi-process Flower deployment plan above.

### Runtime UI fix (2026-10-26): telemetry timeout + sidebar `501`

| Issue | Root cause | Fix |
|-------|------------|-----|
| `Failed to fetch` on clinician workspace | `GET /api/v1/clinician/overview` called `get_capability_details()` → `CausalAnalysisService.get_readiness()` → **full GAT cohort build (~50s)**; frontend `apiRequest` default **45s** abort → network error | **Fast probe** path: `GATCausalCohortService.probe_cohort_availability()` for registry/overview; full build only on `GET /causal/readiness` (`build_cohort=True`) |
| Sidebar badges `501` | **Hardcoded** in `Sidebar.jsx` (not HTTP responses) | Sidebar loads `GET /api/v1/capabilities` and maps statuses via `capabilityNav.js` |
| GAT tab `NOT IMPLEMENTED` | Overview null after timeout + gate required `overview.capabilities` only | Independent `capabilityReport` fetch; `isGatDiagnosticWorkflowEnabled()` uses encoder/GAT `ready` flags |

**Live HTTP verification:** agent shell had no listener on `:8000` during this pass; restart uvicorn and re-check `/health` + `/api/v1/clinician/overview` after pull.

### Tests (earlier capability pass)

| Suite | Result |
|-------|--------|
| `tests/test_capability_registry_integration.py` | pass |
| `tests/test_hospital_admin_dashboard.py` | 9 pass |
| `npm run build` | pass |
| Live `GET /api/v1/causal/readiness` | 200 (summary depends on `dowhy` + dataset) |

### Remaining blockers

1. **DiCE** — no counterfactual model/API; remains NOT_IMPLEMENTED.
2. **Live Flower orchestration** — server/client round loop not implemented; do not mark training active.
3. **DoWhy on minimal installs** — without `[ml]` extra, status `MISSING_DEPENDENCY`.
4. **Hospital-scoped session metrics** — would need session→hospital linkage (not in current schema).

---

## GAT frontend integration (2026-10-26)

| Component | Change |
|-----------|--------|
| Backend | `POST /api/v1/ecg-report/embed` — authorized encoder extracts 128-d ECG + text embeddings (privacy mask server-side; no raw text in response). |
| Runtime | `ECGReportInferenceRuntime.extract_encoder_embeddings()` shared with evaluate preprocessing. |
| Frontend API | `apiEncodeEcgReportEmbeddings`, existing `apiEvaluateGatFusion` (120s timeout). |
| Diagnostic Studio | Separate `handleSubmitGatFusionPrediction` vs `handleSubmitEcgReportPrediction`; distinct button labels and result cards (`confidence_uncalibrated` for GAT). |
| Tests | `frontend/tests/diagnostic_gat_workflow.test.js` (route separation proof), `tests/test_ecg_report_embed_api.py`. |

---

## Authentication

| Item | Detail |
|------|--------|
| Frontend login | `LoginPage.jsx` → `AuthContext.login()` → `POST /api/v1/auth/login` |
| Token envelope | `{ status, data: { access_token, refresh_token, ... } }` — parsed via `parseAuthTokenEnvelope()` |
| API base | Vite proxy `/api` → `http://127.0.0.1:8000` (`frontend/vite.config.js`); `VITE_API_BASE_URL` unset = proxy |
| DB config | `backend/app/core/config.py` loads repo-root `.env`; `DATABASE_URL` built from `POSTGRES_*` |

### Root cause (admin login failure)

1. **Missing accounts/roles** — PostgreSQL had only clinician/patient; no `HOSPITAL_ADMIN` / `SUPER_ADMIN` until seed extended.
2. **Quick-login password mismatch** — UI used `Password123!` (11 chars); seed requires **≥12** for new passwords.
3. **Historical environment issues** — Migrations/`.env` not applied on some machines.

### Repair (fixed in code)

- Extended `scripts/seed_postgres_dev.py` with `admin_a@metro.org` (HOSPITAL_ADMIN) and `admin@medshield.org` (SUPER_ADMIN).
- `--non-interactive` seeding with default `ValidPassword123!` (new accounts only; does not overwrite existing hashes).
- Login quick-access aligned to `ValidPassword123!`.
- Token parsing in `AuthContext` and refresh in `api.js`.

### Verification (local, 2026-10-26)

| Test | Result |
|------|--------|
| Invalid superadmin password | HTTP **401** |
| `admin@medshield.org` + dev seed password | HTTP **200**, `/auth/me` → `SUPER_ADMIN` |
| `admin_a@metro.org` + dev seed password | HTTP **200** |
| Browser quick-login + protected Super Admin shell | **PASS** |
| Frontend production build | **Success** (`npm run build`) |

---

## Capability badges vs real endpoints (2026-10-26)

Source: `GET /api/v1/capabilities` (authenticated) + readiness routes. UI maps via `capability_registry.py` and `integrationStatus.js`.

| Capability key | Live state | Endpoint / basis | UI honesty |
|----------------|------------|------------------|------------|
| `ecg_report_inference` | **AVAILABLE** | `/ecg-report/readiness` + evaluate | Operational when ready |
| `multimodal_gat_fusion` | **AVAILABLE** | `/gat-fusion/readiness` + evaluate | Badge Ready; UI submit not wired for embeddings |
| `flower_federated` | **INFRASTRUCTURE_READY** | `flwr` import present; `POST /federated/start` → **501** | Not shown as live orchestration |
| `dice_counterfactuals` | **NOT_IMPLEMENTED** / **MISSING_DEPENDENCY** | No DiCE pipeline | Purple / offline panels |
| `dowhy_causal` | **NOT_IMPLEMENTED** / **MISSING_DEPENDENCY** | No DoWhy pipeline | Not operational |
| `clinical_decision_support` | **NOT_IMPLEMENTED** | — | — |
| `privacy_masking_pipeline` | **AVAILABLE** | `/privacy/mask-text` | — |
| Hospital `worker_health`, CPU/RAM/GPU | **NOT_MEASURED** | `/hospital/system-health` | Never shown as healthy workers |
| Tabular FL research panel | Research metrics only | `/federated/experiments/tabular/*` | Explicit “no live Flower server” copy |

**Rule verified:** Flower orchestration, explainability (DiCE/DoWhy), and worker telemetry are **not** represented as fully working.

---

## E2E blockers fixed this pass

| Issue | Root cause | Fix |
|-------|------------|-----|
| Hospital overview HTTP 500 | `relation "alerts" does not exist` | Migration `004_alerts_dq` (`backend/app/db/migrations/versions/004_hospital_alerts_and_data_quality.py`) |
| Alembic upgrade failed | Revision id `004_hospital_alerts_and_data_quality` > 32 chars for `alembic_version` | Renamed revision to `004_alerts_dq` |
| GAT readiness blocked | Deploy dir missing checkpoint file | Copy `best_gat_checkpoint.pt` into `models/gat_ecg_report/` (experiment artifacts preserved) |
| GAT tab always “Unavailable” | Hardcoded badge + disabled banner ignored `gatReady` | `DiagnosticWorkspace.jsx`: badge/banner tied to live readiness |
| `test_readiness_shows_artifacts_when_deployed` | Test env lacked deploy artifact | Isolated `MODEL_DIR` + manifest/checkpoint copy in test (19/19 GAT + capability tests) |
| `test_system_health_explicit_hardware_states` | Expected `NOT_IMPLEMENTED` for workers | Assert **NOT_MEASURED** (matches product + API) |

---

## Feature integration matrix

| UI surface | Backend endpoint(s) | Status | Notes |
|-----------|---------------------|--------|-------|
| Login / session | `/api/v1/auth/login`, `/refresh`, `/me`, `/logout` | **Verified** | Envelope parsing unified |
| `ModelStatusPanel` | `/health`, `/ecg-report/readiness`, `/gat-fusion/readiness` | **Operational** | Live readiness + blockers |
| `DiagnosticWorkspace` | `/privacy/mask-text`, `/ecg-report/evaluate`, `/ecg-report/embed`, `/gat-fusion/evaluate` | **Both workflows operational** | GAT: embed → fusion; ECG tab unchanged |
| Hospital overview cards | `/api/v1/hospital/overview` | **Operational** | Requires `hospital_id`; NO_DATA when empty |
| Hospital system health | `/api/v1/hospital/system-health` | **Connected** | Hardware/worker **NOT_MEASURED** |
| Hospital clinicians/patients/audit | `/api/v1/hospital/clinicians`, `/patients`, `/audit` | **Connected** | Pagination wired |
| Clinician overview capabilities | `/api/v1/clinician/overview` | **Verified** | Registry + readiness |
| SuperAdmin hospitals list | `/api/v1/hospital/accessible-nodes` | **Connected** | Admin login required |
| `FederatedLearningPanel` | `/api/v1/federated/*` | **Partial** | Start campaign **501** |
| DiCE / DoWhy / CDS | N/A | **NOT_IMPLEMENTED** | Honest badges |

---

## Status vocabulary (frontend)

| Backend value | UI badge |
|---------------|----------|
| `AVAILABLE` | OPERATIONAL (green) |
| `READY` / `INFRASTRUCTURE_READY` | READY |
| `NO_DATA` | NO DATA |
| `NOT_ENABLED` | NOT ENABLED |
| `NOT_MEASURED` | NOT MEASURED |
| `CONFIGURATION_REQUIRED` / `MISSING_DEPENDENCY` | CONFIG REQUIRED |
| `UNAVAILABLE` | UNAVAILABLE |
| `NOT_IMPLEMENTED` | NOT IMPLEMENTED (purple) |

---

## Tests run (final)

| Suite | Command | Result | Mocks vs live |
|-------|---------|--------|----------------|
| Live E2E API | `py scripts/e2e_verify_integration.py` | **PASS** (incl. frontend `:5173` when dev server running) | **Real** local API |
| GAT + embed + capabilities | `py -m pytest tests/test_ecg_report_embed_api.py tests/test_gat_fusion_api.py tests/test_capability_registry_integration.py -q` | **21 passed** | TestClient + mocks / tmp deploy dir |
| GAT workflow (frontend) | `node frontend/tests/diagnostic_gat_workflow.test.js` | **passed** | Proves GAT plan uses embed + gat-fusion, not ecg evaluate |
| Hospital admin API | `py -m pytest tests/test_hospital_admin_dashboard.py -q` | **9 passed** | **SQLite in-memory** ASGI (not Postgres) |
| Frontend unit | `node frontend/tests/auth.test.js` | **5 passed** | Pure JS, no network |
| Integration status | `node frontend/tests/integration_status.test.js` | **passed** | Pure JS |
| Production build | `npm run build` (in `frontend/`) | **pass** | — |

---

## Configuration requirements (non-secret)

- PostgreSQL reachable with `POSTGRES_*` and `medshield_fl_db`.
- Alembic: `py -m alembic -c alembic.ini upgrade head` (current head: `004_alerts_dq`).
- Dev users: `py scripts/seed_postgres_dev.py --non-interactive`.
- Inference flags in `.env`: `ECG_REPORT_INFERENCE_ENABLED=true`, `FUSION_INFERENCE_ENABLED=true`.
- GAT deploy: `models/gat_ecg_report/manifest` + `best_gat_checkpoint.pt`.
- Run backend on `127.0.0.1:8000`, frontend `npm run dev` on `:5173`.

---

## Remaining blockers

1. **Hospital pytest** — Uses in-memory SQLite; does not substitute for Postgres E2E (use `e2e_verify_integration.py` for live DB).
2. **Flower live rounds** — Intentionally not implemented (`501` on start); UI must stay non-operational for live FL.
3. **GAT patient persistence** — GAT evaluate annotates optional `patient_id` only; no DB write (same as prior API contract).

---

## Files changed (integration + E2E pass)

- `backend/app/services/capability_registry.py`, `hospital_service.py`, `clinician_service.py`
- `backend/app/db/migrations/versions/004_hospital_alerts_and_data_quality.py` (revision `004_alerts_dq`)
- `scripts/seed_postgres_dev.py`, `scripts/e2e_verify_integration.py`, `scripts/db_verify_dev.py`
- `tests/test_gat_fusion_api.py`, `tests/test_hospital_admin_dashboard.py`
- `frontend/src/services/api.js`, `context/AuthContext.jsx`, `utils/integrationStatus.js`
- `ml/inference/ecg_report_runtime.py` (encoder embedding extraction)
- `backend/app/api/v1/endpoints/ecg_report.py` (`/embed`)
- `frontend/src/utils/diagnosticGatWorkflow.js`, `frontend/tests/diagnostic_gat_workflow.test.js`
- `frontend/src/components/DiagnosticWorkspace.jsx` (dual inference workflows)
- `frontend/src/pages/auth/LoginPage.jsx`, clinician/hospital dashboards, `ClinicalStates.jsx`

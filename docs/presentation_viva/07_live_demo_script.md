# Live Demo Script (5–7 minutes)

**Focus:** Patient Portal + Clinician Portal (your assigned modules).  
**Prep:** Backend + frontend running; demo patient and clinician accounts from seed (do not read passwords aloud in viva — type locally).  
**Fallbacks:** `reports/model_evaluation/model_comparison.png`, `gat_fusion_confusion_matrix.png`, screenshots of Patient Results tab.

---

## Minute 0:00–0:45 — Framing

**Say:** “MedShield-FL separates roles: patients consume authorized data and research-proxy results; clinicians run diagnostics and document workflows. AI outputs are PTB-XL proxy labels, not clinical diagnosis.”

**Show:** Login screen (`LoginPage.jsx`).

**Click:** Log in as **clinician** (seed account).

**If login fails:** Show architecture slide from `01_project_architecture.md` Mermaid; open offline evaluation PNG.

---

## Minute 0:45–2:30 — Clinician Portal

**Say:** “I'll open the clinician workspace, show assigned patients, then Diagnostic Studio where inference is triggered.”

1. **Sidebar → Clinician Workspace** — wait for overview (`GET /clinician/overview`).
2. **Assigned Patients** — pick one patient; open dossier if available.
   - **If empty:** “Eligible patients can be assigned via POST assignment; seed may pre-assign demo_patient.”
3. **Diagnostic Studio**
   - **Say:** “Patient must be selected to persist results to their portal.”
   - Load sample ECG (Normal or Pathological sample buttons).
   - Load sample report or paste short synthetic report text.
   - Optional: **Validate privacy** (mask-text) — mention sanitization pipeline.
   - Note **idempotency key** — explain repeat-click safety.
   - **Run inference** (GAT mode if enabled in capabilities).
   - **Expected:** Success panel with NORM/MI/STTC proxy class and confidence; HTTP 200.
   - **If readiness false:** Read message from readiness badges; show pre-seeded patient results instead.

4. **Patient Questions tab**
   - **Say:** “Patients submit non-urgent questions; clinicians respond here.”
   - If empty, note patient can submit from their portal.

**Do not:** Start Flower or retrain models during viva.

---

## Minute 2:30–4:30 — Patient Portal

**Say:** “Patients cannot run models; they see persisted assessments and hospital-sourced documents.”

**Logout → Login as patient.**

1. **My Health** — institution summary, capability badges.
2. **My Results** — timeline/charts if ≥1 prediction (`PatientAssessmentCharts`).
   - **Point out:** Class label from stored index; legacy rows show explicit legacy message.
   - **Confidence:** “Uncalibrated max-softmax for GAT — not the same as AUROC’s P(MI/STTC).”
3. **My Reports** — sanitized document metadata only.
4. **Questions for My Clinician** — submit a short question → **Expected 201/200**.
   - **If 500:** “Database grant issue on questions table — clinician inbox backend exists; ops fix is GRANT script.”
5. Switch back to clinician (optional 20s) — show question in inbox and reply.

---

## Minute 4:30–5:30 — Tie to system

**Say:** “End-to-end path: embed → GAT evaluate with patient_id → `diagnosis_predictions` → patient GET diagnosis. Authorization enforced server-side, not in React alone.”

**Optional 30s:** Open `reports/model_evaluation/model_evaluation_metrics.csv` — “Offline n=1838 PTB-XL test fold; AUROC ~0.965; not live API metrics.”

---

## Minute 5:30–7:00 — Q&A buffer

**If asked FL:** “Super Admin federated panel; FedAvg on GAT head over frozen embeddings — see doc 04.”

**If asked privacy:** “Masking on documents; FL demo does not export raw Postgres ECG to Flower.”

**Closing line:** “My modules demonstrate safe read-only patient experience and clinician-controlled inference with honest uncertainty and legacy data handling.”

---

## Pre-demo checklist

- [ ] `GET /api/v1/gat-fusion/readiness` and `/ecg-report/readiness` true
- [ ] Demo patient has ≥1 prediction OR clinician demo creates one
- [ ] PostgreSQL grants on `patient_clinician_questions` applied
- [ ] Browser tab zoom readable for projector
- [ ] Fallback PNGs copied to desktop

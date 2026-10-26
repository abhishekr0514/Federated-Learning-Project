# MedShield-FL — Operational Guide

> **Status**: ECG + Clinical Report inference is operational. Two-modality GAT (ECG + report embeddings) is trained and deployed under `models/gat_ecg_report/`; enable with `FUSION_INFERENCE_ENABLED=true` (see §5).

---

## 1. Architecture Overview

MedShield-FL exposes two separate inference paths:

| Path | Architecture ID | Status |
|------|----------------|--------|
| **ECG + Report** (Path A) | `medshield.ecg_report_fusion.v1` | ✅ Operational (trained checkpoint loaded) |
| **Two-Modality GAT** (Path B) | `medshield.diagnostic_net.gat_ecg_report` | ✅ Trained checkpoint deployed; inference gated by `FUSION_INFERENCE_ENABLED` |

### Path A — ECG + Clinical Report Fusion
```
12-lead ECG (12×1000 mV)
    └─► ECGEncoder (1D-CNN, 128-d embedding)
                                             ┐
Clinical Report (text string)               ├─► Concat (256-d) → Linear → Sigmoid → P(MI/STTC)
    └─► BioClinicalBERT CLS (128-d)         ┘
```
- **Research proxy task**: PTB-XL NORM (class 0) vs MI/STTC superclasses (class 1)
- **Not validated** for clinical diagnosis

### Path B — Two-Modality GAT Fusion (research architecture)
```
ECG embedding (128-d) ──► node 0 ──┐
                                   ├── fully connected 2-node graph
Clinical report (128-d) ► node 1 ──┘
        └─► linear proj (128→64) → GATConv × 2 (4 heads, residual) → mean pool → MLP → 2-class logits
```
- Embeddings come from the **frozen** ECG-report encoder (`medshield.ecg_report_fusion.v1`) on PTB-XL in-cohort pairs (`ecg_id` + machine `report` field).
- **Research proxy task**: same binary target as Path A — PTB-XL NORM (0) vs MI/STTC (1). Not clinical diagnosis.
- Legacy three-modality GAT (`medshield.diagnostic_net.gat_fusion`) remains in code for reference only; runtime rejects its manifest.

---

## 2. Data Input Contracts

### Path A — ECG + Report Evaluate Request
```json
POST /api/v1/ecg-report/evaluate
Authorization: Bearer <JWT>

{
  "ecg_leads": [[<float> × 1000] × 12],  // 12 leads × 1000 samples @ 100 Hz in mV
  "report_text": "<clinical report string>",  // min 1 char, max TEXT_MAX_LEN=256 tokens
  "patient_id": "<uuid>",   // optional — persists prediction with clinical provenance
  "idempotency_key": "<str>"  // optional — max 128 chars for safe retry
}
```

**ECG constraints**:
- Shape: exactly `(12, 1000)` — 12 leads, 1000 samples each
- Values: finite floats in millivolts (mV), no NaN or Inf
- Signal passes quality gate: per-lead amplitude range and flatline checks

**Report constraints**:
- Non-empty string, privacy-masked via regex PII scrubbing before tokenization

### Path B — GAT Fusion Evaluate Request
```json
POST /api/v1/gat-fusion/evaluate
Authorization: Bearer <JWT>

{
  "ecg_embedding":  [<float> × 128],  // from authorized ECG-report encoder
  "text_embedding": [<float> × 128],  // from authorized clinical-text encoder
  "idempotency_key": "<str>"          // optional
}
```
Both embeddings are **required**. Tabular inputs are not accepted on this path.

---

## 3. Model Artifacts

### Path A — ECG + Report Fusion
| Artifact | Path | Description |
|----------|------|-------------|
| Checkpoint | `models/ecg_report_fusion/best_checkpoint.pt` | Trained PyTorch weights (~414 MB) |
| Manifest | `models/ecg_report_fusion/manifest.json` | Architecture ID, checkpoint path, trained flag |
| Normalization | `models/ecg_report_fusion/normalization.json` | Per-lead mean/std from PTB-XL training split |
| Text encoder | HuggingFace cache | `emilyalsentzer/Bio_ClinicalBERT` (local cache) |

### Path B — GAT Fusion (ECG + report)
| Artifact | Path | Description |
|----------|------|-------------|
| Checkpoint | `models/gat_ecg_report/best_gat_checkpoint.pt` | Trained GAT fusion head (PyTorch Geometric) |
| Manifest | `models/gat_ecg_report/medshield_fusion_manifest.json` | Architecture ID, modality contract, encoder manifest pointer |
| Training run | `data/processed/gat_ecg_report/runs/20261025T182109Z/` | Metrics, splits, encoder provenance (`report.json`) |
| Encoder (frozen) | `models/ecg_report_fusion/` | Supplies 128-d ECG + 128-d text embeddings during training |

---

## 4. Training and Evaluation Commands

### Path A — ECG + Report Fusion (already trained)
```powershell
# Inspect training run metadata
Get-Content data\processed\ecg_report_fusion\runs\20261025T125307Z\report.json

# Re-train (uses PTB-XL dataset — requires ~21GB dataset download)
.\.venv\Scripts\python.exe -m ml.training.train_ecg_report_fusion --config configs\ecg_report_fusion.yaml
```

### Path B — Two-Modality GAT (PTB-XL paired ECG + report)
```powershell
# Pairing + encoder gate (must be ALLOWED before training)
py -c "from ml.fusion.ecg_report_gat_gate import evaluate_ecg_report_gat_gate; g=evaluate_ecg_report_gat_gate(); print(g)"

# Smoke test (small subset, CPU)
py -m ml.training.train_gat_ecg_report --smoke --device cpu

# Bounded research run (example: cap train records, 8 epochs, CPU)
py -m ml.training.train_gat_ecg_report --device cpu --epochs 8 --batch-size 32 --max-train 1500 --seed 42

# Full run (remove --max-train when resources allow)
py -m ml.training.train_gat_ecg_report --device cuda --epochs 20 --batch-size 64 --seed 42
```

**Latest held-out test metrics** (run `20261025T182109Z`, n=1838, patient-disjoint PTB-XL split):
- GAT: accuracy ≈ 88.0%, AUROC 0.951, F1 0.882, sensitivity 0.891, specificity 0.869
- ECG-report baseline (same embeddings, linear head): AUROC 0.946, F1 0.776, sensitivity 0.662

These are **research-proxy** results on PTB-XL only, not clinical validation.

---

## 5. PostgreSQL, Migrations, and Development Login

The backend uses **PostgreSQL only** (`postgresql+asyncpg`). Connection settings come from `POSTGRES_*` in `.env` (the assembled `DATABASE_URL` in `.env` is informational; the app rebuilds the URL from `POSTGRES_*`).

### Diagnose connectivity (no secrets printed)
```powershell
cd C:\Federated-Learning-Project
py scripts\check_postgres_connection.py
```

If you see `password authentication failed for user "medshield_admin"`, the PostgreSQL role password does not match `POSTGRES_PASSWORD` in `.env`. Fix one of:
- Update `POSTGRES_PASSWORD` in `.env` to match the existing role, **or**
- Sync the role (requires local `postgres` superuser password, never commit it):
```powershell
$env:POSTGRES_SUPERUSER_PASSWORD = Read-Host "postgres superuser password" -AsSecureString | `
  ForEach-Object { [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($_)) }
py scripts\bootstrap_postgres_dev_role.py
Remove-Item Env:POSTGRES_SUPERUSER_PASSWORD
```

### Apply schema (Alembic)
```powershell
py -m alembic upgrade head
```

### Seed development accounts (idempotent)
```powershell
py scripts\seed_postgres_dev.py
```
Creates `dr_metro@metro.org` (CLINICIAN) and `patient_alice@example.com` (PATIENT). You choose passwords interactively on first create.

### Login API contract
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/auth/login" -Method Post `
  -ContentType "application/json" `
  -Body '{"email":"dr_metro@metro.org","password":"<your-seed-password>"}'
```
Response envelope: `{ "data": { "access_token", "refresh_token", "token_type", "expires_in" } }`.

---

## 6. Configuration Flags

All flags in `.env` (or environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `ECG_REPORT_MODEL_DIR` | `models/ecg_report_fusion` | Path to ECG+Report checkpoint directory |
| `ECG_REPORT_INFERENCE_ENABLED` | `true` | Enable ECG+Report inference |
| `FUSION_MODEL_DIR` | `models/gat_ecg_report` | Path to two-modality GAT checkpoint directory |
| `FUSION_INFERENCE_ENABLED` | `false` | Set `true` only after verifying trained checkpoint + manifest |

### Safe GAT Enablement Procedure
1. Confirm `models/gat_ecg_report/best_gat_checkpoint.pt` and `medshield_fusion_manifest.json` exist (`checkpoint_trained: true` in readiness).
2. Run `py tests\verify_gat_forward_pass.py` (direct forward pass, no HTTP).
3. Set `FUSION_INFERENCE_ENABLED=true` in `.env`
4. Restart backend
5. Verify: `GET /api/v1/gat-fusion/readiness` → `ready: true`, `pairing_gate_verdict: ALLOWED`

---

## 7. Starting the Backend (PowerShell)

```powershell
# From project root
cd C:\Users\HONGIRAN\Documents\Federated-Learning-Project

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start backend (development mode with auto-reload)
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload --log-level info

# Or using the __main__ entry point
.\.venv\Scripts\python.exe -m backend.app.main
```

The backend starts on **http://127.0.0.1:8000**

Interactive API docs: http://127.0.0.1:8000/docs

---

## 7. Checking Readiness and Running Inference

### PowerShell — Health & Readiness Checks
```powershell
# Health (no auth required)
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing | ConvertFrom-Json | ConvertTo-Json -Depth 5

# ECG+Report readiness (no auth required)
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/ecg-report/readiness" -UseBasicParsing | ConvertFrom-Json | ConvertTo-Json -Depth 5

# GAT fusion readiness (no auth required)
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/gat-fusion/readiness" -UseBasicParsing | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

### Python — Direct Forward Pass Verification (no HTTP required)
```powershell
py tests\verify_forward_pass.py          # ECG + report end-to-end encoder
py tests\verify_gat_forward_pass.py      # two-modality GAT on 128-d embeddings
```

### Python — Full Integration Test (requires running backend)
```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe tests\integration_live_forward_pass.py
```

---

## 8. Example API Request/Response (Synthetic Data Only)

> ⚠️ The data below is **entirely synthetic** — generated programmatically. No real patient data is included.

### Request: ECG + Report Evaluate
```json
POST /api/v1/ecg-report/evaluate
Authorization: Bearer <your-jwt-token>
Content-Type: application/json

{
  "ecg_leads": [
    [0.05, 0.12, -0.03, 0.18, 0.02, ...],   // Lead I  (1000 samples in mV)
    [0.08, 0.14, -0.01, 0.22, 0.03, ...],   // Lead II
    ...                                       // Leads III–V6 (total: 12 leads)
  ],
  "report_text": "Normal sinus rhythm. HR 72 bpm. QRS 88ms. No ST changes. SYNTHETIC TEST ONLY."
}
```

### Response
```json
{
  "status": "EVALUATED",
  "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "patient_id": null,
  "probability": 0.312847,
  "predicted_class_index": 0,
  "predicted_class_label": "NORM",
  "target_definition": "PTB-XL proxy task: NORM (class 0) vs MI or STTC superclasses (class 1). NOT validated clinical heart-disease diagnosis.",
  "architecture_id": "medshield.ecg_report_fusion.v1",
  "model_version_tag": "ecg-report-fusion-v1-ptbxl",
  "is_research_only": true,
  "disclaimer": "EXPERIMENTAL RESEARCH PROXY OUTPUT ONLY. This experimental model is not approved, certified, or validated for clinical diagnostic, prognostic, treatment, or patient triage decision-making.",
  "provenance": "medshield.ecg_report_fusion.v1@trained_checkpoint",
  "evaluated_at": "2026-10-25T16:30:00.000000+00:00"
}
```

### Running the Full Test Suite
```powershell
# All tests excluding DB-dependent tests
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short `
  --ignore=tests/test_postgres_dev_seed.py `
  --ignore=tests/test_migrations.py `
  --ignore=tests/test_db_wiring.py `
  --ignore=tests/test_settings_database_url.py

# New GAT fusion API tests only
.\.venv\Scripts\python.exe -m pytest tests/test_gat_fusion_api.py -v

# ECG+Report inference tests only
.\.venv\Scripts\python.exe -m pytest tests/test_ecg_report_inference.py -v
```

---

## 9. Memory and Runtime Considerations (CPU-only)

| Component | Memory | First-request latency | Subsequent requests |
|-----------|--------|----------------------|---------------------|
| BioClinicalBERT tokenizer + encoder | ~420 MB RAM | ~8–15 s (model load) | ~1–3 s |
| ECGEncoder (1D-CNN) | ~5 MB RAM | included in above | <100 ms |
| Fusion classifier head | <1 MB | included | <10 ms |
| GAT model (untrained) | ~50 MB RAM | N/A — not operational | N/A |

- **CPU inference** (no CUDA): total ECG+Report latency ~1–5 seconds after warm-up
- **Warm-up**: backend eagerly loads model on startup; first request is fast after ~15s warm-up
- **Thread safety**: `threading.Lock` protects all lazy-load paths; safe for concurrent requests

---

## 10. Known Limitations and Research-Only Status

> [!CAUTION]
> **This system is EXPERIMENTAL RESEARCH ONLY. It is not approved, certified, or validated for any clinical use.**

1. **PTB-XL proxy task only**: The ECG+Report model predicts NORM vs MI/STTC on PTB-XL, not a clinically validated diagnostic
2. **Uncalibrated probabilities**: Sigmoid/softmax outputs are raw model scores, not calibrated risk estimates
3. **GAT uses frozen encoder embeddings**: GAT does not re-read raw waveforms at inference unless clients call Path A encoders first; API expects authorized 128-d vectors
4. **10-class PTB-XL label space**: Collapsed to binary for research; many clinically important classes are excluded
5. **No external validation**: Model has not been validated on external datasets beyond PTB-XL test split
6. **CPU-only**: Inference runs on CPU; GPU would reduce latency significantly

---

## 11. Data Requirements for Future Training

### To train the three-modality GAT (Path B):
1. **Dataset**: MIMIC-IV-ECG (ECG waveforms) + MIMIC-IV (clinical notes + structured data), linked by `subject_id`/`hadm_id`
2. **Credentialing**: PhysioNet credentialed access (https://physionet.org/settings/credentialing/)
3. **Linkage audit**: Update `docs/data-audit/multimodal_linkage_audit.json` with `three_modality_gat_training_readiness: CONDITIONAL_GO`
4. **Sample size**: Target ≥28,000 aligned records (ECG + note + tabular) with patient-disjoint splits
5. **Label unification**: Define a unified CVD label consistent across all three data sources
6. **Training command** (after data is in place):
   ```powershell
   .\.venv\Scripts\python.exe -m ml.training.train_gat_fusion --config configs/gat_fusion.yaml
   ```

---

## 12. API Endpoints Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | None | Infrastructure health + dynamic component status |
| GET | `/api/v1/ecg-report/readiness` | None | ECG+Report model readiness diagnostics |
| POST | `/api/v1/ecg-report/evaluate` | CLINICIAN+ | Two-modality ECG + report forward pass |
| GET | `/api/v1/gat-fusion/readiness` | None | GAT fusion pipeline readiness diagnostics |
| POST | `/api/v1/gat-fusion/evaluate` | CLINICIAN+ | Three-modality GAT forward pass (blocked until trained) |
| GET | `/api/v1/diagnosis/readiness` | None | Legacy single-modality diagnosis readiness |
| GET | `/docs` | None | Interactive Swagger UI |

---

*Last updated: 2026-10-25 | Environment: development | Trained checkpoint: ECG+Report v1 (PTB-XL)*

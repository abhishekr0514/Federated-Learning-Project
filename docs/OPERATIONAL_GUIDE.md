# MedShield-FL — Operational Guide

> **Status**: ECG + Clinical Report inference is operational. Three-modality GAT fusion is implemented but blocked pending a trained checkpoint (see §8).

---

## 1. Architecture Overview

MedShield-FL exposes two separate inference paths:

| Path | Architecture ID | Status |
|------|----------------|--------|
| **ECG + Report** (Path A) | `medshield.ecg_report_fusion.v1` | ✅ Operational (trained checkpoint loaded) |
| **Three-Modality GAT** (Path B) | `medshield.diagnostic_net.gat_fusion` | 🔴 Blocked — awaiting trained checkpoint |

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

### Path B — Three-Modality GAT Fusion (research architecture)
```
ECG embedding (128-d)          ┐
Clinical text embedding (128-d)├─► 3-node fully-connected graph
Tabular embedding (64-d)       ┘       └─► GATConv × 2 → Flatten → MLP → binary logit
```
- Requires multimodal dataset with aligned ECG + clinical text + tabular records per patient
- Data linkage gate: **BLOCKED** (no cross-cohort patient linkage available between PTB-XL, cardio_train, NBME)

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
  "ecg_embedding":     [<float> × 128],  // output of ECGEmbeddingProvider
  "text_embedding":    [<float> × 128],  // output of ClinicalTextEncodingPipeline
  "tabular_embedding": [<float> × 64],   // output of NeuralTabularEncoder
  "idempotency_key":   "<str>"           // optional
}
```
At least one embedding must be provided. Missing modalities are zero-padded + availability-masked.

### Tabular Feature Schema (`medshield.tabular.cardio_survey v1.0.0`)
10 structured features (see `ml/tabular/feature_schema.py`):
`age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, st_slope`

---

## 3. Model Artifacts

### Path A — ECG + Report Fusion
| Artifact | Path | Description |
|----------|------|-------------|
| Checkpoint | `models/ecg_report_fusion/best_checkpoint.pt` | Trained PyTorch weights (~414 MB) |
| Manifest | `models/ecg_report_fusion/manifest.json` | Architecture ID, checkpoint path, trained flag |
| Normalization | `models/ecg_report_fusion/normalization.json` | Per-lead mean/std from PTB-XL training split |
| Text encoder | HuggingFace cache | `emilyalsentzer/Bio_ClinicalBERT` (local cache) |

### Path B — GAT Fusion
| Artifact | Path | Description |
|----------|------|-------------|
| Checkpoint | `models/gat_fusion/best_checkpoint.pt` | **Does not exist yet — training required** |
| Manifest | `models/gat_fusion/medshield_fusion_manifest.json` | **Required before enabling** |

---

## 4. Training and Evaluation Commands

### Path A — ECG + Report Fusion (already trained)
```powershell
# Inspect training run metadata
Get-Content data\processed\ecg_report_fusion\runs\20261025T125307Z\report.json

# Re-train (uses PTB-XL dataset — requires ~21GB dataset download)
.\.venv\Scripts\python.exe -m ml.training.train_ecg_report_fusion --config configs\ecg_report_fusion.yaml
```

### Path B — GAT Fusion (NOT YET TRAINED — gated)
```powershell
# Check what is blocking training
.\.venv\Scripts\python.exe -c "from ml.training.linkage_gate import evaluate_three_modality_gate; r = evaluate_three_modality_gate(); print(r.to_dict())"

# Training is currently BLOCKED. Blockers:
# 1. No aligned multimodal dataset (PTB-XL ECG + clinical text + tabular per patient)
# 2. No cross-cohort patient linkage audit verdict CONDITIONAL_GO
# Required dataset: MIMIC-IV-ECG + MIMIC-IV (requires PhysioNet credentialing)
```

---

## 5. Configuration Flags

All flags in `.env` (or environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `ECG_REPORT_MODEL_DIR` | `models/ecg_report_fusion` | Path to ECG+Report checkpoint directory |
| `ECG_REPORT_INFERENCE_ENABLED` | `true` | Enable ECG+Report inference |
| `FUSION_MODEL_DIR` | `models/gat_fusion` | Path to GAT fusion checkpoint directory |
| `FUSION_INFERENCE_ENABLED` | `false` | **Keep false until trained checkpoint is placed** |

### Safe GAT Enablement Procedure
1. Place trained checkpoint in `models/gat_fusion/`
2. Create `models/gat_fusion/medshield_fusion_manifest.json` (see `models/medshield_fusion_manifest.example.json`)
3. Set `FUSION_INFERENCE_ENABLED=true` in `.env`
4. Restart backend
5. Verify: `GET /api/v1/gat-fusion/readiness` returns `ready: true`

---

## 6. Starting the Backend (PowerShell)

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
.\.venv\Scripts\python.exe tests\verify_forward_pass.py
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
3. **No multimodal alignment**: The GAT fusion requires aligned ECG + clinical text + tabular data per patient — not yet available
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

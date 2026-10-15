# Neural Tabular Encoder (Prompt 14)

Research-only component for structured survey/tabular features. **Not** production inference, **not** multimodal GAT fusion, **not** federated training.

## Schema

| Field | Value |
|-------|--------|
| **Schema ID** | `medshield.tabular.cardio_survey` |
| **Version** | `1.0.0` |
| **Code** | `ml/tabular/feature_schema.py` |

### Authoritative feature list (order = model columns 0–9)

Derived from `docs/ARCHITECTURE.md` (Structured Tabular Modality) mapped to `data/datasets/cardio_train.csv`:

| # | Feature name | Source | Type | Notes |
|---|--------------|--------|------|--------|
| 0 | `age_years` | `age` | continuous | days ÷ 365.25 |
| 1 | `gender_female` | `gender` | binary | 1=female (code 1), 0=male (code 2) |
| 2 | `ap_hi` | `ap_hi` | continuous | systolic BP mmHg |
| 3 | `ap_lo` | `ap_lo` | continuous | diastolic BP mmHg |
| 4 | `cholesterol_level` | `cholesterol` | ordinal | 1–3 survey levels |
| 5 | `glucose_level` | `gluc` | ordinal | 1–3 survey levels |
| 6 | `bmi` | `height`, `weight` | continuous | kg/m² |
| 7 | `smoke` | `smoke` | binary | 0/1 |
| 8 | `alco` | `alco` | binary | 0/1 |
| 9 | `active` | `active` | binary | 0/1 |

### Excluded columns (never model inputs)

- `id` — splitting and record tracking only (`ml/datasets/cardio_tabular.py`)
- `cardio` — binary survey target only; **not** verified clinical ground truth

Height and weight are used only to compute BMI and are not separate encoder inputs.

## Preprocessing

Module: `ml/tabular/preprocessing.py`

- **Fit**: training split only (by unique `id` before fit in full experiments).
- **Scaling**: z-score on continuous/ordinal numeric features (`age_years`, `ap_hi`, `ap_lo`, `bmi`, `cholesterol_level`, `glucose_level`).
- **Binary** features (`gender_female`, `smoke`, `alco`, `active`): passed through unchanged (0/1).
- **Missing values**: reject at row extraction; median imputation reserved for non-finite values at fit time on training columns.
- **Serialization**: JSON state includes `schema_id`, `schema_version`, means/stds, and outlier policy snapshot.
- **Validation/test**: `transform` must not update fitted parameters (`allow_update_stats` raises).

## Outlier and invalid-value policy

Audit counts (`docs/data-audit/DATASET_AUDIT_REPORT.md`, full `cardio_train.csv` pass):

- `ap_lo > ap_hi`: **1234** rows
- `ap_hi` outside 80–250: **247**
- `ap_lo` outside 40–150: **1034**
- height/weight outliers: **30** / **7**

Default policy (`OutlierPolicy`):

1. **Invalid BP pair** (`ap_lo > ap_hi`): **reject row** with `TabularRowValidationError` (no silent fix).
2. **Range flags**: documented in schema `valid_range`; optional future strict mode — default does not auto-delete plausible-format outliers.
3. **Quantile clipping**: **off** by default (`clip_to_train_quantiles=False`); when enabled, clipping uses **training** quantiles only and records policy in saved state (does not drop rows).

## Encoder architecture (Prompt 14)

Module: `ml/models/tabular_encoder.py`

```
Input [B, 10]
  → Linear(10, 64)
  → BatchNorm1d(64)
  → ReLU
  → Dropout(p=0.2)
  → Linear(64, 64)
  → ReLU
Output [B, 64]
```

Embedding dimension: **64**. Differs from `MODEL_CONTRACT.md` Block 2 (residual, Dropout 0.1) — see **ADR-008**.

## Dataset boundaries

- Standalone tabular experiment on `cardio_train.csv` only.
- **No** join to PTB-XL, NBME, or other cohorts.
- Target `cardio` is survey self-report / screening label — do **not** describe as confirmed heart-disease diagnosis.

## Tests

`tests/test_tabular_encoder.py` — synthetic rows only. Smoke forward pass is a **software** check, not performance evidence.

## Blockers / follow-ups

- Full-cohort training and calibrated reporting not run in Prompt 14.
- Upgrade encoder to full `MODEL_CONTRACT.md` tabular block before GAT fusion.
- Strict range-based row rejection for audit outliers (optional config) not enabled by default.

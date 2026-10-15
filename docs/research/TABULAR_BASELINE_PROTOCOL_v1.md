# Tabular Research Baseline Protocol v1.0.0

**Status:** Frozen for Prompt 17  
**Protocol ID:** `medshield.tabular.cardio_survey.baseline`  
**Protocol version:** `1.0.0`  
**Seed:** `42`

This is a **survey-target** research experiment on `cardio_train.csv`. It is **not** a validated clinical heart-disease diagnosis model.

## Dataset

| Field | Value |
|-------|--------|
| Path | `data/datasets/cardio_train.csv` |
| Delimiter | `;` |
| Expected SHA-256 | `21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c` |
| Source columns | `id`, `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `cardio` |

**Target:** `cardio` ∈ {0, 1} — survey-style binary outcome. Clinical validity is **not** established.

## Feature schema (unchanged)

| Field | Value |
|-------|--------|
| Schema ID | `medshield.tabular.cardio_survey` |
| Version | `1.0.0` |
| Order | `age_years`, `gender_female`, `ap_hi`, `ap_lo`, `cholesterol_level`, `glucose_level`, `bmi`, `smoke`, `alco`, `active` |

**Excluded from model inputs:** `id`, `cardio`. Height/weight used only to compute BMI.

**Outlier policy (unchanged, Prompt 14):** reject `ap_lo > ap_hi`; no silent clip; no quantile clip (`clip_to_train_quantiles=False`).

## Splits

- **Rule:** unique record `id` (one row per id in this file).
- **Seed:** 42.
- **Fractions:** train **0.70**, validation **0.15**, test **0.15** (remainder of each stratum assigned to test).
- **Stratify** by `cardio` within each class, then shuffle with seed 42.
- Every **accepted** id is in exactly one partition; partitions are disjoint.
- Rejected invalid-BP rows are **excluded** (not assigned) and counted.
- **Preprocessing** (z-score medians/stds) fitted on **training accepted rows only**; frozen for val/test.

## Model

- Encoder: `NeuralTabularEncoder` (Prompt 14) — Linear(10,64) → BN → ReLU → Dropout(0.2) → Linear(64,64) → ReLU.
- Head: `Linear(64, 1)` producing a single logit (`TabularBinaryClassifier`).
- Loss: `BCEWithLogitsLoss` (no class weights; source classes are near-balanced).
- Optimizer: Adam, lr=`1e-3`, weight_decay=`1e-4`.
- Batch size: 256. Max epochs: 20. Device: CPU or CUDA (`auto`).
- Early stopping: patience **5** on **validation AUROC**.
- Checkpoint selection: highest validation AUROC. **Test used once** after selection.

## Threshold

Selected on **validation** only: Youden’s J (`sensitivity + specificity − 1`) over thresholds `{0.05, 0.10, …, 0.95}`. Frozen for test. Default fallback 0.5 if undefined.

## Metrics (survey target)

Primary: **test AUROC**. Secondary: AUPRC, precision, F1, sensitivity, specificity, confusion matrix, Brier, ECE (10 bins).

## Artifacts

Git-ignored: `data/processed/tabular_baseline/` (checkpoints, preprocessor, run reports).  
Split counts (no raw ids) may be summarized in docs. Hashed id lists may live under gitignored `data/manifests/`.

## Boundaries

No join to PTB-XL, NBME, or MIMIC. No GAT/FL/production inference. Schema and encoder architecture frozen.

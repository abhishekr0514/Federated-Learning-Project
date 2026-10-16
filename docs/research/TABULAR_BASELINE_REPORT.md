# Tabular Baseline Research Report (Prompt 18)

**Protocol ID:** `medshield.tabular.cardio_survey.baseline`  
**Protocol Version:** `1.0.0`  
**Authoritative Dataset:** `data/datasets/cardio_train.csv` (SHA-256: `21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c`)  
**Schema ID:** `medshield.tabular.cardio_survey` (Version `1.0.0`)  
**Architecture:** `TabularBinaryClassifier` (NeuralTabularEncoder + Linear(64, 1))  

---

## 1. Executive Summary & Status

| Milestone | Status | Details |
|---|:---:|---|
| **Feature Schema Repair** | **Complete** | Corrected `gender_female` derived allowed values to `{0, 1}` and added raw gender validation in `ml/tabular/feature_schema.py`. Valid male records (`gender == 2`) are fully preserved. |
| **Model Architecture** | **Complete** | Implemented `TabularBinaryClassifier` in `ml/models/tabular_encoder.py` (Linear(10,64) → BN → ReLU → Dropout(0.2) → Linear(64,64) → ReLU → Linear(64,1) returning `(logits, embeddings)`). |
| **Cohort & Split Logic** | **Complete** | Implemented `load_and_filter_cardio_cohort` and class-stratified split (70% train, 15% val, 15% test) in `ml/datasets/cardio_tabular.py`. Hashed manifest generation under `data/manifests/`. |
| **Evaluation Metrics & Thresholding** | **Complete** | Implemented validation-only Youden's J grid search (`{0.05, 0.10, ..., 0.95}`) and AUROC/AUPRC/Brier/ECE metrics in `ml/evaluation/tabular_metrics.py`. |
| **Training Pipeline** | **Complete** | Implemented CLI entry point in `ml/training/train_tabular_baseline.py` following frozen Prompt 17 protocol. |
| **Audit Import Portability** | **Complete** | Made `ml/datasets/__init__.py` and `ml/models/__init__.py` lazy, unblocking metadata audits without requiring PyTorch. |
| **Checksum Reconciliation** | **Complete** | Reconciled canonical Git-tracked LF blob hash (`2c495cb81e6077d6…`) in `docs/data-audit/dataset_inventory.json` without modifying raw files. |
| **Unit & Protocol Tests** | **Passing** | 16/16 schema & protocol tests passing in `tests/test_tabular_schema_and_protocol.py`; 9/9 mimic audit tests passing. |
| **Real-Cohort Training Execution** | **Blocked** | Execution requires a Python environment with PyTorch. System Python is 3.14 (lacking PyTorch binary support); Python 3.12 is absent from the host. |

---

## 2. Cohort Exclusions & Class Distributions

Exact numbers calculated from `data/datasets/cardio_train.csv`:

- **Total raw rows in dataset:** 70,000
- **Excluded rows (`ap_lo > ap_hi`):** Exactly **1,234** rows (100% of exclusions).
  - Excluded males: 497
  - Excluded females: 737
- **Accepted cohort rows:** **68,766** rows (100% unique `id`s, all valid blood pressure).
  - Accepted males (`gender_female == 0.0`): **23,973** (34.86%)
  - Accepted females (`gender_female == 1.0`): **44,793** (65.14%)
  - Overall positive rate (`cardio == 1`): **49.48%** (34,024 positive, 34,742 negative)

### Stratified Partitions (Seed 42, 70/15/15)

| Partition | Total Rows | Class 0 (`cardio = 0`) | Class 1 (`cardio = 1`) | Positive Rate |
|---|---:|---:|---:|---:|
| **Train (70%)** | 48,136 | 24,319 | 23,817 | 49.48% |
| **Validation (15%)** | 10,314 | 5,211 | 5,103 | 49.48% |
| **Test (15% remainder)** | 10,316 | 5,212 | 5,104 | 49.48% |
| **Total Accepted** | **68,766** | **34,742** | **34,024** | **49.48%** |

*Partitions are strictly disjoint; every accepted ID appears in exactly one partition. Excluded IDs are excluded from all partitions.*

---

## 3. Training & Evaluation Protocol (Frozen)

- **Model:** `TabularBinaryClassifier`
- **Loss:** `nn.BCEWithLogitsLoss()` (no class weighting; cohort is balanced)
- **Optimizer:** Adam (lr=`1e-3`, weight_decay=`1e-4`)
- **Batch size:** 256
- **Epochs:** 20 max, early stopping patience 5 on validation AUROC
- **Preprocessing:** Fitted on training set only; frozen for validation and test
- **Threshold Selection:** Selected on validation set using Youden's J over `{0.05, 0.10, ..., 0.95}`; fallback 0.5; frozen for test evaluation
- **Test Evaluation:** Held-out test set evaluated exactly once using the frozen checkpoint and threshold
- **Artifacts Location:** `data/processed/tabular_baseline/runs/<timestamp>/`
  - `best_checkpoint.pt`
  - `preprocessor.json`
  - `manifest.json`
  - `report.json`

---

## 4. Execution Readiness & Smallest Supported Setup

To execute the experiment on a system with network/package management capabilities:

```bash
# 1. Create a Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. Install CPU-only PyTorch and project dependencies (minimal footprint, <500 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"

# 3. Execute the real-cohort experiment
python -m ml.training.train_tabular_baseline --data-path data/datasets/cardio_train.csv --seed 42
```

Estimated training runtime on CPU: ~2–3 minutes. Total artifact disk usage: < 5 MB.

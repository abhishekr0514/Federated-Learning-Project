# Model Evaluation Metrics and Confusion Matrices

**Authoritative artifacts (inspection):**

- `reports/model_evaluation/evaluation_manifest.json`
- `reports/model_evaluation/model_evaluation_metrics.csv`
- `reports/model_evaluation/model_evaluation_metrics.json`
- PNGs: `gat_fusion_confusion_matrix.png`, `fl_model_roc_curve.png`, `model_comparison.png`

**Generator:** `scripts/generate_model_evaluation_report.py`  
**Metric library:** `ml/evaluation/classification_metrics.py`, `ml/evaluation/gat_bundle_evaluator.py`

---

## 1. Two evaluation populations (do not merge)

| Population | n | AUROC (~) | Source | Status |
|------------|---|-----------|--------|--------|
| **Smoke / capped test** | 200 | ~0.96 (historical dashboard narrative) | Bundle built with `--max-test 200` (see `tests/test_build_gat_fl_bundle_caps.py`) | **Older run** — not in current manifest |
| **Full fold-10 test** | **1838** (911 NORM / 927 MI/STTC) | **0.9651** both models | Current `evaluation_manifest.json` `test_split_mode: "full"`, `smoke_caps.max_test: null` | **VERIFIED** in generated JSON/CSV |

**Current manifest timestamp:** `generated_at_utc`: 2026-10-27T04:14:32Z.

**Train embeddings in bundle:** 1500 (reused cache) vs **12,898** eligible train records — evaluation compares heads on **full test**; FL training bundle is **subset** for demo training cost.

---

## 2. Checkpoints evaluated (same report run)

| Model key | Checkpoint path (from manifest) |
|-----------|----------------------------------|
| `gat_centralized` | `models/gat_ecg_report/best_gat_checkpoint.pt` |
| `gat_federated_global` | Newest `global_federated_checkpoint.pt` under `data/processed/federated/gat_ecg_report_runs/` (uuid `6d5241aa-1902-434d-b0ed-b7d62263bb76` at generation time) |

**Comparable examples:** `true` — same test indices/embeddings in bundle for both heads.

---

## 3. Verified metrics (n = 1838)

From `model_evaluation_metrics.csv`:

| Model | Accuracy | Balanced acc | Macro F1 | Weighted F1 | Class-1 precision | Class-1 recall | Class-0 specificity | AUROC | AUPRC | MCC |
|-------|----------|--------------|----------|-------------|-------------------|----------------|---------------------|-------|-------|-----|
| gat_centralized | 0.8950 | 0.8951 | 0.8950 | 0.8950 | 0.9060 | 0.8835 | 0.9067 | **0.9651** | 0.9628 | 0.7903 |
| gat_federated_global | 0.8993 | 0.8992 | 0.8993 | 0.8993 | 0.8914 | 0.9115 | 0.8869 | **0.9651** | 0.9629 | 0.7988 |

---

## 4. Confusion matrices (rows = actual, cols = predicted)

Labels order: `[NORM, MI/STTC]` for both axes (`confusion_matrix_raw` in JSON).

### Centralized GAT

|  | Pred NORM | Pred MI/STTC |
|--|-----------|--------------|
| **Actual NORM** | 826 | 85 |
| **Actual MI/STTC** | 108 | 819 |

### Federated global GAT

|  | Pred NORM | Pred MI/STTC |
|--|-----------|--------------|
| **Actual NORM** | 808 | 103 |
| **Actual MI/STTC** | 82 | 845 |

**Orientation:** sklearn `confusion_matrix(y_true, y_pred, labels=[0,1])` — rows actual, columns predicted (`classification_metrics.py` lines 54–77).

---

## 5. Metric definitions (as implemented)

| Metric | Definition in code |
|--------|-------------------|
| Accuracy | `accuracy_score` |
| Balanced accuracy | `balanced_accuracy_score` |
| Macro / weighted F1 | `f1_score` with `average=macro/weighted`, `labels=[0,1]` |
| Class-1 precision / recall | MI/STTC positive class |
| Class-0 specificity | TN / (TN+FP) from cm ravel |
| AUROC | `roc_auc_score(y_true, y_score_positive)` where **`y_score_positive = softmax[:, 1]`** (P(MI/STTC)) |
| Average precision | sklearn AP on same score |
| MCC | `matthews_corrcoef` |

**Critical distinction (manifest `confidence_note`):**

- **API / patient UI confidence (GAT):** max(softmax) — uncalibrated, not necessarily P(MI/STTC).
- **AUROC in report:** uses **softmax probability of class 1** only.

**Verification:** `model_evaluation_metrics.json` `score_semantics`: `"softmax_probability_class_1_for_auroc"`.

---

## 6. ROC / PR curves

Stored as arrays in `model_evaluation_metrics.json` (`roc_curve`, `precision_recall_curve`) and plotted to PNG under `reports/model_evaluation/`.

---

## 7. What metrics prove vs do not prove

**Prove (within research setup):**

- Discrimination on **PTB-XL-derived proxy labels** (NORM vs MI/STTC) for held-out **fold 10** patients with precomputed embeddings.
- Centralized vs federated global head are **similar** AUROC on this split (~0.965).

**Do not prove:**

- Clinical validity for real hospital patients.
- Calibration of “confidence” shown in portal.
- Performance when encoders run on live noisy ECG uploads vs PTB-XL preprocessing.
- Privacy guarantees across institutions.

**Disclaimer in manifest:** `"NOT validated clinical heart-disease diagnosis."`

---

## 8. API vs static reports

`GET /api/v1/metrics/evaluation` → **501 FeatureNotImplementedError** (`metrics.py`).  
Super Admin / UI must not claim live API metrics unless reading files from disk or embedded dashboard data.

---

## 9. Tests

- `tests/test_model_evaluation_report.py` — **9 passed** (reported in prior dev; re-run optional).
- `tests/test_build_gat_fl_bundle_caps.py` — documents `--max-test 200` behavior.

**Inspection:** Full report generation previously succeeded (exit 0 per conversation summary); metrics files on disk match n=1838.

---

## 10. Viva one-liner

“We report offline evaluation on 1838 PTB-XL test-fold proxy labels with frozen embeddings; AUROC uses P(MI/STTC), while the clinician portal shows max-softmax confidence — these are intentionally different quantities.”

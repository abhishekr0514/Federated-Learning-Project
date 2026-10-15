# ECG Baseline Research Report

## Protocol readiness

| Item | Status |
|------|--------|
| Protocol document | `docs/research/ECG_BASELINE_PROTOCOL_v1.md` v1.0.0 |
| PTB-XL install | Verified Prompt 11B |
| Label mapping | DATA_CONTRACT NORM vs MI/STTC with exclusions |
| Patient-disjoint splits | PTB-XL `strat_fold` policy v1 |

## Filtered cohort (metadata-only counts)

| | Records | Patients (approx.) |
|---|--:|--:|
| Total metadata | 21,799 | 18,869 |
| **Included** | **18,400** | (see run manifest) |
| Class 0 (`{NORM}` only) | 9,069 | — |
| Class 1 (MI/STTC, no NORM) | 9,331 | — |
| Excluded | 3,399 | — |

Exclusion breakdown: 411 empty superclass; 34 NORM+pathology conflict; 2,954 other non-binary.

## Training status

Run the entry point locally (requires `[ml]` extras: `torch`, `wfdb`, `scikit-learn`):

```bash
pip install -e ".[ml]"
python -m ml.training.train_ecg_baseline --smoke   # smoke only
python -m ml.training.train_ecg_baseline           # full run (long)
```

**Metrics:** populated only in `data/processed/ecg_baseline/runs/<id>/report.json` after a run.

### Smoke run (2026-10-15, `--smoke`, 32/16/16 records — **not** a full experiment)

| Split | Val AUROC (selection) | Test AUROC |
|------|---------------------:|-----------:|
| Smoke | 0.615 | 0.571 |

Full-cohort training was **not** executed in CI/agent environment (runtime). Re-run without `--smoke` locally for definitive test-fold metrics.

## Warnings

- Proxy SCP superclasses — **not** clinical heart-disease ground truth.
- Production `model_inference` remains **not enabled**.
- Multimodal GAT fusion **not** implemented in this phase.

## Prompt 12A — Quality, gate, calibration

| Component | Status |
|-----------|--------|
| `assess_ecg_signal` | Hard checks on 12×1000, finiteness; default saturation heuristic 100 mV; flatline all-leads check **off** by default |
| `apply_ecg_quality_gate` | Research-only; no API route |
| Calibration in `report.json` | Brier + ECE (10 bins) when evaluation split valid; smoke-run estimates **not** clinically meaningful |

**Limitations:** Heuristic thresholds are not validated against clinical acquisition standards. Passing quality or calibration metrics does not prove diagnostic validity.

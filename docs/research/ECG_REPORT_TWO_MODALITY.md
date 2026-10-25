# PTB-XL two-modality ECG + report experiment

This experiment is **distinct** from the three-modality ECG + clinical text + tabular GAT (`medshield.diagnostic_net.gat_fusion`). It uses **only** PTB-XL waveforms and the PTB-XL machine `report` field linked by **`ecg_id`** on the same database row.

## Target

Binary proxy per `ml/ptbxl/labels.py`: class **0** = NORM superclass; class **1** = MI or STTC. Not a validated clinical heart-disease diagnosis.

## Cohort filters

- Same-row `ecg_id` linkage only (PTB-XL `report` column).
- Records with empty reports excluded (2 on this copy).
- Reports that fail `PrivacyMasker` validation excluded (`excluded_masking_failed`, typically ~20).

## Splits

Patient-independent via `strat_fold` (`ml/ptbxl/splits.py`). Fold **9** is unused.

## Model

- ECG: `ECGEncoder` (128-d)
- Text: frozen BioClinicalBERT + trainable projection (128-d), reports privacy-masked before tokenization
- Fusion: concat → linear head (no three-node GAT)

## Artifacts

- Runs: `data/processed/ecg_report_fusion/runs/<UTC>/`
- Deploy pointer: `models/ecg_report_fusion/manifest.json` (checkpoint path only; weights gitignored)

## Reproduce

```bash
# Smoke (CI-friendly subset)
py -m ml.training.train_ecg_report_fusion --smoke

# Bounded real run (`--max-*` takes the first N records per split in index order)
py -m ml.training.train_ecg_report_fusion --max-train 2048 --max-val 512 --max-test 512 --epochs 8

# Full-cohort run (long; GPU recommended)
py -m ml.training.train_ecg_report_fusion --epochs 12 --batch-size 16

# Verify checkpoint
py scripts/verify_ecg_report_checkpoint.py --checkpoint data/processed/ecg_report_fusion/runs/<run_id>/best_checkpoint.pt
```

## API

Three-modality `POST /api/v1/diagnosis/evaluate` must **not** load this checkpoint. A dedicated two-modality endpoint is future work.

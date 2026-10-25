# Multimodal GAT Training Protocol (Phase 3)

**Status:** **BLOCKED** — no supervised three-modality training or production checkpoint export until linkage audit permits.

**Authority:** `docs/data-audit/multimodal_linkage_audit.json` (`three_modality_gat_training_readiness.verdict`).

---

## 1. Linkage feasibility (verified)

| Modality source (local) | Labels | Linkage keys | Join to other local cohorts |
|-------------------------|--------|--------------|-----------------------------|
| PTB-XL ECG | SCP / superclass proxy | `ecg_id`, `patient_id` | **BLOCKED** vs NBME, **BLOCKED** vs cardio |
| `cardio_train.csv` | Survey `cardio` proxy | `id` only | **BLOCKED** vs PTB/NBME |
| NBME notes | Phrase-span NLP task | `pn_num`, `case_num` | **BLOCKED** vs PTB/cardio |

**Numeric overlap:** ~13k equal integers between `cardio_train.id` and PTB-XL `patient_id` — **not** linkage evidence (ADR-004).

**MIMIC-IV triad:** **BLOCKED locally** (not installed); **CONDITIONAL_GO** only after credentialed install + Prompt 16B audit.

**Conclusion:** **Zero** verified same-subject triplets (ECG 128-d + text 128-d + tabular 64-d) with a **single shared audited target** in this workspace.

---

## 2. Target definition (only if linkage passes)

When an authorized linked cohort exists, the target must be documented with:

- Explicit proxy vs clinical language (e.g. PTB-XL NSR vs pathological superclasses = **proxy**, not validated heart-disease diagnosis).
- Inclusion/exclusion rules and observation window.
- `prediction_type: proxy_evaluation` in manifest unless a separate clinical validation ADR exists.

**Current local triad:** incompatible supervision — **no unified target**.

---

## 3. Training protocol (deferred)

Requirements when `CONDITIONAL_GO`:

1. Patient-level (or documented encounter-level) train/val/test splits — no leakage.
2. Reuse `ECGEncoder`, `ClinicalTextEncodingPipeline`, `NeuralTabularEncoder` + documented preprocessing.
3. Emit **128 + 128 + 64** embeddings; full modality masks per `FusionModelInput`.
4. Train `MedShieldDiagnosticNet` only on eligible linked rows.
5. Single held-out test evaluation after protocol freeze.
6. Export via `save_fusion_checkpoint()` with `trained: true` only after completed run.

**Entry point (gate only today):**

```bash
py -m ml.training.train_multimodal_fusion --check-linkage-only
```

Exit code **2** = blocked (expected now).

---

## 4. Checkpoint export

After a real training run:

1. `save_fusion_checkpoint(path, model, extra_metadata={...})` with `FusionModelConfig(trained=True)`.
2. Copy/adapt `models/medshield_fusion_manifest.example.json` → `models/medshield_fusion_manifest.json`.
3. Verify: `py scripts/verify_fusion_checkpoint.py`
4. Keep `INFERENCE_ENABLED=false` until operational review.

---

## 5. Smallest next step

Complete PhysioNet credentialing → local MIMIC install → `ml/datasets/mimic_linkage_audit.py` → update audit JSON → only then implement training loop in `train_multimodal_fusion.py`.

Do **not** train on PTB-XL + NBME + cardio cross joins.

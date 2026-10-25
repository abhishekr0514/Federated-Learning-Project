# Federated Privacy Audit — Tabular Cardio Experiment

**Scope:** `medshield.fl.tabular_cardio.v1` small simulated run (3 clients × 2 rounds).  
**Date context:** Post first real tabular FL experiment (`20261025T053718Z`).  
**Not claimed:** Differential privacy, secure aggregation, multi-hospital clinical validation.

## 1. Data flows traced

| Path | Payload | Sensitive content |
|------|---------|-------------------|
| Client → aggregator | `FitRes`: ndarray list (model weights), `num_examples`, metrics dict | **No raw rows**; weights are full model state (inversion risk — see residuals) |
| Aggregator → client | Global `Parameters` (ndarrays) | Same as above |
| Client metrics | `loss` only (allowlisted in `FederationConfig.allowed_client_metrics`) | Scalar aggregate |
| Client properties | `federation_id`, `task_id`, `architecture_id`, `architecture_version` | Non-PHI contract metadata |
| Checkpoint file | `metadata` + `parameter_names` + `global_state_ndarrays` | No patient rows; no direct record ids |
| `report.json` (local artifact) | Config, round history, evaluation, **hashed record-id lists per client** | See residual risks |
| Backend API (`GET /experiments/tabular/latest`) | Sanitized summary | **Hash lists stripped** |

**Implementation references:**

- `fl/client/medshield_numpy_client.py` — validates parameters; metrics allowlist.
- `fl/adapters/tabular.py` — local `DataLoader` only on client; returns weights + `loss`.
- `fl/training/tabular_federated.py` — manual FedAvg loop; no CSV serialization.
- `fl/privacy.py` — forbidden metadata keys; DP disclaimer.
- `fl/checkpoints.py` — `assert_safe_metadata_dict`; compatibility checks on load.

## 2. Verification checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Raw rows not transmitted | **Pass** | Features never leave client except via weight updates |
| Direct identifiers not in FL messages | **Pass** | No `id`/`patient_id` in Flower payloads |
| Patient-level predictions not sent | **Pass** | Evaluation runs server-side locally after FL in runner, not per-client upload |
| Only approved metrics | **Pass** | `loss` only; non-allowlisted keys raise `ParameterValidationError` |
| Partition metadata in API | **Pass** | Counts only via `federated_experiment_service._sanitize_partition_plan` |
| Logs expose sensitive data | **Pass** | FL modules do not log raw text/rows; runner prints JSON summaries |
| Checkpoint load validation | **Pass** | `load_federated_checkpoint` + schema signature; corrupt file → `CheckpointError` |
| Malformed updates rejected | **Pass** | `validate_ndarray_update`, `ValidatedFedAvg`, weighted FedAvg |

## 3. Residual risks (honest)

1. **Model parameter leakage:** Standard FedAvg weight vectors can enable membership or property inference attacks in general — **not mitigated** in this codebase.
2. **Hashed record IDs on disk:** `partition_plan.json` and full `report.json` include SHA-256(`str(record_id)`) per client. These are **pseudonymous linkage aids**, not raw ids, but remain in **gitignored** artifacts. API responses **omit** hash lists.
3. **Sample counts:** `num_examples` per client reveals partition sizes (800/800/800 in smoke config) — acceptable for research simulation.
4. **Confusion matrices in `report.json`:** Aggregate test counts only — no per-patient predictions stored.
5. **Backend `differential_privacy` flag** in `/privacy/status` is a **product config label**, not evidence that FL rounds apply DP noise.

## 4. Reproducibility verification

Use:

```bash
py scripts/verify_tabular_federated_run.py --run-dir data/processed/federated/tabular_cardio_runs/20261025T053718Z
```

Confirms: 11 tensors (TabularBinaryClassifier), `trained=true`, round 2, schema signature match, parameter L2 change.

Re-run experiment (explicit):

```bash
py -m fl.runner run-tabular-experiment configs/federations/tabular_cardio_v1_real_smoke.json --confirm-run
```

Same config + seed → same partitions; floating-point details may vary slightly by device (CPU vs CUDA).

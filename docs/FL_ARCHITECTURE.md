# MedShield-FL Flower Federated Learning Architecture (Prompt 20)

## Readiness levels

| Level | Meaning | Prompt 20 status |
|-------|---------|------------------|
| **1 — Infrastructure** | Client/server lifecycle, parameter validation, FedAvg, checkpoints, runner | **Implemented** |
| **2 — Task experiment** | Compatible architecture + schema + target on real local data per client | **Prepared** (tabular first); **not executed** at full cohort |
| **3 — Clinical multimodal FL** | Legitimate linked multimodal records + validated evaluation | **Blocked** (GAT) |

Level 1 does **not** imply levels 2 or 3.

## Flower version and API

- Declared dependency: `flwr[simulation]>=1.7.0` (`pyproject.toml` optional extra `fl`).
- Verified API surface: `flwr.client.NumPyClient`, `flwr.server.strategy.FedAvg`, `flwr.common.ndarrays_to_parameters` / `parameters_to_ndarrays`.
- Installed version during implementation: **1.42.0** (NumPyClient under `flwr.compat.client.numpy_client`).
- Ray-based `flwr.simulation.start_simulation` is **deprecated**; Prompt 20 uses a **local synthetic round** in `fl.runner` plus `ValidatedFedAvg` strategy construction — not full Ray multi-client simulation.

## Dataset boundaries (do not merge)

| Dataset | Task | Federation ID |
|---------|------|----------------|
| `cardio_train.csv` | Survey `cardio` binary | `medshield.fl.tabular_cardio.v1` |
| PTB-XL | ECG proxy superclass | `medshield.fl.ecg_ptbxl.v1` |
| NBME | Concept spans | **Blocked** (`medshield.fl.nbme_concepts.v1`) |
| Multimodal GAT | Linked ECG+text+tabular | **Blocked** (`medshield.fl.gat_multimodal.v1`) |

## Client responsibilities

Implemented via `TaskClientAdapter` and `MedShieldNumPyClient`:

- Local model + local data only (when training enabled in future).
- Returns **parameters** and allowlisted metrics (`loss` only by default).
- Sample counts for weighted FedAvg.
- Fail-closed on schema/shape/non-finite parameters.
- **No** raw rows, ECG, notes, identifiers, or gradients in messages.

## Server responsibilities

- `ValidatedFedAvg` extends Flower `FedAvg` and validates each `FitRes` before calling `fl.aggregation.fedavg.weighted_fedavg`.
- Rejects incompatible architectures, malformed tensors, zero sample counts, and (by default) client failures.
- Checkpoints via `fl.checkpoints` (metadata + ndarray list only).

## Parameter contract

- Stable **sorted** state-dict keys → ndarray list (`fl.parameters.serialize`).
- `ParameterSchema.signature()` hashes names/shapes/dtypes for checkpoint compatibility.

## Privacy

See `fl.privacy.PRIVACY_LIMITATIONS`. **FedAvg is not differential privacy.** Backend `DP_*` settings are not applied automatically in this infrastructure phase.

## Checkpoint format

Path (gitignored): `data/processed/federated/`

Payload keys: `metadata`, `parameter_names`, `global_state_ndarrays`.  
`trained=false` for infrastructure smoke; never mark random init as trained.

## Runner (disabled by default)

```bash
py -m fl.runner validate-config configs/federations/tabular_cardio_v1.example.json
py -m fl.runner readiness --config configs/federations/tabular_cardio_v1.example.json
py -m fl.runner simulate configs/federations/tabular_cardio_v1.example.json --confirm-run --synthetic-infrastructure
```

- `readiness` and `validate-config` **never** start training.
- `simulate` requires `--confirm-run` and `--synthetic-infrastructure`; performs **one synthetic round** only.

## Simulated vs real hospitals

Local synthetic clients or future partitioned `cardio_train.csv` slices are **simulation aids**, not evidence of multi-hospital clinical FL validation.

## Privacy audit and demonstration

- Audit: `docs/research/FEDERATED_PRIVACY_AUDIT.md`
- Demo workflow: `docs/research/TABULAR_FL_DEMONSTRATION.md`
- Script: `scripts/demonstrate_tabular_fl.ps1` (no implicit training)
- Verify run: `scripts/verify_tabular_federated_run.py`
- API: `GET /api/v1/federated/experiments/tabular/latest` (requires `federated:monitor`)

## Real tabular experiment (small)

```bash
py -m fl.runner run-tabular-experiment configs/federations/tabular_cardio_v1_real_smoke.json --confirm-run
```

- Caps federated train rows (`max_federated_train_rows`, default 2400 in smoke config).
- 3 simulated clients, disjoint slices of protocol **train** split; val/test unchanged for evaluation.
- Checkpoints: `data/processed/federated/tabular_cardio_runs/<UTC>/global_federated_checkpoint.pt` (`trained=true` after real local epochs).

## Deferred

- Full-cohort federated training on all train rows.
- Ray/`flwr run` production simulation.
- Network FL server with TLS (use existing security settings when exposing boundaries).
- GAT / NBME federations until tasks are legitimate and implemented.

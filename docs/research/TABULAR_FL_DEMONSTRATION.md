# Tabular Federated Learning — Demonstration Workflow

Simulated clients on `cardio_train.csv` — **not** independent hospitals.  
FedAvg only — **not** differential privacy.

## Prerequisites

- `data/datasets/cardio_train.csv` (expected SHA-256 in config)
- `pip install -e ".[fl,ml]"` (PyTorch + flwr)
- No step runs training unless `--confirm-run` is passed

## 1. Validate configuration

```powershell
py -m fl.runner validate-config configs/federations/tabular_cardio_v1_real_smoke.json
```

## 2. Readiness (no training)

```powershell
py -m fl.runner readiness --config configs/federations/tabular_cardio_v1_real_smoke.json
```

## 3. Run small federated experiment (explicit)

```powershell
py -m fl.runner run-tabular-experiment configs/federations/tabular_cardio_v1_real_smoke.json --confirm-run
```

**Default smoke config:** 3 clients, 2 rounds, 2 local epochs, max 2400 train rows.

## 4. Verify saved runs

```powershell
py scripts/verify_tabular_federated_run.py
py scripts/verify_ecg_baseline_run.py
```

ECG baseline is **ECG-only** (`data/processed/ecg_baseline/`) — separate from ECG+report and GAT checkpoints.

## 4b. Admin UI

Super Admin → **Federated Engine** tab loads sanitized results from `GET /api/v1/federated/experiments/tabular/latest` (requires `federated:monitor`).

## 5. Inspect artifacts (local, gitignored)

| File | Purpose |
|------|---------|
| `data/processed/federated/tabular_cardio_runs/<UTC>/report.json` | Rounds, metrics, evaluation |
| `.../global_federated_checkpoint.pt` | Global weights + metadata |
| `.../partition_plan.json` | Client counts + hashed id lists (not for public demo) |
| `.../preprocessor.json` | Frozen train-fit preprocessing |

## 6. API summary (authenticated)

Requires `federated:monitor` (Super Admin role in default RBAC):

```http
GET /api/v1/federated/experiments/tabular/latest
Authorization: Bearer <token>
```

Optional: `?run_id=20261025T053718Z`

Returns sanitized JSON — no raw rows or hash lists.

## 7. Tests

```powershell
py -m pytest tests/test_tabular_federated.py tests/test_fl_infrastructure.py tests/test_federated_experiment_api.py -q
```

## Live multi-site demonstration (remaining work)

- Real siloed data per site (not slices of one CSV)
- TLS + authentication on Flower transport
- Per-site preprocessing policy alignment
- Operational monitoring without logging weights
- Legal/consent and IRB where applicable

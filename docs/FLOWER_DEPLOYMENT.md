# Flower federated deployment (MedShield-FL)

## Architecture

- **Flower 1.30** — SuperLink (`flower-superlink`) + independent SuperNodes (`flower-supernode`) + `ServerApp` / `ClientApp` in `fl/flower_deploy/`.
- **FastAPI** — authenticated control plane (`/api/v1/federated/*`); does not perform in-process FedAvg when `FLOWER_DEPLOYMENT_MODE=superlink`.
- **Task (dev integration)** — `SyntheticInfrastructureAdapter` only (non-clinical linear model). No ECG waveforms, reports, or patient rows on the wire.
- **GAT inference** — unchanged; separate centralized training under `ml/training/train_gat_ecg_report.py`. Optional GAT-head FL experiments: `python -m fl.runner run-ecg-report-gat-head-experiment …` (does not alter inference checkpoints).

Production defaults remain **`FLOWER_ORCHESTRATION_ENABLED=false`** and **`FLOWER_TRAINING_ENABLED=false`**.

## Port topology (Flower 1.30, local dev)

| Port | API | TLS | Used by |
|------|-----|-----|---------|
| **9093** | Control API | yes (`ca.pem`) | `flwr supernode register`, `flwr run`, supervisor readiness |
| **9092** | Fleet API (gRPC-rere) | yes | `flower-supernode --superlink` |
| **9091** | ServerAppIo | (SuperLink internal) | SuperLink → `flwr-serverapp` |
| **9094 / 9095** | ClientAppIo | no (loopback) | SuperNode 0 / 1 |

`~/.flwr/config.toml` connection **`medshield-dev-tls`** must point at **Control (9093)**, not Fleet.

## Troubleshooting

- **`FAILED_PRECONDITION: SuperNode … could not be activated`** — Flower only activates nodes in `REGISTERED`/`OFFLINE`. A **stale `flower-superlink`** still bound to 9092/9093 (often with an old `--database`) leaves the same public keys stuck **`ONLINE`**. Kill listeners on 9092–9095 before starting a new run (`FlowerSupervisor` calls `ensure_ports_free` automatically).
- **`flwr run` exits 0 but no training** — Without `--stream`, the CLI returns after `StartRun`. The supervisor uses **`flwr run --stream`** and checks `live_supervisor_state.json`.
- **FAB / ServerApp** — The app under `fl/flower_deploy/` is **self-contained** (no imports from repo-root `fl.*`); SuperLink installs the FAB under `~/.flwr/apps/`. Set **`MEDSHIELD_FL_STATE_PATH`** (supervisor sets this on all child processes) so run evidence lands in `data/processed/federated/live_supervisor_state.json`.

## Phase 1 reconnaissance (summary)

| Area | Finding |
|------|---------|
| Prior orchestration | In-process thread + fake “registered clients” |
| GAT for FL | Trainable on frozen 128+128 embeddings; `federated_trainable_scope=gat_head` in `fl/config.py` — use explicit runner, not SuperLink smoke |
| flwr | **1.30.0**, ServerApp/ClientApp/SuperLink API (no legacy `start_server`) |
| Security | SuperNode EC private keys + TLS on Fleet/Control APIs |

## Dev TLS material (Windows PowerShell)

```powershell
Set-Location C:\Federated-Learning-Project
.\scripts\flower\generate_dev_materials.ps1
```

Outputs under `deployment/flower/dev/` (gitignored): `ca.pem`, `server.pem`, `server.key`, `client0`/`client1` keys and `.pub`.

## Manual run (two SuperNodes, TLS, auth)

1. **SuperLink** (terminal 1):

```powershell
flower-superlink `
  --ssl-certfile deployment/flower/dev/server.pem `
  --ssl-keyfile deployment/flower/dev/server.key `
  --ssl-ca-certfile deployment/flower/dev/ca.pem `
  --enable-supernode-auth `
  --control-api-address 127.0.0.1:9093 `
  --fleet-api-address 127.0.0.1:9092
```

2. **Register keys** (from `fl/flower_deploy`, after Flower writes `~/.flwr/config.toml` superlink entry — or use API supervisor):

```powershell
cd fl/flower_deploy
flwr supernode register ..\..\deployment\flower\dev\client0.pub medshield-dev-tls
flwr supernode register ..\..\deployment\flower\dev\client1.pub medshield-dev-tls
```

3. **SuperNodes** (terminals 2–3):

```powershell
flower-supernode --superlink 127.0.0.1:9092 --root-certificates deployment/flower/dev/ca.pem `
  --auth-supernode-private-key deployment/flower/dev/client0 `
  --clientappio-api-address 127.0.0.1:9094 `
  --node-config "partition-id=0 num-partitions=2"

flower-supernode --superlink 127.0.0.1:9092 --root-certificates deployment/flower/dev/ca.pem `
  --auth-supernode-private-key deployment/flower/dev/client1 `
  --clientappio-api-address 127.0.0.1:9095 `
  --node-config "partition-id=1 num-partitions=2"
```

4. **Submit run**:

```powershell
cd fl/flower_deploy
$env:PYTHONPATH = "C:\Federated-Learning-Project"
flwr run . medshield-dev-tls --stream --run-config num-server-rounds=2 local-epochs=1
```

5. **Status** — `data/processed/federated/live_supervisor_state.json` or `GET /api/v1/federated/status` (requires `federated:monitor`).

## API control plane

Enable in `.env` (development only — **do not** commit production with these true):

```env
FLOWER_ORCHESTRATION_ENABLED=true
FLOWER_TRAINING_ENABLED=true
FLOWER_DEPLOYMENT_MODE=superlink
FLOWER_REQUIRE_TLS=true
FLOWER_NUM_ROUNDS=2
```

Restart the FastAPI process after changing flags. The UI **Federated Engine** tab reads `GET /api/v1/federated/status` and shows demo nodes from SuperLink (not the legacy tabular experiment API).

**RBAC:** All `/api/v1/federated/*` orchestration routes require `federated:monitor`. Use the seeded Super Admin **`admin@medshield.org`** (default password `ValidPassword123!` when seeded with `--non-interactive`). Clinician and Hospital Admin accounts are rejected with HTTP 403 by design. If an existing DB user was created before `is_super_admin` was set, re-run `python scripts/seed_postgres_dev.py --non-interactive` to sync flags.

`POST /api/v1/federated/start` starts SuperLink + SuperNodes + `flwr run` via `FlowerSupervisor` (Super Admin / `federated:monitor`).

## E2E test

```powershell
$env:FLOWER_E2E = "1"
python -m pytest tests/test_flower_superlink_e2e.py -v
```

## Limitations

- Local multi-process on one machine demonstrates **protocol + aggregation**, not multi-hospital isolation.
- **No** secure aggregation or differential privacy on this path.
- DoWhy/DiCE/clinician inference paths are unaffected.

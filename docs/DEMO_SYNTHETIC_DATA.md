# Synthetic demo data (local research only)

All records created by `scripts/seed_demo_synthetic.py` are labelled **[SYNTHETIC]** and must not be treated as real patients or hospitals.

## Command

```bash
python scripts/seed_demo_synthetic.py --non-interactive
```

Requires PostgreSQL database `medshield_fl_db` (see `.env` `DATABASE_URL`). Password: `DemoSeed2026!` or set `MEDSHIELD_DEMO_SEED_PASSWORD`.

## What it creates (idempotent)

| Entity | Count | Notes |
|--------|------:|-------|
| Hospitals | 2 | `DEMO-METRO`, `DEMO-RIVERSIDE` |
| Patients + portal users | 15 | `demo_patient_01@demo.org` … `demo_patient_15@demo.org` |
| Clinicians | 2 | One per demo hospital |
| Clinical metadata per patient | 3 | Document ref, ECG ref, observation ref (local URIs only) |

Does **not** insert `DiagnosisPrediction` rows or fabricated AUROC. Audit rows appear when users log in and use the portal (real `AuditWriter` events).

## Base dev seed (single hospital)

```bash
python scripts/seed_postgres_dev.py --non-interactive
```

Creates **Metro General Hospital** (`HOSP-METRO`), `patient_alice@example.com`, and related dev accounts.

If you use **both** seeds on the same database, run `seed_demo_synthetic.py` after the postgres dev seed.
It realigns `dr_metro@metro.org` to the synthetic **`DEMO-METRO`** hospital so Diagnostic Studio lists
`DEMO-SYNTH-PATIENT-*` assignments. One-off repair SQL: `scripts/repair_demo_clinician_hospital_alignment.sql`.

## Cleanup

See SQL block at the top of `scripts/seed_demo_synthetic.py` (dev only).

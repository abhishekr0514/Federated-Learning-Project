"""Grant medshield_admin access to patient_clinician_questions (superuser required once)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402
from backend.app.db.connectivity import database_target_summary  # noqa: E402


def _find_psql() -> Path | None:
    for candidate in (
        Path(r"C:\Program Files\PostgreSQL\18\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\17\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\16\bin\psql.exe"),
    ):
        if candidate.is_file():
            return candidate
    found = shutil.which("psql")
    return Path(found) if found else None


def main() -> int:
    super_password = os.environ.get("POSTGRES_SUPERUSER_PASSWORD")
    if not super_password:
        print(
            "Set POSTGRES_SUPERUSER_PASSWORD to the local postgres superuser password, then re-run:\n"
            "  py scripts/apply_patient_questions_grants.py"
        )
        return 1

    psql = _find_psql()
    if psql is None:
        print("psql not found.")
        return 1

    target = database_target_summary()
    app_user = settings.POSTGRES_USER
    db = str(target["database"])
    grant_sql = (
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.patient_clinician_questions TO {app_user};"
    )

    env = os.environ.copy()
    env["PGPASSWORD"] = super_password
    proc = subprocess.run(
        [
            str(psql),
            "-h",
            str(target["host"]),
            "-p",
            str(target["port"]),
            "-U",
            "postgres",
            "-d",
            db,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            grant_sql,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("Grant failed (table missing or wrong superuser password).")
        if proc.stderr:
            print(proc.stderr.strip().split("\n")[0][:240])
        return proc.returncode

    print(f"Granted patient_clinician_questions privileges to {app_user} on {db}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

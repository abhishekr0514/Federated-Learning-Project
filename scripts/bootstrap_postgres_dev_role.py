"""
Create or sync the development PostgreSQL role/database to match POSTGRES_* in .env.

Requires the PostgreSQL superuser password in environment variable POSTGRES_SUPERUSER_PASSWORD
(never commit this value). Uses psql when available; does not print secrets.
"""

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
from backend.app.db.connectivity import database_target_summary, verify_async_connection  # noqa: E402


def _find_psql() -> Path | None:
    for candidate in (
        Path(r"C:\Program Files\PostgreSQL\18\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\17\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\16\bin\psql.exe"),
    ):
        if candidate.is_file():
            return candidate
    return shutil.which("psql") and Path(shutil.which("psql"))  # type: ignore[arg-type]


def main() -> int:
    super_password = os.environ.get("POSTGRES_SUPERUSER_PASSWORD")
    if not super_password:
        print(
            "Set POSTGRES_SUPERUSER_PASSWORD to the local PostgreSQL superuser (postgres) password, "
            "then re-run this script."
        )
        return 1

    psql = _find_psql()
    if psql is None:
        print("psql not found. Install PostgreSQL client tools or add psql to PATH.")
        return 1

    target = database_target_summary()
    user = str(target["username"])
    db = str(target["database"])
    app_password = settings.POSTGRES_PASSWORD

    pwd_sql = app_password.replace("'", "''")
    role_sql = (
        f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{user}') THEN "
        f"CREATE ROLE {user} WITH LOGIN PASSWORD '{pwd_sql}'; "
        f"ELSE ALTER ROLE {user} WITH LOGIN PASSWORD '{pwd_sql}'; END IF; END $$;"
    )
    db_sql = f"CREATE DATABASE {db} OWNER {user};"

    env = os.environ.copy()
    env["PGPASSWORD"] = super_password
    base_cmd = [
        str(psql),
        "-h",
        str(target["host"]),
        "-p",
        str(target["port"]),
        "-U",
        "postgres",
        "-d",
        "postgres",
        "-v",
        "ON_ERROR_STOP=1",
    ]
    proc = subprocess.run([*base_cmd, "-c", role_sql], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Bootstrap failed while creating/updating application role.")
        if proc.stderr:
            print(proc.stderr.strip().split("\n")[0][:200])
        return proc.returncode

    proc_db = subprocess.run([*base_cmd, "-c", db_sql], env=env, capture_output=True, text=True)
    if proc_db.returncode != 0 and "already exists" not in (proc_db.stderr or "").lower():
        print("Bootstrap failed while creating database (may already exist).")
        if proc_db.stderr:
            print(proc_db.stderr.strip().split("\n")[0][:200])
        return proc_db.returncode

    proc = subprocess.run(
        [*base_cmd, "-c", f"GRANT ALL PRIVILEGES ON DATABASE {db} TO {user};"],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("Bootstrap failed (check superuser password and PostgreSQL service).")
        if proc.stderr:
            print(proc.stderr.strip().split("\n")[0][:200])
        return proc.returncode

    import asyncio

    result = asyncio.run(verify_async_connection())
    if result.ok:
        print(f"Application role can connect to {user}@{target['host']}:{target['port']}/{db}")
        print("Next: py -m alembic upgrade head")
        print("Then: py scripts/seed_postgres_dev.py")
        return 0
    print(f"Role updated but app connection still failed: {result.error_type}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify application PostgreSQL settings can connect (no secrets printed)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.db.connectivity import (  # noqa: E402
    assert_postgres_configuration,
    database_target_summary,
    verify_async_connection,
)


async def main() -> int:
    assert_postgres_configuration()
    target = database_target_summary()
    print(f"Checking PostgreSQL connectivity to {target['username']}@{target['host']}:{target['port']}/{target['database']}")
    result = await verify_async_connection()
    if result.ok:
        print("Connection: OK")
        return 0
    print(f"Connection: FAILED ({result.error_type})")
    print(
        "Hint: set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, and POSTGRES_DB "
        f"in {_ENV_HINT} to match your local PostgreSQL role."
    )
    return 1


_ENV_HINT = str(REPO_ROOT / ".env")

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

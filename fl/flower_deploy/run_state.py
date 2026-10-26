"""Shared run-state file for Flower ServerApp and FastAPI supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_state_path() -> Path:
    override = os.environ.get("MEDSHIELD_FL_STATE_PATH")
    if override:
        return Path(override)
    return _repo_root() / "data" / "processed" / "federated" / "live_supervisor_state.json"


def parameter_vector_hash(arrays: list) -> str:
    import numpy as np

    h = hashlib.sha256()
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        h.update(a.tobytes())
        h.update(str(a.shape).encode())
    return h.hexdigest()[:16]


def write_state(payload: dict[str, Any], *, path: Path | str | None = None) -> None:
    target = Path(path) if path else default_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        prior: dict[str, Any] = {}
        if target.is_file():
            try:
                prior = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = {}
        body = {
            **prior,
            **payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        target.write_text(json.dumps(body, indent=2), encoding="utf-8")


def read_state(*, path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path else default_state_path()
    if not target.is_file():
        return {}
    with _LOCK:
        return json.loads(target.read_text(encoding="utf-8"))

"""Process/port helpers for Flower SuperLink deployment on Windows and POSIX."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def flwr_executable() -> str:
    """Resolve the same `flwr` CLI the user installed with the active interpreter."""
    import shutil

    found = shutil.which("flwr")
    if found:
        return found
    scripts = Path(sys.executable).parent / ("flwr.exe" if sys.platform == "win32" else "flwr")
    if scripts.is_file():
        return str(scripts)
    return "flwr"


def flower_superlink_executable() -> str:
    import shutil

    found = shutil.which("flower-superlink")
    if found:
        return found
    scripts = Path(sys.executable).parent / (
        "flower-superlink.exe" if sys.platform == "win32" else "flower-superlink"
    )
    if scripts.is_file():
        return str(scripts)
    return "flower-superlink"


def flower_supernode_executable() -> str:
    import shutil

    found = shutil.which("flower-supernode")
    if found:
        return found
    scripts = Path(sys.executable).parent / (
        "flower-supernode.exe" if sys.platform == "win32" else "flower-supernode"
    )
    if scripts.is_file():
        return str(scripts)
    return "flower-supernode"


def pids_listening_on_ports(ports: Iterable[int]) -> set[int]:
    port_set = {int(p) for p in ports}
    pids: set[int] = set()
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError):
            return pids
        for line in out.splitlines():
            if "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local = parts[1]
            for port in port_set:
                if local.endswith(f":{port}"):
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        return pids

    try:
        out = subprocess.check_output(["ss", "-ltnp"], text=True, encoding="utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return pids
    for line in out.splitlines():
        for port in port_set:
            if f":{port}" in line and "pid=" in line:
                frag = line.split("pid=")[1].split(",")[0]
                try:
                    pids.add(int(frag))
                except ValueError:
                    pass
    return pids


def terminate_pids(pids: Iterable[int], *, grace_seconds: float = 5.0) -> None:
    for pid in set(pids):
        if pid <= 0:
            continue
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                subprocess.run(["kill", "-TERM", str(pid)], capture_output=True, check=False)
        except OSError as exc:
            logger.warning("Failed to terminate pid=%s: %s", pid, exc)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not pids_listening_on_ports([]):
            break
        time.sleep(0.2)


def ensure_ports_free(ports: Iterable[int], *, timeout: float = 30.0) -> None:
    ports = list(ports)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = pids_listening_on_ports(ports)
        if not pids:
            return
        logger.warning("Freeing Flower ports %s (pids=%s)", ports, sorted(pids))
        terminate_pids(pids)
        time.sleep(1.0)
    remaining = pids_listening_on_ports(ports)
    if remaining:
        raise RuntimeError(f"Ports {ports} still in use by PIDs {sorted(remaining)}")

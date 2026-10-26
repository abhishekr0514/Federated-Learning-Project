"""Manage real Flower SuperLink + SuperNode processes (control plane only)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.services.flower_process_utils import (
    ensure_ports_free,
    flwr_executable,
    flower_superlink_executable,
    flower_supernode_executable,
    pids_listening_on_ports,
    safe_diagnostic_text,
    utf8_subprocess_env,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APP_DIR = _REPO_ROOT / settings.FLOWER_APP_DIR
_CERT_DIR = _REPO_ROOT / settings.FLOWER_TLS_CERT_DIR


@dataclass
class _Proc:
    name: str
    popen: subprocess.Popen
    started_at: float = field(default_factory=time.time)


class FlowerSupervisor:
    _lock = threading.Lock()
    _processes: List[_Proc] = []
    _log_dir = _REPO_ROOT / "data" / "processed" / "federated" / "logs"
    _run_thread: Optional[threading.Thread] = None
    _lifecycle: str = "SERVER_STOPPED"
    _run_id: Optional[str] = None
    _last_error: Optional[str] = None
    _flwr_run_proc: Optional[subprocess.Popen] = None

    @classmethod
    def _cert_paths(cls) -> dict[str, Path]:
        return {
            "ca": _CERT_DIR / "ca.pem",
            "server_cert": _CERT_DIR / "server.pem",
            "server_key": _CERT_DIR / "server.key",
            "client0_key": _CERT_DIR / "client0",
            "client1_key": _CERT_DIR / "client1",
            "client0_pub": _CERT_DIR / "client0.pub",
            "client1_pub": _CERT_DIR / "client1.pub",
        }

    @classmethod
    def security_readiness(cls) -> tuple[bool, list[str]]:
        blockers: list[str] = []
        if not settings.FLOWER_REQUIRE_TLS:
            blockers.append("FLOWER_REQUIRE_TLS must be true for SuperLink deployment")
        paths = cls._cert_paths()
        for label, path in paths.items():
            if not path.is_file():
                blockers.append(f"Missing TLS/auth material: {path.relative_to(_REPO_ROOT)}")
        return len(blockers) == 0, blockers

    @staticmethod
    def _base_env() -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(_APP_DIR) + os.pathsep + str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        )
        env["MEDSHIELD_FL_STATE_PATH"] = str(
            _REPO_ROOT / "data" / "processed" / "federated" / "live_supervisor_state.json"
        )
        from ml.federated.gat_ecg_report_fl import default_bundle_path

        env["MEDSHIELD_GAT_FL_BUNDLE"] = str(default_bundle_path())
        return utf8_subprocess_env(env)

    @staticmethod
    def _safe_exc_message(exc: BaseException) -> str:
        return safe_diagnostic_text(f"{type(exc).__name__}: {exc}")

    @classmethod
    def _register_output_acceptable(cls, *, returncode: int, combined: str) -> bool:
        if returncode == 0:
            return True
        lower = combined.lower()
        if "already in use" in lower:
            return True
        if "created node_id" in lower or "registernode" in lower:
            return True
        if "charmap" in lower and "encode" in lower:
            # Rich/typer console rendering failed on Windows pipes; verify via Control API instead.
            return False
        return False

    @classmethod
    def _exec_flwr_register(cls, pub: Path) -> tuple[int, str]:
        """Run `flwr supernode register` with UTF-8 log file (avoid Windows pipe charmap crashes)."""
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = cls._log_dir / f"register-{pub.stem}.log"
        cmd = [
            flwr_executable(),
            "supernode",
            "register",
            str(pub),
            "medshield-dev-tls",
        ]
        with log_path.open("wb") as log_handle:
            proc = subprocess.run(
                cmd,
                cwd=str(_APP_DIR),
                check=False,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=cls._base_env(),
            )
        combined = safe_diagnostic_text(log_path.read_text(encoding="utf-8", errors="replace"))
        return proc.returncode, combined

    @classmethod
    def _registered_node_count(cls) -> int:
        try:
            nodes = cls._list_nodes_via_control()
        except Exception:
            return 0
        ok = {"registered", "online"}
        return sum(1 for n in nodes if str(n.get("status", "")).lower() in ok)

    @classmethod
    def _aggregation_evidence(cls, state: dict[str, Any], *, num_rounds: int) -> bool:
        init_h = state.get("initial_parameter_hash")
        final_h = state.get("global_parameter_hash")
        if not init_h or not final_h or init_h == final_h:
            return False
        try:
            completed_rounds = int(state.get("current_round") or 0)
        except (TypeError, ValueError):
            completed_rounds = 0
        if completed_rounds >= num_rounds:
            return True
        history = state.get("round_history") or []
        return len(history) >= num_rounds

    @classmethod
    def _popen(
        cls,
        name: str,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict | None = None,
        log_basename: str | None = None,
    ) -> None:
        logger.info("flower_supervisor starting %s: %s", name, " ".join(cmd))
        merged = cls._base_env()
        if env:
            merged.update(env)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        log_handle = None
        if log_basename:
            log_path = cls._log_dir / f"{log_basename}.log"
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            stdout_dest = log_handle
            stderr_dest = subprocess.STDOUT
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or _REPO_ROOT),
            env=merged,
            stdout=stdout_dest,
            stderr=stderr_dest,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        if log_handle is not None:
            log_handle.close()
        cls._processes.append(_Proc(name=name, popen=proc))

    @classmethod
    def _flower_ports(cls) -> list[int]:
        from backend.app.core.config import settings as st

        ports = []
        for addr in (st.FLOWER_FLEET_API_ADDRESS, st.FLOWER_SUPERLINK_CONTROL_ADDRESS):
            if ":" in addr:
                ports.append(int(addr.rsplit(":", 1)[1]))
        ports.extend([9094, 9095])
        return sorted(set(ports))

    @classmethod
    def _prepare_clean_slate(cls) -> None:
        cls._cleanup_processes()
        ensure_ports_free(cls._flower_ports(), timeout=45.0)

    @classmethod
    def _read_log_tail(cls, basename: str, *, max_chars: int = 4000) -> str:
        path = cls._log_dir / f"{basename}.log"
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    @classmethod
    def _list_nodes_via_control(cls) -> list[dict[str, Any]]:
        import grpc
        from flwr.common.grpc import create_channel
        from flwr.proto.control_pb2 import ListNodesRequest  # pylint: disable=E0611
        from flwr.proto.control_pb2_grpc import ControlStub

        channel = create_channel(
            server_address=settings.FLOWER_SUPERLINK_CONTROL_ADDRESS,
            root_certificates=cls._cert_paths()["ca"].read_bytes(),
            insecure=False,
        )
        try:
            resp = ControlStub(channel).ListNodes(ListNodesRequest(), timeout=10)
            nodes = []
            for n in resp.nodes_info:
                nodes.append(
                    {
                        "node_id": int(n.node_id),
                        "status": n.status,
                        "owner": n.owner_name,
                    }
                )
            return nodes
        finally:
            channel.close()

    @classmethod
    def _wait_supernodes_active(cls, *, min_online: int = 2, timeout: float = 120.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        last: list[dict] = []
        while time.monotonic() < deadline:
            for proc in cls._processes:
                if proc.name.startswith("supernode") and proc.popen.poll() is not None:
                    tail = cls._read_log_tail(proc.name)
                    raise RuntimeError(
                        f"{proc.name} exited early (code={proc.popen.returncode}). Log tail:\n{tail}"
                    )
            try:
                last = cls._list_nodes_via_control()
            except Exception as exc:
                logger.debug("ListNodes not ready: %s", exc)
                time.sleep(1.0)
                continue
            online = [n for n in last if str(n.get("status", "")).lower() == "online"]
            registered = [n for n in last if str(n.get("status", "")).lower() == "registered"]
            if len(online) >= min_online:
                return last
            if len(online) + len(registered) >= min_online and len(online) >= 1:
                time.sleep(2.0)
                continue
            time.sleep(1.0)
        tails = "\n".join(
            f"--- {p.name} ---\n{cls._read_log_tail(p.name, max_chars=1500)}"
            for p in cls._processes
            if p.name.startswith("supernode")
        )
        raise RuntimeError(
            f"Timed out waiting for {min_online} active SuperNodes. Last nodes={last}. Logs:\n{tails}"
        )

    @classmethod
    def _wait_control_api(cls, timeout: float = 60.0) -> bool:
        import grpc
        from flwr.common.grpc import create_channel
        from flwr.proto.control_pb2 import ListFederationsRequest  # pylint: disable=E0611
        from flwr.proto.control_pb2_grpc import ControlStub

        addr = settings.FLOWER_SUPERLINK_CONTROL_ADDRESS
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                channel = create_channel(
                    server_address=addr,
                    root_certificates=cls._cert_paths()["ca"].read_bytes(),
                    insecure=False,
                )
                ControlStub(channel).ListFederations(ListFederationsRequest(), timeout=3)
                channel.close()
                return True
            except Exception:
                time.sleep(0.5)
        return False

    @classmethod
    def _ensure_flwr_connection_config(cls) -> None:
        from flwr.supercore.utils import get_flwr_home

        cfg_path = get_flwr_home() / "config.toml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        ca = cls._cert_paths()["ca"]
        body = f"""
[superlink.medshield-dev-tls]
address = "{settings.FLOWER_SUPERLINK_CONTROL_ADDRESS}"
root-certificates = "{ca.as_posix()}"
insecure = false
"""
        cfg_path.write_text(body.strip() + "\n", encoding="utf-8")

    @classmethod
    def _register_supernodes(cls) -> None:
        cls._ensure_flwr_connection_config()
        paths = cls._cert_paths()
        logs: list[str] = []
        for pub in (paths["client0_pub"], paths["client1_pub"]):
            before = cls._registered_node_count()
            returncode, combined = cls._exec_flwr_register(pub)
            after = cls._registered_node_count()
            logs.append(f"{pub.name} exit={returncode} nodes={before}->{after}\n{combined}")
            if cls._register_output_acceptable(returncode=returncode, combined=combined):
                logger.info("supernode register %s exit=%s (cli ok)", pub.name, returncode)
                continue
            if after > before:
                logger.warning(
                    "supernode register %s exit=%s but Control API node count increased",
                    pub.name,
                    returncode,
                )
                continue
            raise RuntimeError(
                safe_diagnostic_text(
                    f"supernode register failed for {pub.name} (exit={returncode}): {combined}"
                )
            )
        if cls._registered_node_count() < 2:
            raise RuntimeError(
                safe_diagnostic_text(
                    "SuperLink Control API reports fewer than 2 registered SuperNodes after register. "
                    + "\n---\n".join(logs)
                )
            )

    @classmethod
    def _start_superlink(cls, *, run_id: str) -> None:
        p = cls._cert_paths()
        db_path = _REPO_ROOT / "data" / "processed" / "federated" / f"superlink_{run_id[:8]}.db"
        cmd = [
            flower_superlink_executable(),
            "--ssl-certfile",
            str(p["server_cert"]),
            "--ssl-keyfile",
            str(p["server_key"]),
            "--ssl-ca-certfile",
            str(p["ca"]),
            "--enable-supernode-auth",
            "--control-api-address",
            settings.FLOWER_SUPERLINK_CONTROL_ADDRESS,
            "--fleet-api-address",
            settings.FLOWER_FLEET_API_ADDRESS,
            "--database",
            str(db_path),
        ]
        cls._popen("superlink", cmd, log_basename="superlink")
        time.sleep(1.0)
        if cls._processes[-1].popen.poll() is not None:
            raise RuntimeError(
                "flower-superlink exited on startup:\n" + cls._read_log_tail("superlink")
            )

    @classmethod
    def _start_supernodes(cls) -> None:
        p = cls._cert_paths()
        base = [
            flower_supernode_executable(),
            "--superlink",
            settings.FLOWER_FLEET_API_ADDRESS,
            "--root-certificates",
            str(p["ca"]),
        ]
        configs = [
            ("supernode-0", "9094", "partition-id=0 num-partitions=2", p["client0_key"]),
            ("supernode-1", "9095", "partition-id=1 num-partitions=2", p["client1_key"]),
        ]
        for name, port, node_config, key in configs:
            cmd = [
                *base,
                "--auth-supernode-private-key",
                str(key),
                "--node-config",
                node_config,
            ]
            cmd.extend(["--clientappio-api-address", f"127.0.0.1:{port}"])
            cls._popen(name, cmd, cwd=_APP_DIR, log_basename=name)

    @classmethod
    def _state_file_path(cls) -> Path:
        return _REPO_ROOT / "data" / "processed" / "federated" / "live_supervisor_state.json"

    @classmethod
    def _run_flwr_app(cls, *, num_rounds: int, run_id: str) -> subprocess.CompletedProcess:
        env = utf8_subprocess_env(os.environ.copy())
        env["PYTHONPATH"] = (
            str(_APP_DIR) + os.pathsep + str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        )
        env["FLOWER_NUM_ROUNDS"] = str(num_rounds)
        env["MEDSHIELD_FL_STATE_PATH"] = str(cls._state_file_path())
        env["MEDSHIELD_FL_RUN_ID"] = run_id
        cmd = [
            flwr_executable(),
            "run",
            ".",
            "medshield-dev-tls",
            "--stream",
            "--run-config",
            f"num-server-rounds={num_rounds}",
            "--run-config",
            "local-epochs=1",
        ]
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = cls._log_dir / "flwr-run.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            return subprocess.run(
                cmd,
                cwd=str(_APP_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )

    @classmethod
    def _cleanup_processes(cls) -> None:
        for proc in reversed(cls._processes):
            if proc.popen.poll() is None:
                try:
                    proc.popen.terminate()
                    proc.popen.wait(timeout=10)
                except Exception:
                    proc.popen.kill()
        cls._processes.clear()
        if cls._flwr_run_proc and cls._flwr_run_proc.poll() is None:
            cls._flwr_run_proc.terminate()

    @classmethod
    def _run_worker(cls, *, run_id: str, num_rounds: int) -> None:
        from fl.flower_deploy.run_state import read_state, write_state

        try:
            write_state(
                {
                    "lifecycle": "STARTING",
                    "run_id": run_id,
                    "total_rounds": num_rounds,
                    "round_history": [],
                    "last_error": None,
                    "supervisor_phase_failed": False,
                    "supervisor_error": None,
                }
            )
            os.environ["MEDSHIELD_FL_RUN_ID"] = run_id
            cls._prepare_clean_slate()
            cls._start_superlink(run_id=run_id)
            if not cls._wait_control_api():
                raise RuntimeError(
                    "SuperLink control API did not become ready:\n" + cls._read_log_tail("superlink")
                )
            cls._register_supernodes()
            cls._start_supernodes()
            nodes = cls._wait_supernodes_active(min_online=2, timeout=120.0)
            write_state(
                {
                    "lifecycle": "WAITING_FOR_CLIENTS",
                    "run_id": run_id,
                    "registered_nodes": nodes,
                }
            )
            cls._lifecycle = "WAITING_FOR_CLIENTS"

            cls._lifecycle = "RUNNING"
            proc = cls._run_flwr_app(num_rounds=num_rounds, run_id=run_id)
            code = proc.returncode
            if code != 0:
                raise RuntimeError(
                    f"flwr run exited with code {code}:\n{cls._read_log_tail('flwr-run', max_chars=2000)}"
                )
            state = read_state(path=cls._state_file_path())
            init_h = state.get("initial_parameter_hash")
            final_h = state.get("global_parameter_hash")
            if init_h and final_h and init_h == final_h:
                cls._lifecycle = "FAILED"
                cls._last_error = "Global parameters unchanged after run (no aggregation evidence)"
                write_state({"lifecycle": "FAILED", "last_error": cls._last_error, "run_id": run_id})
            elif not init_h or not final_h:
                cls._lifecycle = "FAILED"
                cls._last_error = "Missing parameter hash evidence in run state file"
                write_state({"lifecycle": "FAILED", "last_error": cls._last_error, "run_id": run_id})
            elif cls._aggregation_evidence(state, num_rounds=num_rounds):
                cls._lifecycle = "COMPLETED"
                write_state({"lifecycle": "COMPLETED", "run_id": run_id})
                try:
                    from ml.federated.gat_ecg_report_fl import finalize_federated_run

                    finalize_federated_run(run_id, live_state=state)
                except Exception as fin_exc:
                    msg = cls._safe_exc_message(fin_exc)
                    cls._last_error = f"Run aggregated but GAT evaluation failed: {msg}"
                    write_state(
                        {
                            "lifecycle": "COMPLETED",
                            "run_id": run_id,
                            "last_error": cls._last_error,
                            "gat_evaluation_failed": True,
                        }
                    )
            else:
                cls._lifecycle = "FAILED"
                cls._last_error = "Run finished without sufficient aggregation evidence"
                write_state(
                    {
                        "lifecycle": "FAILED",
                        "last_error": cls._last_error,
                        "run_id": run_id,
                    }
                )
        except Exception as exc:
            msg = cls._safe_exc_message(exc)
            logger.error("Flower supervisor run failed: %s", msg)
            state = read_state(path=cls._state_file_path())
            agg_ok = cls._aggregation_evidence(state, num_rounds=num_rounds)
            cls._last_error = msg
            if agg_ok:
                cls._lifecycle = "FAILED"
                cls._last_error = (
                    f"Supervisor error after successful aggregation ({msg}). "
                    "Round results preserved in run state."
                )
                write_state(
                    {
                        "last_error": cls._last_error,
                        "supervisor_error": msg,
                        "supervisor_phase_failed": True,
                        "run_id": run_id,
                    }
                )
            else:
                cls._lifecycle = "FAILED"
                write_state(
                    {
                        "lifecycle": "FAILED",
                        "last_error": cls._last_error,
                        "run_id": run_id,
                    }
                )
        finally:
            cls._cleanup_processes()

    @classmethod
    def start(cls, *, num_rounds: int) -> dict[str, Any]:
        ok, blockers = cls.security_readiness()
        if not ok:
            raise RuntimeError("SECURITY_NOT_CONFIGURED: " + "; ".join(blockers))
        with cls._lock:
            if cls._lifecycle in ("RUNNING", "STARTING", "WAITING_FOR_CLIENTS", "STOPPING"):
                raise RuntimeError("A federated run is already in progress")
            cls._prepare_clean_slate()
            run_id = str(uuid.uuid4())
            cls._run_id = run_id
            cls._last_error = None
            cls._lifecycle = "STARTING"
            cls._run_thread = threading.Thread(
                target=cls._run_worker,
                kwargs={"run_id": run_id, "num_rounds": num_rounds},
                name=f"flower-supervisor-{run_id[:8]}",
                daemon=True,
            )
            cls._run_thread.start()
        return {"run_id": run_id, "status": "STARTED", "num_rounds": num_rounds}

    @classmethod
    def stop(cls) -> dict[str, Any]:
        with cls._lock:
            if cls._lifecycle not in ("RUNNING", "WAITING_FOR_CLIENTS", "STARTING"):
                return {"status": "NO_ACTIVE_RUN", "lifecycle": cls._lifecycle}
            cls._lifecycle = "STOPPING"
        cls._cleanup_processes()
        cls._lifecycle = "SERVER_STOPPED"
        return {"status": "STOPPED"}

    @classmethod
    def get_snapshot(cls) -> dict[str, Any]:
        from fl.flower_deploy.run_state import read_state

        file_state = read_state()
        procs = [
            {"name": p.name, "pid": p.popen.pid, "alive": p.popen.poll() is None}
            for p in cls._processes
        ]
        sec_ok, sec_blockers = cls.security_readiness()
        return {
            "lifecycle": cls._lifecycle,
            "run_id": cls._run_id or file_state.get("run_id"),
            "processes": procs,
            "security_configured": sec_ok,
            "security_blockers": sec_blockers,
            "file_state": file_state,
            "last_error": cls._last_error or file_state.get("last_error"),
        }

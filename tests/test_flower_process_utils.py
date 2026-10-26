"""Flower subprocess UTF-8 helpers (Windows-safe)."""

from __future__ import annotations

from backend.app.services.flower_process_utils import (
    decode_subprocess_bytes,
    safe_diagnostic_text,
    utf8_subprocess_env,
)
from backend.app.services.flower_supervisor import FlowerSupervisor


def test_utf8_subprocess_env_sets_flags():
    env = utf8_subprocess_env({})
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_decode_subprocess_bytes_replaces_invalid_utf8():
    raw = b"ok \xff\xfe checkmark \xe2\x9c\x85"
    text = decode_subprocess_bytes(raw)
    assert "ok" in text
    assert "\ufffd" in text or "checkmark" in text


def test_safe_diagnostic_text_survives_checkmark():
    text = safe_diagnostic_text("Registered \u2705 node")
    assert "\u2705" in text
    # Must be encodable as utf-8 for logs/files
    text.encode("utf-8")


def test_register_output_acceptable_already_in_use():
    assert FlowerSupervisor._register_output_acceptable(
        returncode=1,
        combined="Public key already in use",
    )


def test_register_output_acceptable_success_exit():
    assert FlowerSupervisor._register_output_acceptable(returncode=0, combined="\u2705 done")


def test_normalize_round_history_dedupes():
    from fl.flower_deploy.run_state import normalize_round_history, upsert_round_history

    hist = upsert_round_history([], {"round": 1, "successful_clients": 2}, run_id="a")
    hist = upsert_round_history(hist, {"round": 1, "successful_clients": 2}, run_id="a")
    hist = upsert_round_history(hist, {"round": 2, "successful_clients": 2}, run_id="a")
    assert len(hist) == 2
    assert normalize_round_history(hist + hist) == hist

"""Multimodal training gate and checkpoint verify script (no training, no PHI)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml.training.linkage_gate import evaluate_three_modality_gate

REPO = Path(__file__).resolve().parents[1]


def test_three_modality_gate_blocked_by_audit():
    gate = evaluate_three_modality_gate()
    assert gate.allowed is False
    assert gate.verdict == "BLOCKED"
    assert gate.blockers


def test_train_multimodal_cli_check_linkage_exits_blocked():
    proc = subprocess.run(
        [sys.executable, "-m", "ml.training.train_multimodal_fusion", "--check-linkage-only"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert body["allowed"] is False


def test_verify_fusion_checkpoint_missing_manifest():
    proc = subprocess.run(
        [sys.executable, "scripts/verify_fusion_checkpoint.py"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "MODEL_DIR": str(REPO / "models_empty_test_dir")},
    )
    assert proc.returncode == 1

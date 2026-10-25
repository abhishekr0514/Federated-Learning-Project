"""Human-readable inference setup diagnostics (no secrets, no paths with credentials)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ml.inference.manifest import MANIFEST_FILENAME


def _optional_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_next_steps(
    *,
    inference_enabled: bool,
    readiness_blockers: list[str],
    model_dir: Path,
    manifest_present: bool,
    checkpoint_present: bool,
) -> list[str]:
    steps: list[str] = []
    if not _optional_import("torch_geometric"):
        steps.append(
            "Install PyTorch Geometric: py -m pip install \"torch-geometric>=2.4.0\" "
            "(or py -m pip install -e \".[ml]\" from the repo root)."
        )
    if not inference_enabled:
        steps.append(
            "Keep INFERENCE_ENABLED=false until a verified trained checkpoint is deployed. "
            "Then set INFERENCE_ENABLED=true in .env and restart the API."
        )
    if not manifest_present:
        steps.append(
            f"Copy models/medshield_fusion_manifest.example.json to "
            f"{{MODEL_DIR}}/{MANIFEST_FILENAME} and point checkpoint_relative_path "
            "at an exported fusion checkpoint."
        )
    if manifest_present and not checkpoint_present:
        steps.append(
            "Place the trained fusion checkpoint at the path declared in the manifest "
            "(payload from save_fusion_checkpoint with trained=true)."
        )
    if any("trained" in b.lower() for b in readiness_blockers):
        steps.append(
            "Train or export a genuine multimodal GAT fusion checkpoint (proxy task only). "
            "No production checkpoint exists in this repository; test-only synthetic "
            "checkpoints must not be used for clinical inference."
        )
    if not steps:
        steps.append(
            "All runtime gates passed; inference may be enabled only after operational review."
        )
    return steps


def environment_summary() -> dict[str, Any]:
    torch_version = None
    if _optional_import("torch"):
        import torch

        torch_version = torch.__version__
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "torch_geometric_installed": _optional_import("torch_geometric"),
        "transformers_installed": _optional_import("transformers"),
    }

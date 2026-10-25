"""Verified fusion checkpoint manifest (no secrets in logs)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "medshield_fusion_manifest.json"
MANIFEST_VERSION = "1"
DEFAULT_ARCHITECTURE_ID = "medshield.diagnostic_net.gat_fusion"
DEFAULT_ARCHITECTURE_VERSION = "1.0.0"


@dataclass(frozen=True)
class FusionManifest:
    manifest_version: str
    checkpoint_relative_path: str
    architecture_id: str
    architecture_version: str
    model_version_tag: str
    target_definition: str
    prediction_type: str
    require_trained_checkpoint: bool

    def resolve_checkpoint(self, model_dir: Path) -> Path:
        ckpt = (model_dir / self.checkpoint_relative_path).resolve()
        model_root = model_dir.resolve()
        if model_root not in ckpt.parents and ckpt != model_root:
            raise ValueError("Checkpoint path escapes MODEL_DIR")
        return ckpt


def load_fusion_manifest(model_dir: Path) -> FusionManifest:
    path = model_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing inference manifest: {MANIFEST_FILENAME}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest_version: {data.get('manifest_version')!r}")
    return FusionManifest(
        manifest_version=str(data["manifest_version"]),
        checkpoint_relative_path=str(data["checkpoint_relative_path"]),
        architecture_id=str(data.get("architecture_id", DEFAULT_ARCHITECTURE_ID)),
        architecture_version=str(data.get("architecture_version", DEFAULT_ARCHITECTURE_VERSION)),
        model_version_tag=str(data["model_version_tag"]),
        target_definition=str(data["target_definition"]),
        prediction_type=str(data.get("prediction_type", "proxy_evaluation")),
        require_trained_checkpoint=bool(data.get("require_trained_checkpoint", True)),
    )

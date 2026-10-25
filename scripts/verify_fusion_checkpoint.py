"""Verify exported GAT fusion checkpoint and manifest (no secrets, no inference enable)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402
from ml.inference.manifest import load_fusion_manifest, MANIFEST_FILENAME  # noqa: E402
from ml.models.gnn_fusion import (  # noqa: E402
    ARCHITECTURE_ID,
    ECG_EMBED_DIM,
    TABULAR_EMBED_DIM,
    TEXT_EMBED_DIM,
    checkpoint_summary,
    load_fusion_checkpoint,
)


def _model_dir() -> Path:
    root = Path(settings.MODEL_DIR)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def main() -> int:
    model_dir = _model_dir()
    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        print(f"Missing manifest: {manifest_path}")
        return 1

    manifest = load_fusion_manifest(model_dir)
    ckpt_path = manifest.resolve_checkpoint(model_dir)
    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path.name} (under MODEL_DIR)")
        return 1

    summary = checkpoint_summary(ckpt_path)
    if summary.get("architecture_id") != ARCHITECTURE_ID:
        print("architecture_id mismatch")
        return 1

    if manifest.require_trained_checkpoint and not summary.get("trained"):
        print("Checkpoint not marked trained; export rejected for production manifest")
        return 1

    try:
        model = load_fusion_checkpoint(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"load_fusion_checkpoint failed: {type(exc).__name__}")
        return 1

    import torch
    from ml.models.gnn_fusion import FusionModelInput, NUM_MODALITIES

    model.eval()
    with torch.inference_mode():
        batch = 1
        inp = FusionModelInput(
            ecg=torch.randn(batch, ECG_EMBED_DIM),
            clinical_text=torch.randn(batch, TEXT_EMBED_DIM),
            tabular=torch.randn(batch, TABULAR_EMBED_DIM),
            available=torch.ones(batch, NUM_MODALITIES, dtype=torch.bool),
            quality_ok=torch.ones(batch, NUM_MODALITIES, dtype=torch.bool),
        )
        out = model(inp)
        if out.confidence is None or out.predicted_class is None:
            print("Forward pass did not produce confidence outputs")
            return 1

    report = {
        "manifest_ok": True,
        "checkpoint_ok": True,
        "architecture_id": summary.get("architecture_id"),
        "trained": summary.get("trained"),
        "target_definition": manifest.target_definition[:120],
        "prediction_type": manifest.prediction_type,
        "model_version_tag": manifest.model_version_tag,
        "embedding_contract": {
            "ecg": ECG_EMBED_DIM,
            "clinical_text": TEXT_EMBED_DIM,
            "tabular": TABULAR_EMBED_DIM,
        },
        "smoke_forward_shape_ok": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

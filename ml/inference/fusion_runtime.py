"""Lazy-loaded GAT fusion inference with explicit readiness gates."""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.inference.constants import ECG_EMBED_DIM, TABULAR_EMBED_DIM, TEXT_EMBED_DIM
from ml.inference.embedding_contract import validate_modality_embeddings
from ml.inference.manifest import FusionManifest, load_fusion_manifest

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: FusionInferenceRuntime | None = None


@dataclass
class InferenceReadiness:
    ready: bool
    blockers: list[str] = field(default_factory=list)
    dependencies: dict[str, bool] = field(default_factory=dict)
    manifest_present: bool = False
    checkpoint_present: bool = False
    checkpoint_trained: bool | None = None
    model_loaded: bool = False
    inference_api_wired: bool = True

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "dependencies": dict(self.dependencies),
            "manifest_present": self.manifest_present,
            "checkpoint_present": self.checkpoint_present,
            "checkpoint_trained": self.checkpoint_trained,
            "model_loaded": self.model_loaded,
            "inference_api_wired": self.inference_api_wired,
        }


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch_geometric_available() -> bool:
    return importlib.util.find_spec("torch_geometric") is not None


class FusionInferenceRuntime:
    """Thread-safe lazy loader for verified fusion checkpoints."""

    def __init__(self, model_dir: Path, *, inference_enabled: bool) -> None:
        self._model_dir = model_dir
        self._inference_enabled = inference_enabled
        self._manifest: FusionManifest | None = None
        self._model: Any = None
        self._load_lock = threading.Lock()

    def assess_readiness(self) -> InferenceReadiness:
        blockers: list[str] = []
        deps = {
            "torch": _torch_available(),
            "torch_geometric": _torch_geometric_available(),
        }
        if not self._inference_enabled:
            blockers.append("INFERENCE_ENABLED is false")
        if not deps["torch"]:
            blockers.append("PyTorch not installed")
        if not deps["torch_geometric"]:
            blockers.append("torch-geometric not installed")
        if not self._model_dir.is_dir():
            blockers.append("MODEL_DIR does not exist")

        manifest_present = False
        checkpoint_present = False
        checkpoint_trained: bool | None = None

        if not blockers and self._model_dir.is_dir():
            from ml.models.gnn_fusion import ARCHITECTURE_ID, checkpoint_summary
            try:
                manifest = load_fusion_manifest(self._model_dir)
                manifest_present = True
                ckpt = manifest.resolve_checkpoint(self._model_dir)
                checkpoint_present = ckpt.is_file()
                if not checkpoint_present:
                    blockers.append("Manifest checkpoint file missing")
                else:
                    try:
                        summary = checkpoint_summary(ckpt)
                    except Exception as exc:
                        blockers.append(f"Checkpoint unreadable: {type(exc).__name__}")
                        summary = {}
                    checkpoint_trained = bool(summary.get("trained", False))
                    if summary and summary.get("architecture_id") != ARCHITECTURE_ID:
                        blockers.append("Checkpoint architecture_id mismatch")
                    if manifest.require_trained_checkpoint and not checkpoint_trained:
                        blockers.append("Checkpoint not marked trained (research-only weights rejected)")
            except (FileNotFoundError, ValueError, OSError) as exc:
                blockers.append(f"Manifest invalid: {exc}")

        model_loaded = self._model is not None
        ready = len(blockers) == 0
        return InferenceReadiness(
            ready=ready,
            blockers=blockers,
            dependencies=deps,
            manifest_present=manifest_present,
            checkpoint_present=checkpoint_present,
            checkpoint_trained=checkpoint_trained,
            model_loaded=model_loaded,
        )

    def _ensure_loaded(self) -> tuple[FusionManifest, Any]:
        from ml.models.gnn_fusion import MedShieldDiagnosticNet, load_fusion_checkpoint

        readiness = self.assess_readiness()
        if not readiness.ready:
            raise RuntimeError("; ".join(readiness.blockers))

        with self._load_lock:
            if self._model is not None and self._manifest is not None:
                return self._manifest, self._model

            manifest = load_fusion_manifest(self._model_dir)
            ckpt = manifest.resolve_checkpoint(self._model_dir)
            model = load_fusion_checkpoint(ckpt, map_location="cpu")
            model.eval()
            self._manifest = manifest
            self._model = model
            return manifest, model

    def run_proxy_inference(
        self,
        *,
        ecg: list[float],
        clinical_text: list[float],
        tabular: list[float],
    ) -> dict[str, Any]:
        validate_modality_embeddings(ecg, clinical_text, tabular)
        manifest, model = self._ensure_loaded()

        import torch
        from ml.models.gnn_fusion import FusionModelInput, NUM_MODALITIES

        with torch.inference_mode():
            ecg_t = torch.tensor([ecg], dtype=torch.float32)
            text_t = torch.tensor([clinical_text], dtype=torch.float32)
            tab_t = torch.tensor([tabular], dtype=torch.float32)
            masks = torch.ones(1, NUM_MODALITIES, dtype=torch.bool)
            fusion_input = FusionModelInput(
                ecg=ecg_t,
                clinical_text=text_t,
                tabular=tab_t,
                available=masks,
                quality_ok=masks,
            )
            output = model(fusion_input)
            if output.confidence is None or output.predicted_class is None:
                raise RuntimeError("Fusion forward did not produce confidence outputs")

            conf = float(output.confidence[0].item())
            pred_class = int(output.predicted_class[0].item())

        return {
            "target_definition": manifest.target_definition,
            "prediction_type": manifest.prediction_type,
            "model_version_tag": manifest.model_version_tag,
            "confidence_uncalibrated": conf,
            "predicted_class_index": pred_class,
            "predicted_class_label": "CVD" if pred_class == 1 else "NORM",
            "research_only": True,
            "proxy_task": True,
            "disclaimer": (
                "Uncalibrated research proxy output. Not validated clinical heart-disease diagnosis."
            ),
        }


def get_fusion_inference_runtime(
    model_dir: Path,
    *,
    inference_enabled: bool,
    reset: bool = False,
) -> FusionInferenceRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        recreate = (
            reset
            or _RUNTIME is None
            or _RUNTIME._model_dir != model_dir.resolve()
            or _RUNTIME._inference_enabled != inference_enabled
        )
        if recreate:
            _RUNTIME = FusionInferenceRuntime(model_dir, inference_enabled=inference_enabled)
        return _RUNTIME

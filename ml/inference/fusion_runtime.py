"""Lazy-loaded two-modality GAT fusion inference."""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.inference.manifest import FusionManifest, load_fusion_manifest

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: FusionInferenceRuntime | None = None

GAT_ECG_REPORT_ARCH = "medshield.diagnostic_net.gat_ecg_report"
LEGACY_THREE_MOD_ARCH = "medshield.diagnostic_net.gat_fusion"


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
    architecture_id: str | None = None

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
            "architecture_id": self.architecture_id,
        }


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch_geometric_available() -> bool:
    return importlib.util.find_spec("torch_geometric") is not None


class FusionInferenceRuntime:
    def __init__(self, model_dir: Path, *, inference_enabled: bool) -> None:
        self._model_dir = model_dir
        self._inference_enabled = inference_enabled
        self._manifest: FusionManifest | None = None
        self._model: Any = None
        self._load_lock = threading.Lock()

    def assess_readiness(self) -> InferenceReadiness:
        blockers: list[str] = []
        deps = {"torch": _torch_available(), "torch_geometric": _torch_geometric_available()}
        arch_id: str | None = None

        inference_disabled = not self._inference_enabled
        if not deps["torch"]:
            blockers.append("PyTorch not installed")
        if not deps["torch_geometric"]:
            blockers.append("torch-geometric not installed")
        if not self._model_dir.is_dir():
            blockers.append("FUSION_MODEL_DIR does not exist")

        manifest_present = checkpoint_present = False
        checkpoint_trained: bool | None = None

        artifact_blockers: list[str] = []
        if self._model_dir.is_dir() and deps["torch"]:
            try:
                manifest = load_fusion_manifest(self._model_dir)
                manifest_present = True
                arch_id = manifest.architecture_id
                if manifest.architecture_id == LEGACY_THREE_MOD_ARCH:
                    artifact_blockers.append(
                        "Legacy three-modality GAT manifest rejected; use gat_ecg_report."
                    )
                elif manifest.architecture_id != GAT_ECG_REPORT_ARCH:
                    artifact_blockers.append("Unsupported fusion architecture_id in manifest")

                ckpt = manifest.resolve_checkpoint(self._model_dir)
                checkpoint_present = ckpt.is_file()
                if not checkpoint_present:
                    artifact_blockers.append("Manifest checkpoint file missing")
                elif deps["torch_geometric"]:
                    from ml.models.gat_ecg_report_fusion import checkpoint_summary

                    summary = checkpoint_summary(ckpt)
                    checkpoint_trained = bool(summary.get("trained", False))
                    if manifest.require_trained_checkpoint and not checkpoint_trained:
                        artifact_blockers.append("Checkpoint not marked trained")
            except Exception as exc:
                artifact_blockers.append(f"Manifest invalid: {type(exc).__name__}")

        blockers.extend(artifact_blockers)
        if inference_disabled:
            blockers.append("FUSION_INFERENCE_ENABLED is false")

        ready = len(blockers) == 0
        return InferenceReadiness(
            ready=ready,
            blockers=blockers,
            dependencies=deps,
            manifest_present=manifest_present,
            checkpoint_present=checkpoint_present,
            checkpoint_trained=checkpoint_trained,
            model_loaded=self._model is not None,
            architecture_id=arch_id,
        )

    def _ensure_loaded(self) -> tuple[FusionManifest, Any]:
        from ml.models.gat_ecg_report_fusion import MedShieldGATECGReportNet, load_gat_ecg_report_checkpoint

        readiness = self.assess_readiness()
        if not readiness.ready:
            raise RuntimeError("; ".join(readiness.blockers))

        with self._load_lock:
            if self._model is not None and self._manifest is not None:
                return self._manifest, self._model
            manifest = load_fusion_manifest(self._model_dir)
            ckpt = manifest.resolve_checkpoint(self._model_dir)
            model = load_gat_ecg_report_checkpoint(ckpt, map_location="cpu")
            model.eval()
            self._manifest = manifest
            self._model = model
            return manifest, model

    def run_proxy_inference(
        self,
        *,
        ecg: list[float],
        clinical_text: list[float],
    ) -> dict[str, Any]:
        if len(ecg) != 128 or len(clinical_text) != 128:
            raise ValueError("ECG and clinical_text embeddings must be 128-dimensional.")
        manifest, model = self._ensure_loaded()

        import torch
        from ml.models.gat_ecg_report_fusion import GATECGReportInput

        with torch.inference_mode():
            ecg_t = torch.tensor([ecg], dtype=torch.float32)
            text_t = torch.tensor([clinical_text], dtype=torch.float32)
            logits = model(GATECGReportInput(ecg_t, text_t))
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            pred_class = int(pred[0].item())
            conf_f = float(conf[0].item())

        label = "MI/STTC" if pred_class == 1 else "NORM"
        return {
            "target_definition": manifest.target_definition,
            "prediction_type": manifest.prediction_type,
            "model_version_tag": manifest.model_version_tag,
            "confidence_uncalibrated": conf_f,
            "predicted_class_index": pred_class,
            "predicted_class_label": label,
            "architecture_id": manifest.architecture_id,
            "research_only": True,
            "proxy_task": True,
            "disclaimer": (
                "Uncalibrated research proxy output (two-modality GAT). "
                "Not validated clinical heart-disease diagnosis."
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

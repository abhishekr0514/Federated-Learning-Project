"""Thread-safe, lazy-loaded runtime for the two-modality ECG + clinical report model.

Architecture: medshield.ecg_report_fusion.v1
Research proxy task: PTB-XL NORM (0) vs MI/STTC (1).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig
from ml.models.ecg_report_fusion import (
    ARCHITECTURE_ID,
    ECGReportFusionClassifier,
    load_ecg_report_checkpoint,
)
from ml.preprocessing.clinical_text_privacy import mask_clinical_text_for_encoding
from ml.preprocessing.ecg_normalization import ECGNormalizationStats
from ml.preprocessing.ecg_signal_quality import assess_ecg_signal, LEAD_ORDER
from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline
from ml.text.errors import MaskingFailedError, TextInputValidationError

logger = logging.getLogger(__name__)

TEXT_MAX_LEN = 256
TARGET_DEFINITION = (
    "PTB-XL proxy task: NORM (class 0) vs MI or STTC superclasses (class 1). "
    "NOT validated clinical heart-disease diagnosis."
)
DISCLAIMER = (
    "EXPERIMENTAL RESEARCH PROXY OUTPUT ONLY. This experimental model is not "
    "approved, certified, or validated for clinical diagnostic, prognostic, "
    "treatment, or patient triage decision-making."
)
DEFAULT_VERSION_TAG = "ecg-report-fusion-v1-ptbxl"

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: ECGReportInferenceRuntime | None = None


@dataclass
class ECGReportReadiness:
    ready: bool
    blockers: list[str] = field(default_factory=list)
    manifest_present: bool = False
    checkpoint_present: bool = False
    checkpoint_trained: bool | None = None
    normalization_present: bool = False
    text_model_loaded: bool = False
    model_loaded: bool = False
    architecture_id: str = ARCHITECTURE_ID

    runtime_initialized: bool = False

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "manifest_present": self.manifest_present,
            "checkpoint_present": self.checkpoint_present,
            "checkpoint_trained": self.checkpoint_trained,
            "normalization_present": self.normalization_present,
            "text_model_loaded": self.text_model_loaded,
            "model_loaded": self.model_loaded,
            "runtime_initialized": self.runtime_initialized,
            "architecture_id": self.architecture_id,
        }


class ECGReportInferenceRuntime:
    """
    Lazy-loaded, thread-safe runtime for ECG + clinical report evaluation.
    Keeps model on CPU in evaluation mode and reuses weights across requests.
    """

    def __init__(
        self,
        model_dir: Path | str,
        *,
        enabled: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self._model_dir = Path(model_dir)
        self._enabled = enabled
        self._device = torch.device(device)
        self._load_lock = threading.Lock()
        self._model: ECGReportFusionClassifier | None = None
        self._text_pipeline: ClinicalTextEncodingPipeline | None = None
        self._norm_stats: ECGNormalizationStats | None = None
        self._manifest: dict[str, Any] | None = None
        self._load_error: str | None = None

    def _resolve_repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute() and p.exists():
            return p
        repo_root = self._resolve_repo_root()
        candidate = repo_root / p
        if candidate.exists():
            return candidate
        candidate_dir = self._model_dir / p.name
        if candidate_dir.exists():
            return candidate_dir
        return candidate

    def _load_manifest_if_exists(self) -> dict[str, Any] | None:
        if self._manifest is not None:
            return self._manifest
        manifest_path = self._model_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse manifest.json: %s", type(exc).__name__)
        return self._manifest

    def assess_readiness(self, *, verify_forward_pass: bool = True) -> ECGReportReadiness:
        blockers: list[str] = []
        if not self._enabled:
            blockers.append("ECG_REPORT_INFERENCE_ENABLED is false")

        manifest = self._load_manifest_if_exists()
        manifest_present = manifest is not None
        if not manifest_present:
            blockers.append("Model manifest missing in MODEL_DIR")

        ckpt_path: Path | None = None
        norm_path: Path | None = None
        checkpoint_present = False
        checkpoint_trained: bool | None = None
        normalization_present = False

        if manifest:
            arch_id = manifest.get("architecture_id")
            if arch_id != ARCHITECTURE_ID:
                blockers.append(f"Manifest architecture_id mismatch: {arch_id}")
            checkpoint_trained = bool(manifest.get("trained", False))
            if not checkpoint_trained:
                blockers.append("Checkpoint in manifest is not marked trained")

            raw_ckpt = manifest.get("checkpoint_path", "best_checkpoint.pt")
            ckpt_path = self._resolve_path(raw_ckpt)
            checkpoint_present = ckpt_path.is_file()
            if not checkpoint_present:
                blockers.append("Model checkpoint file missing")

            raw_norm = manifest.get("normalization_path", "normalization.json")
            norm_path = self._resolve_path(raw_norm)
            normalization_present = norm_path.is_file()
            if not normalization_present:
                blockers.append("ECG normalization statistics file missing")
        else:
            default_ckpt = self._model_dir / "best_checkpoint.pt"
            default_norm = self._model_dir / "normalization.json"
            checkpoint_present = default_ckpt.is_file()
            normalization_present = default_norm.is_file()
            if not checkpoint_present:
                blockers.append("Checkpoint best_checkpoint.pt missing")
            if not normalization_present:
                blockers.append("Normalization stats normalization.json missing")

        if self._load_error:
            blockers.append(f"Model loading error: {self._load_error}")

        # Only report ready if all artifacts pass AND a real forward pass succeeds
        if not blockers and verify_forward_pass:
            if not getattr(self, "_forward_verified", False):
                try:
                    self.ensure_loaded()
                    # Execute a verification forward pass
                    dummy_signal = np.zeros((12, 1000), dtype=np.float32)
                    dummy_text = "ECG Report Model Verification"
                    self.run_inference(dummy_signal, dummy_text)
                    self._forward_verified = True
                except Exception as exc:
                    blockers.append(f"Forward pass verification failed: {type(exc).__name__}")
                    self._forward_verified = False

        runtime_initialized = self._model is not None or getattr(self, "_forward_verified", False)
        if verify_forward_pass:
            ready = len(blockers) == 0 and getattr(self, "_forward_verified", False)
        else:
            # Registry path: configuration + artifacts only (weights load on first inference).
            ready = len(blockers) == 0

        return ECGReportReadiness(
            ready=ready,
            blockers=blockers,
            manifest_present=manifest_present,
            checkpoint_present=checkpoint_present,
            checkpoint_trained=checkpoint_trained,
            normalization_present=normalization_present,
            text_model_loaded=self._text_pipeline is not None,
            model_loaded=self._model is not None,
            runtime_initialized=runtime_initialized,
            architecture_id=ARCHITECTURE_ID,
        )

    def ensure_loaded(self) -> None:
        """
        Loads the BioClinicalBERT encoder, normalization stats, and fusion checkpoint.
        Thread-safe; executes once unless an error occurs.
        """
        if self._model is not None and self._norm_stats is not None:
            return

        with self._load_lock:
            if self._model is not None and self._norm_stats is not None:
                return

            if not self._enabled:
                raise RuntimeError("ECG + report inference is disabled in configuration.")

            manifest = self._load_manifest_if_exists()
            if not manifest:
                raise FileNotFoundError("Model manifest.json not found in model directory.")

            raw_ckpt = manifest.get("checkpoint_path", "best_checkpoint.pt")
            ckpt_path = self._resolve_path(raw_ckpt)
            if not ckpt_path.is_file():
                raise FileNotFoundError("Model checkpoint file not found.")

            raw_norm = manifest.get("normalization_path", "normalization.json")
            norm_path = self._resolve_path(raw_norm)
            if not norm_path.is_file():
                raise FileNotFoundError("ECG normalization statistics file not found.")

            try:
                # 1. Load normalization stats
                self._norm_stats = ECGNormalizationStats.load(norm_path)

                # 2. Load BioClinicalBERT pipeline
                text_cfg = ClinicalTextEncoderConfig(
                    model_id="emilyalsentzer/Bio_ClinicalBERT",
                    max_sequence_length=TEXT_MAX_LEN,
                    embedding_dim=128,
                )
                pipeline, _meta = ClinicalTextEncodingPipeline.from_pretrained(
                    config=text_cfg,
                    device=self._device,
                )
                self._text_pipeline = pipeline

                # 3. Load checkpoint
                model = load_ecg_report_checkpoint(
                    ckpt_path,
                    pipeline.encoder,
                    map_location=self._device,
                )
                model.to(self._device)
                model.eval()
                self._model = model
                self._load_error = None
                logger.info("ECG + report fusion model successfully loaded into memory.")
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {str(exc)[:120]}"
                logger.error("Failed to load ECG + report model: %s", type(exc).__name__)
                raise RuntimeError("Failed to load ECG + report model artifacts.") from exc

    def _prepare_modalities(
        self,
        signal_mv: np.ndarray | Sequence[Sequence[float]],
        report_text: str,
        *,
        lead_order: Sequence[str] = LEAD_ORDER,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate ECG + privacy-masked report and build model tensors."""
        self.ensure_loaded()
        assert self._model is not None
        assert self._norm_stats is not None
        assert self._text_pipeline is not None

        if not isinstance(signal_mv, np.ndarray):
            signal_mv = np.asarray(signal_mv, dtype=np.float32)

        if signal_mv.shape != (12, 1000):
            raise ValueError(f"ECG array must have shape (12, 1000), got {signal_mv.shape}")

        if not np.all(np.isfinite(signal_mv)):
            raise ValueError("ECG signal contains non-finite (NaN or Inf) values.")

        quality = assess_ecg_signal(signal_mv, lead_order=lead_order)
        if not quality.passed:
            reasons = "; ".join(quality.descriptions)
            raise ValueError(f"ECG signal failed quality check: {reasons}")

        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("Clinical report text must be a non-empty string.")

        try:
            masked_text = mask_clinical_text_for_encoding(report_text)
        except MaskingFailedError as exc:
            raise ValueError("Clinical report failed privacy masking validation.") from exc
        except Exception as exc:
            raise ValueError("Clinical report privacy masking error.") from exc

        norm_signal = self._norm_stats.transform(signal_mv)
        ecg_tensor = torch.from_numpy(norm_signal).unsqueeze(0).to(self._device)

        tokenizer = self._text_pipeline.tokenizer
        tokenized = tokenizer(
            [masked_text],
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_LEN,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)
        return ecg_tensor, input_ids, attention_mask

    def extract_encoder_embeddings(
        self,
        signal_mv: np.ndarray | Sequence[Sequence[float]],
        report_text: str,
        *,
        lead_order: Sequence[str] = LEAD_ORDER,
    ) -> dict[str, Any]:
        """
        Return 128-d ECG and clinical-text embeddings from the authorized encoder stack.
        Privacy masking is applied server-side; raw report text is not returned.
        """
        ecg_tensor, input_ids, attention_mask = self._prepare_modalities(
            signal_mv, report_text, lead_order=lead_order
        )
        assert self._model is not None

        with torch.inference_mode():
            _logit, ecg_emb, text_emb = self._model(
                ecg=ecg_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ecg_vec = ecg_emb.squeeze(0).detach().cpu().numpy().astype(np.float64)
            text_vec = text_emb.squeeze(0).detach().cpu().numpy().astype(np.float64)

        if ecg_vec.shape != (128,) or text_vec.shape != (128,):
            raise RuntimeError("Encoder produced unexpected embedding shape.")
        if not np.all(np.isfinite(ecg_vec)) or not np.all(np.isfinite(text_vec)):
            raise RuntimeError("Encoder produced non-finite embedding values.")

        return {
            "ecg_embedding": [float(x) for x in ecg_vec.tolist()],
            "text_embedding": [float(x) for x in text_vec.tolist()],
            "architecture_id": ARCHITECTURE_ID,
            "model_version_tag": DEFAULT_VERSION_TAG,
            "privacy_validation_status": "VALIDATED",
        }

    def run_inference(
        self,
        signal_mv: np.ndarray | Sequence[Sequence[float]],
        report_text: str,
        *,
        lead_order: Sequence[str] = LEAD_ORDER,
    ) -> dict[str, Any]:
        """
        Execute forward pass on a 12-lead (12, 1000) ECG signal and clinical report string.
        """
        ecg_tensor, input_ids, attention_mask = self._prepare_modalities(
            signal_mv, report_text, lead_order=lead_order
        )
        assert self._model is not None

        with torch.inference_mode():
            logit, _ecg_emb, _text_emb = self._model(
                ecg=ecg_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            prob = torch.sigmoid(logit).squeeze().item()

        if not np.isfinite(prob):
            raise RuntimeError("Model evaluated non-finite output probability.")

        predicted_class_index = int(1 if prob >= 0.5 else 0)
        predicted_class_label = "MI/STTC" if prob >= 0.5 else "NORM"

        return {
            "probability": float(prob),
            "predicted_class_index": predicted_class_index,
            "predicted_class_label": predicted_class_label,
            "target_definition": TARGET_DEFINITION,
            "architecture_id": ARCHITECTURE_ID,
            "model_version_tag": DEFAULT_VERSION_TAG,
            "is_research_only": True,
            "disclaimer": DISCLAIMER,
            "provenance": f"{ARCHITECTURE_ID}@trained_checkpoint",
        }


def get_ecg_report_runtime(
    model_dir: Path | str,
    *,
    enabled: bool = True,
    reset: bool = False,
) -> ECGReportInferenceRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None or reset:
            _RUNTIME = ECGReportInferenceRuntime(model_dir, enabled=enabled)
        return _RUNTIME

"""Evaluate MedShieldGATECGReportNet checkpoints on the federated FL bundle test split."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.classification_metrics import compute_full_binary_metrics
from ml.federated.gat_ecg_report_fl import default_bundle_path, load_bundle
from ml.models.gat_ecg_report_fusion import (
    ARCHITECTURE_ID,
    GATECGReportInput,
    load_gat_ecg_report_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENTRALIZED_CKPT = REPO_ROOT / "models" / "gat_ecg_report" / "best_gat_checkpoint.pt"
GAT_FL_RUNS_ROOT = REPO_ROOT / "data" / "processed" / "federated" / "gat_ecg_report_runs"


@dataclass
class GATCheckpointEvaluation:
    model_role: str
    checkpoint_path: Path
    metrics: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score_class1: np.ndarray
    bundle_path: Path
    bundle_meta: Dict[str, Any]


@torch.no_grad()
def run_gat_inference(
    model: torch.nn.Module,
    ecg: torch.Tensor,
    text: torch.Tensor,
    *,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns y_pred (argmax), y_score P(class=1), y_confidence max softmax."""
    model.eval()
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    labels_placeholder = torch.zeros(ecg.size(0), dtype=torch.long)
    loader = DataLoader(TensorDataset(ecg, text, labels_placeholder), batch_size=batch_size, shuffle=False)
    for e, t, _ in loader:
        logits = model(GATECGReportInput(e, t))
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        preds.append(pred.cpu().numpy())
        scores.append(probs[:, 1].cpu().numpy())
        confs.append(probs.max(dim=-1).values.cpu().numpy())
    y_pred = np.concatenate(preds).astype(int)
    y_score = np.concatenate(scores).astype(float)
    y_conf = np.concatenate(confs).astype(float)
    return y_pred, y_score, y_conf


def evaluate_checkpoint_on_bundle(
    checkpoint_path: Path,
    *,
    model_role: str,
    bundle_path: Optional[Path] = None,
    batch_size: int = 128,
) -> GATCheckpointEvaluation:
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    bundle_p = Path(bundle_path) if bundle_path else default_bundle_path()
    bundle = load_bundle(bundle_p)
    test = bundle["test"]
    labels = test["labels"].long()
    y_true = labels.cpu().numpy().astype(int)

    model = load_gat_ecg_report_checkpoint(ckpt, map_location="cpu")
    y_pred, y_score, _y_conf = run_gat_inference(
        model,
        test["ecg"].float(),
        test["clinical_text"].float(),
        batch_size=batch_size,
    )
    if y_pred.shape[0] != y_true.shape[0]:
        raise RuntimeError("Prediction/label count mismatch")

    metrics = compute_full_binary_metrics(y_true, y_pred, y_score_positive=y_score)
    metrics["checkpoint_path"] = str(ckpt.resolve())
    metrics["architecture_id"] = ARCHITECTURE_ID
    metrics["prediction_rule"] = "argmax(softmax(logits)); AUROC uses softmax[:,1]"

    return GATCheckpointEvaluation(
        model_role=model_role,
        checkpoint_path=ckpt.resolve(),
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_score_class1=y_score,
        bundle_path=bundle_p.resolve(),
        bundle_meta=dict(bundle.get("meta") or {}),
    )


def find_latest_fl_checkpoint(runs_root: Path | None = None) -> Optional[Path]:
    root = runs_root or GAT_FL_RUNS_ROOT
    if not root.is_dir():
        return None
    candidates: List[Tuple[float, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        ck = child / "global_federated_checkpoint.pt"
        if ck.is_file():
            candidates.append((ck.stat().st_mtime, ck))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_evaluation_manifest(
    *,
    centralized: Optional[GATCheckpointEvaluation],
    federated: Optional[GATCheckpointEvaluation],
    seed: int,
    output_dir: Path,
    fl_selection_note: str,
) -> Dict[str, Any]:
    import platform
    import sys

    import sklearn

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir.resolve()),
        "class_mapping": {"0": "NORM", "1": "MI/STTC"},
        "positive_class": {"index": 1, "label": "MI/STTC"},
        "random_seed_config": seed,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "bundle_path": str(centralized.bundle_path if centralized else federated.bundle_path if federated else ""),
            "split": "test (precomputed embeddings from PTB-XL test fold per bundle meta)",
            "smoke_caps": (centralized or federated).bundle_meta.get("smoke_caps") if (centralized or federated) else None,
            "split_counts": (centralized or federated).bundle_meta.get("split_counts")
            if (centralized or federated)
            else None,
            "test_split_mode": (centralized or federated).bundle_meta.get("test_split_mode")
            if (centralized or federated)
            else None,
            "split_policy": (centralized or federated).bundle_meta.get("split_policy") if (centralized or federated) else None,
            "target_definition": (centralized or federated).bundle_meta.get("target_definition") if (centralized or federated) else None,
        },
        "models": {
            "gat_centralized": {
                "checkpoint": str(centralized.checkpoint_path) if centralized else None,
                "role": "centralized_gat_ecg_report_fusion",
            },
            "gat_federated_global": {
                "checkpoint": str(federated.checkpoint_path) if federated else None,
                "selection_note": fl_selection_note,
                "role": "fedavg_global_head_over_frozen_embeddings",
            },
        },
        "comparable_examples": bool(
            centralized
            and federated
            and centralized.metrics.get("n_samples") == federated.metrics.get("n_samples")
        ),
        "research_disclaimer": (
            "PTB-XL proxy classification (NORM vs MI/STTC) is for research demonstration only. "
            "Not clinically validated diagnosis."
        ),
    }


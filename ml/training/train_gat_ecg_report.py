"""
Train two-modality GAT on frozen ECG-report encoder embeddings (PTB-XL in-cohort).

Artifacts: data/processed/gat_ecg_report/ and models/gat_ecg_report/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.ptbxl_ecg_report import PTBXLECGReportDataset  # noqa: E402
from ml.evaluation.calibration import CalibrationInputError, compute_calibration_metrics  # noqa: E402
from ml.evaluation.ecg_metrics import compute_binary_metrics  # noqa: E402
from ml.evaluation.tabular_metrics import find_best_youden_threshold  # noqa: E402
from ml.training.gat_cohort import (  # noqa: E402
    apply_head_record_caps,
    canonical_ecg_report_partitions,
    fl_canonical_test_ecg_ids,
    sorted_ecg_ids,
    verify_test_cohort_against_reference,
)
from ml.models.ecg_report_fusion import load_ecg_report_checkpoint  # noqa: E402
from ml.models.gat_ecg_report_fusion import (  # noqa: E402
    ARCHITECTURE_ID,
    GATECGReportConfig,
    GATECGReportInput,
    MedShieldGATECGReportNet,
    save_gat_ecg_report_checkpoint,
)
from ml.preprocessing.ecg_normalization import ECGNormalizationStats  # noqa: E402
from ml.ptbxl.ecg_report_index import build_ecg_report_index  # noqa: E402
from ml.ptbxl.splits import SPLIT_POLICY_V1, assign_split, verify_patient_disjoint  # noqa: E402
from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline  # noqa: E402
from ml.training.train_ecg_report_fusion import (  # noqa: E402
    PTBXL_ROOT,
    TEXT_MAX_LEN,
    make_collate_fn,
    partition_records,
    patient_counts,
    fit_train_normalization,
)
from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig  # noqa: E402

ARTIFACT_ROOT = REPO_ROOT / "data" / "processed" / "gat_ecg_report"
DEPLOY_DIR = REPO_ROOT / "models" / "gat_ecg_report"
ENCODER_MANIFEST = REPO_ROOT / "models" / "ecg_report_fusion" / "manifest.json"
MANIFEST_NAME = "medshield_fusion_manifest.json"
FL_EXPERIMENT_A_REPORT = (
    REPO_ROOT / "data/processed/federated/ecg_report_experiments/full_frozen_head_s42/report.json"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder_bundle(device: torch.device):
    manifest = json.loads(ENCODER_MANIFEST.read_text(encoding="utf-8"))
    ckpt = REPO_ROOT / manifest["checkpoint_path"]
    norm = REPO_ROOT / manifest["normalization_path"]
    if not ckpt.is_file() or not norm.is_file():
        raise FileNotFoundError("ECG-report encoder artifacts missing")
    text_cfg = ClinicalTextEncoderConfig(max_sequence_length=TEXT_MAX_LEN)
    pipe, _ = ClinicalTextEncodingPipeline.from_pretrained(config=text_cfg, device=device)
    model = load_ecg_report_checkpoint(ckpt, pipe.encoder, map_location="cpu")
    model.to(device)
    model.eval()
    if hasattr(pipe, "encoder"):
        pipe.encoder.to(device)
    for p in model.parameters():
        p.requires_grad = False
    norm_stats = ECGNormalizationStats.load(norm)
    return model, pipe, norm_stats, manifest


@torch.no_grad()
def extract_embeddings(model, pipe, norm_stats, records, ptbxl_root, device, batch_size=32):
    collate = make_collate_fn(pipe.tokenizer, TEXT_MAX_LEN)
    ds = PTBXLECGReportDataset(records, ptbxl_root, norm_stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    ecg_embs, text_embs, labels = [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"]
        ecg_emb = model.ecg_encoder(x)
        text_emb = model.text_encoder(input_ids, attention_mask)
        ecg_embs.append(ecg_emb.cpu())
        text_embs.append(text_emb.cpu())
        labels.append(y)
    return (
        torch.cat(ecg_embs),
        torch.cat(text_embs),
        torch.cat(labels),
    )


def train_gat(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for ecg, text, y in loader:
        ecg, text, y = ecg.to(device), text.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(GATECGReportInput(ecg, text))
        loss = criterion(logits, y.long())
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_gat(model, loader, device):
    model.eval()
    ys, scores = [], []
    for ecg, text, y in loader:
        ecg, text = ecg.to(device), text.to(device)
        logits = model(GATECGReportInput(ecg, text))
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.append(y.numpy())
        scores.append(prob)
    return np.concatenate(ys), np.concatenate(scores)


def enrich_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    base = compute_binary_metrics(y_true, y_score, threshold=threshold)
    cm = base["confusion_matrix"]
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    base["n_samples"] = int(n)
    base["accuracy"] = float((tp + tn) / n) if n else float("nan")
    sens = base.get("sensitivity", float("nan"))
    spec = base.get("specificity", float("nan"))
    if not math.isnan(sens) and not math.isnan(spec):
        base["balanced_accuracy"] = float((sens + spec) / 2.0)
    else:
        base["balanced_accuracy"] = float("nan")
    base["applied_threshold"] = float(threshold)
    try:
        cal = compute_calibration_metrics(y_true, y_score)
        base["brier_score"] = cal.get("brier_score")
        base["ece"] = cal.get("expected_calibration_error", {}).get("ece")
    except CalibrationInputError as exc:
        base["calibration_note"] = str(exc)
    return base


@dataclass
class GATTrainingConfig:
    ptbxl_root: Path = PTBXL_ROOT
    seed: int = 42
    batch_size: int = 64
    epochs: int = 12
    patience: int = 4
    lr: float = 1e-3
    device: str = "auto"
    smoke: bool = False
    max_train: int | None = None
    max_val: int | None = None
    max_test: int | None = None
    output_dir: Path | None = None
    experiment_label: str | None = None
    skip_deploy: bool = False
    verify_fl_test_cohort: bool = True


def run_gat_ecg_report_training(cfg: GATTrainingConfig) -> dict[str, Any]:
    if not cfg.ptbxl_root.exists() or not ENCODER_MANIFEST.is_file():
        raise FileNotFoundError("PTB-XL or encoder manifest missing")

    set_seed(cfg.seed)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    records, index_stats = build_ecg_report_index(cfg.ptbxl_root)
    disjoint = verify_patient_disjoint(records)
    parts, _index_stats2 = canonical_ecg_report_partitions(cfg.ptbxl_root)
    assert index_stats == _index_stats2
    reference_test_ids = fl_canonical_test_ecg_ids(cfg.ptbxl_root)
    caps_applied = {
        "max_train": cfg.max_train,
        "max_val": cfg.max_val,
        "max_test": cfg.max_test,
    }

    max_train, max_val, max_test = cfg.max_train, cfg.max_val, cfg.max_test
    epochs, patience = cfg.epochs, cfg.patience
    if cfg.smoke:
        max_train = max_train or 128
        max_val = max_val or 64
        max_test = max_test or 64
        epochs = min(epochs, 3)
        patience = 1

    parts = apply_head_record_caps(
        parts, max_train=max_train, max_val=max_val, max_test=max_test
    )
    cohort_check = verify_test_cohort_against_reference(parts["test"], reference_test_ids)
    if cfg.verify_fl_test_cohort and max_test is None and not cohort_check["match"]:
        raise ValueError(f"Test cohort mismatch vs FL canonical split: {cohort_check}")

    if cfg.output_dir is not None:
        run_dir = Path(cfg.output_dir).resolve()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (ARTIFACT_ROOT / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = cfg.experiment_label or run_dir.name

    split_manifest = {
        "split_policy": SPLIT_POLICY_V1,
        "train_ecg_ids": sorted_ecg_ids(parts["train"]),
        "validation_ecg_ids": sorted_ecg_ids(parts["validation"]),
        "test_ecg_ids": sorted_ecg_ids(parts["test"]),
        "test_ecg_id_fingerprint": cohort_check["test_ecg_id_fingerprint"],
        "fl_canonical_test_match": cohort_check["match"],
    }
    (run_dir / "split_ecg_ids.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    encoder_model, pipe, _norm_from_encoder, enc_manifest = load_encoder_bundle(device)
    norm = fit_train_normalization(cfg.ptbxl_root, parts["train"], None)
    norm.save(run_dir / "normalization.json")

    embed_bs = cfg.batch_size
    train_ecg, train_text, train_y = extract_embeddings(
        encoder_model, pipe, norm, parts["train"], cfg.ptbxl_root, device, embed_bs
    )
    val_ecg, val_text, val_y = extract_embeddings(
        encoder_model, pipe, norm, parts["validation"], cfg.ptbxl_root, device, embed_bs
    )
    test_ecg, test_text, test_y = extract_embeddings(
        encoder_model, pipe, norm, parts["test"], cfg.ptbxl_root, device, embed_bs
    )

    torch.save(
        {"train": (train_ecg, train_text, train_y), "encoder_manifest": enc_manifest},
        run_dir / "embedding_cache_train.pt",
    )

    train_loader = DataLoader(
        TensorDataset(train_ecg, train_text, train_y),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(val_ecg, val_text, val_y), batch_size=cfg.batch_size)
    test_loader = DataLoader(TensorDataset(test_ecg, test_text, test_y), batch_size=cfg.batch_size)

    gat = MedShieldGATECGReportNet(GATECGReportConfig(trained=False)).to(device)
    opt = torch.optim.Adam(gat.parameters(), lr=cfg.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_auroc = -1.0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        tr_loss = train_gat(gat, train_loader, opt, crit, device)
        y_v, s_v = evaluate_gat(gat, val_loader, device)
        val_m = compute_binary_metrics(y_v, s_v)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val": val_m})
        va = val_m.get("auroc", float("nan"))
        if va == va and va > best_auroc:
            best_auroc = va
            best_state = {k: v.cpu().clone() for k, v in gat.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        gat.load_state_dict(best_state)
    gat.config = GATECGReportConfig(trained=True)

    y_v, s_v = evaluate_gat(gat, val_loader, device)
    y_t, s_t = evaluate_gat(gat, test_loader, device)
    youden_thr, youden_j, _grid = find_best_youden_threshold(y_v, s_v)
    val_metrics = enrich_binary_metrics(y_v, s_v, youden_thr)
    test_metrics_youden = enrich_binary_metrics(y_t, s_t, youden_thr)
    test_metrics_default = compute_binary_metrics(y_t, s_t, threshold=0.5)

    with torch.no_grad():
        baseline_scores = []
        baseline_ys = []
        for batch in DataLoader(
            PTBXLECGReportDataset(parts["test"], cfg.ptbxl_root, norm),
            batch_size=cfg.batch_size,
            collate_fn=make_collate_fn(pipe.tokenizer, TEXT_MAX_LEN),
        ):
            x = batch["signal"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logit, _, _ = encoder_model(x, ids, mask)
            baseline_scores.append(torch.sigmoid(logit).cpu().numpy())
            baseline_ys.append(batch["label"].numpy())
        baseline_y = np.concatenate(baseline_ys)
        baseline_s = np.concatenate(baseline_scores)
        val_baseline_scores = []
        val_baseline_ys = []
        for batch in DataLoader(
            PTBXLECGReportDataset(parts["validation"], cfg.ptbxl_root, norm),
            batch_size=cfg.batch_size,
            collate_fn=make_collate_fn(pipe.tokenizer, TEXT_MAX_LEN),
        ):
            x = batch["signal"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logit, _, _ = encoder_model(x, ids, mask)
            val_baseline_scores.append(torch.sigmoid(logit).cpu().numpy())
            val_baseline_ys.append(batch["label"].numpy())
        baseline_thr, _, _ = find_best_youden_threshold(
            np.concatenate(val_baseline_ys),
            np.concatenate(val_baseline_scores),
        )
        baseline_metrics = enrich_binary_metrics(baseline_y, baseline_s, baseline_thr)

    ckpt_path = run_dir / "best_gat_checkpoint.pt"
    save_gat_ecg_report_checkpoint(
        ckpt_path,
        gat,
        extra={
            "selection_metric": "val_auroc",
            "best_val_auroc": best_auroc,
            "encoder_manifest": enc_manifest,
            "index_stats": index_stats,
            "experiment_label": run_id,
        },
    )

    if not cfg.skip_deploy:
        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        deploy_ckpt = DEPLOY_DIR / "best_gat_checkpoint.pt"
        save_gat_ecg_report_checkpoint(deploy_ckpt, gat, extra={"run_id": run_id})
        fusion_manifest = {
            "manifest_version": "1",
            "checkpoint_relative_path": "best_gat_checkpoint.pt",
            "architecture_id": ARCHITECTURE_ID,
            "architecture_version": "1.0.0",
            "model_version_tag": f"gat-ecg-report-{run_id}",
            "target_definition": (
                "PTB-XL proxy: NORM (0) vs MI/STTC (1). NOT validated clinical heart-disease diagnosis."
            ),
            "prediction_type": "proxy_evaluation",
            "require_trained_checkpoint": True,
            "encoder_manifest_path": "models/ecg_report_fusion/manifest.json",
            "modalities": {"ecg": 128, "clinical_text": 128},
        }
        (DEPLOY_DIR / MANIFEST_NAME).write_text(json.dumps(fusion_manifest, indent=2), encoding="utf-8")

    exp_a_ref = None
    if FL_EXPERIMENT_A_REPORT.is_file():
        exp_a_ref = json.loads(FL_EXPERIMENT_A_REPORT.read_text(encoding="utf-8")).get(
            "unified_comparison_same_records", {}
        )

    report = {
        "experiment": "centralized_gat_on_frozen_ecg_report_embeddings",
        "experiment_label": run_id,
        "architecture_id": ARCHITECTURE_ID,
        "training_mode": {
            "description": (
                "GAT head trained on frozen ECG-report encoder embeddings; "
                "not federated and not end-to-end encoder training."
            ),
            "encoder_frozen": True,
            "gat_trainable": True,
        },
        "warning": "Research proxy task on PTB-XL paired ECG+report embeddings.",
        "config": {
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": cfg.lr,
            "device": str(device),
            "caps_applied": caps_applied,
            "smoke_run": cfg.smoke,
        },
        "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "encoder_manifest_path": str(ENCODER_MANIFEST.relative_to(REPO_ROOT)).replace("\\", "/"),
        "normalization_path": str((run_dir / "normalization.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "best_val_auroc_for_checkpoint_selection": best_auroc,
        "evaluation": {
            "threshold_policy": "youden_j_on_validation_per_model",
            "youden_threshold": youden_thr,
            "youden_j": youden_j,
            "validation": val_metrics,
            "test": test_metrics_youden,
            "test_default_threshold_0_5": test_metrics_default,
        },
        "test_metrics_ecg_report_baseline_on_test": baseline_metrics,
        "unified_comparison_same_records": {
            "protocol": "identical_val_test_records",
            "normalization": "gat_train_fit_only",
            "val_n": int(len(val_y)),
            "test_n": int(len(test_y)),
            "threshold_policy": "youden_j_on_validation_per_model",
            "gat_global_model": {"validation": val_metrics, "test": test_metrics_youden, "youden_threshold": youden_thr},
        },
        "fl_experiment_a_reference": exp_a_ref,
        "cohort_verification": cohort_check,
        "split_ecg_ids_path": str((run_dir / "split_ecg_ids.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "test_n": int(len(test_y)),
        "val_n": int(len(val_y)),
        "train_n": int(len(train_y)),
        "history": history,
        "patient_disjoint": disjoint,
        "partition_counts": {k: len(v) for k, v in parts.items()},
        "partition_patients": {k: patient_counts(v) for k, v in parts.items()},
        "index_stats": index_stats,
        "legacy_report_fields": {
            "test_metrics_gat": test_metrics_default,
            "best_val_auroc": best_auroc,
        },
        "completed_training": True,
        "deploy_updated": not cfg.skip_deploy,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "test_auroc": test_metrics_youden.get("auroc"),
        "test_accuracy": test_metrics_youden.get("accuracy"),
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-modality GAT ECG+report training")
    parser.add_argument("--ptbxl-root", type=Path, default=PTBXL_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory (default: timestamp under data/processed/gat_ecg_report/runs/)",
    )
    parser.add_argument("--experiment-label", type=str, default=None)
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Do not overwrite models/gat_ecg_report deploy checkpoint/manifest",
    )
    args = parser.parse_args()

    try:
        result = run_gat_ecg_report_training(
            GATTrainingConfig(
                ptbxl_root=args.ptbxl_root,
                seed=args.seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                device=args.device,
                smoke=args.smoke,
                max_train=args.max_train,
                max_val=args.max_val,
                max_test=args.max_test,
                output_dir=args.output_dir,
                experiment_label=args.experiment_label,
                skip_deploy=args.skip_deploy,
            )
        )
    except FileNotFoundError as exc:
        print(f"BLOCKER: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"BLOCKER: {exc}", file=sys.stderr)
        return 3

    print(
        json.dumps(
            {
                "run_dir": result["run_dir"],
                "test_auroc_gat": result["test_auroc"],
                "test_accuracy_gat_youden": result["test_accuracy"],
                "deploy_updated": not args.skip_deploy,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

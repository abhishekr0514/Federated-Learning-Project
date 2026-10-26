"""
Train two-modality GAT on frozen ECG-report encoder embeddings (PTB-XL in-cohort).

Artifacts: data/processed/gat_ecg_report/ and models/gat_ecg_report/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    model = load_ecg_report_checkpoint(ckpt, pipe.encoder, map_location=device)
    model.eval()
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
    args = parser.parse_args()

    if not args.ptbxl_root.exists() or not ENCODER_MANIFEST.is_file():
        print("BLOCKER: PTB-XL or encoder manifest missing", file=sys.stderr)
        return 2

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    records, index_stats = build_ecg_report_index(args.ptbxl_root)
    disjoint = verify_patient_disjoint(records)
    parts = partition_records(records)

    if args.smoke:
        args.max_train = args.max_train or 128
        args.max_val = args.max_val or 64
        args.max_test = args.max_test or 64
        args.epochs = min(args.epochs, 3)
        args.patience = 1

    for key in ("train", "validation", "test"):
        cap = getattr(args, f"max_{key.replace('validation', 'val')}", None)
        if cap:
            parts[key] = parts[key][:cap]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ARTIFACT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder_model, pipe, _norm_from_encoder, enc_manifest = load_encoder_bundle(device)
    norm = fit_train_normalization(args.ptbxl_root, parts["train"], None)
    norm.save(run_dir / "normalization.json")

    train_ecg, train_text, train_y = extract_embeddings(
        encoder_model, pipe, norm, parts["train"], args.ptbxl_root, device, args.batch_size
    )
    val_ecg, val_text, val_y = extract_embeddings(
        encoder_model, pipe, norm, parts["validation"], args.ptbxl_root, device, args.batch_size
    )
    test_ecg, test_text, test_y = extract_embeddings(
        encoder_model, pipe, norm, parts["test"], args.ptbxl_root, device, args.batch_size
    )

    torch.save(
        {
            "train": (train_ecg, train_text, train_y),
            "encoder_manifest": enc_manifest,
        },
        run_dir / "embedding_cache_train.pt",
    )

    train_loader = DataLoader(
        TensorDataset(train_ecg, train_text, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(val_ecg, val_text, val_y), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_ecg, test_text, test_y), batch_size=args.batch_size)

    gat = MedShieldGATECGReportNet(GATECGReportConfig(trained=False)).to(device)
    opt = torch.optim.Adam(gat.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_auroc = -1.0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
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
            if stale >= args.patience:
                break

    if best_state:
        gat.load_state_dict(best_state)
    gat.config = GATECGReportConfig(trained=True)

    y_t, s_t = evaluate_gat(gat, test_loader, device)
    test_metrics = compute_binary_metrics(y_t, s_t)
    try:
        test_metrics["calibration"] = compute_calibration_metrics(y_t, s_t)
    except CalibrationInputError as exc:
        test_metrics["calibration"] = {"error": str(exc)}

    with torch.no_grad():
        baseline_scores = []
        baseline_ys = []
        for batch in DataLoader(
            PTBXLECGReportDataset(parts["test"], args.ptbxl_root, norm),
            batch_size=args.batch_size,
            collate_fn=make_collate_fn(pipe.tokenizer, TEXT_MAX_LEN),
        ):
            x = batch["signal"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logit, _, _ = encoder_model(x, ids, mask)
            baseline_scores.append(torch.sigmoid(logit).cpu().numpy())
            baseline_ys.append(batch["label"].numpy())
        baseline_metrics = compute_binary_metrics(
            np.concatenate(baseline_ys), np.concatenate(baseline_scores)
        )

    ckpt_path = run_dir / "best_gat_checkpoint.pt"
    save_gat_ecg_report_checkpoint(
        ckpt_path,
        gat,
        extra={
            "selection_metric": "val_auroc",
            "best_val_auroc": best_auroc,
            "encoder_manifest": enc_manifest,
            "index_stats": index_stats,
        },
    )

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

    report = {
        "warning": "Research proxy task on PTB-XL paired ECG+report embeddings.",
        "best_val_auroc": best_auroc,
        "test_metrics_gat": test_metrics,
        "test_metrics_ecg_report_baseline": baseline_metrics,
        "test_n": int(len(test_y)),
        "history": history,
        "patient_disjoint": disjoint,
        "partition_counts": {k: len(v) for k, v in parts.items()},
        "partition_patients": {k: patient_counts(v) for k, v in parts.items()},
        "smoke_run": args.smoke,
        "completed_training": True,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"run_dir": str(run_dir), "deploy_dir": str(DEPLOY_DIR), "test_auroc_gat": test_metrics.get("auroc"), "test_auroc_baseline": baseline_metrics.get("auroc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

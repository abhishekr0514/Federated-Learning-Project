"""
Two-modality PTB-XL ECG + machine report training (NOT three-modality GAT).

Artifacts: data/processed/ecg_report_fusion/ (gitignored)
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.ptbxl_ecg import load_ecg_mv  # noqa: E402
from ml.datasets.ptbxl_ecg_report import PTBXLECGReportDataset  # noqa: E402
from ml.evaluation.calibration import CalibrationInputError, compute_calibration_metrics  # noqa: E402
from ml.evaluation.ecg_metrics import compute_binary_metrics  # noqa: E402
from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig  # noqa: E402
from ml.models.ecg_report_fusion import (  # noqa: E402
    ECGReportFusionClassifier,
    ECGReportFusionConfig,
    save_ecg_report_checkpoint,
)
from ml.preprocessing.clinical_text_privacy import mask_clinical_text_for_encoding  # noqa: E402
from ml.preprocessing.ecg_normalization import (  # noqa: E402
    ECGNormalizationStats,
    fit_normalization_from_signals,
)
from ml.ptbxl.ecg_report_index import build_ecg_report_index  # noqa: E402
from ml.ptbxl.splits import SPLIT_POLICY_V1, assign_split, verify_patient_disjoint  # noqa: E402
from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline  # noqa: E402

ARTIFACT_ROOT = REPO_ROOT / "data" / "processed" / "ecg_report_fusion"
PTBXL_ROOT = REPO_ROOT / "data" / "raw" / "ptb-xl"
DATASET_SHA256 = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"
TEXT_MAX_LEN = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_records(records: list[dict]) -> dict[str, list[dict]]:
    parts: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for rec in records:
        split = assign_split(rec["strat_fold"])
        if split:
            parts[split].append(rec)
    return parts


def patient_counts(records: list[dict]) -> int:
    return len({str(r["patient_id"]) for r in records})


def fit_train_normalization(
    ptbxl_root: Path, train_records: list[dict], max_records: int | None
) -> ECGNormalizationStats:
    subset = train_records if max_records is None else train_records[:max_records]
    signals = [load_ecg_mv(ptbxl_root, r["filename_lr"]) for r in subset]
    return fit_normalization_from_signals(signals)


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        signals = torch.stack([b["signal"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        masked = [mask_clinical_text_for_encoding(b["report"]) for b in batch]
        tokenized = tokenizer(
            masked,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "signal": signals,
            "label": labels,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "ecg_id": [b["ecg_id"] for b in batch],
        }

    return collate


@torch.no_grad()
def evaluate(model: ECGReportFusionClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, scores = [], []
    for batch in loader:
        x = batch["signal"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].cpu().numpy()
        logit, _, _ = model(x, input_ids, attention_mask)
        prob = torch.sigmoid(logit).cpu().numpy()
        ys.append(y)
        scores.append(prob)
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    metrics = compute_binary_metrics(y_true, y_score)
    try:
        metrics["calibration"] = compute_calibration_metrics(y_true, y_score)
    except CalibrationInputError as exc:
        metrics["calibration"] = {"error": str(exc), "note": "Calibration skipped for this split."}
    return metrics


def train_one_epoch(
    model: ECGReportFusionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logit, _, _ = model(x, input_ids, attention_mask)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def freeze_bert(model: ECGReportFusionClassifier) -> None:
    for param in model.text_encoder.bert.parameters():
        param.requires_grad = False


def main() -> int:
    parser = argparse.ArgumentParser(description="PTB-XL ECG + report two-modality fusion")
    parser.add_argument("--ptbxl-root", type=Path, default=PTBXL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--norm-fit-cap", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    if not args.ptbxl_root.exists():
        print("ERROR: PTB-XL root missing.", file=sys.stderr)
        return 2

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records, index_stats = build_ecg_report_index(args.ptbxl_root)
    disjoint = verify_patient_disjoint(records)
    parts = partition_records(records)

    if args.smoke:
        args.max_train = args.max_train or 24
        args.max_val = args.max_val or 12
        args.max_test = args.max_test or 12
        args.norm_fit_cap = args.norm_fit_cap or 16
        args.epochs = min(args.epochs, 2)
        args.patience = 1
        args.batch_size = min(args.batch_size, 8)

    for key in ("train", "validation", "test"):
        cap_name = "max_val" if key == "validation" else f"max_{key}"
        cap = getattr(args, cap_name, None)
        if cap:
            parts[key] = parts[key][:cap]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.artifact_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    label_dist = {"0": 0, "1": 0}
    for rec in records:
        label_dist[str(rec["label"])] += 1

    manifest = {
        "experiment": "two_modality_ecg_report_ptbxl",
        "not_three_modality_gat": True,
        "protocol_version": index_stats["protocol_version"],
        "split_policy": SPLIT_POLICY_V1,
        "index_stats": index_stats,
        "label_distribution_all_paired": label_dist,
        "patient_disjoint": disjoint,
        "partition_record_counts": {k: len(v) for k, v in parts.items()},
        "partition_patient_counts": {k: patient_counts(v) for k, v in parts.items()},
        "seed": args.seed,
        "dataset_sha256": DATASET_SHA256,
        "text_max_sequence_length": TEXT_MAX_LEN,
        "text_model": "emilyalsentzer/Bio_ClinicalBERT",
        "bert_frozen": True,
        "smoke_run": args.smoke,
        "device": str(device),
        "pytorch_version": torch.__version__,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    text_cfg = ClinicalTextEncoderConfig(max_sequence_length=TEXT_MAX_LEN)
    try:
        pipe, text_meta = ClinicalTextEncodingPipeline.from_pretrained(config=text_cfg, device=device)
    except Exception as exc:
        print(f"ERROR: Clinical text encoder unavailable: {exc}", file=sys.stderr)
        return 3

    manifest["text_load_meta"] = text_meta
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    norm = fit_train_normalization(args.ptbxl_root, parts["train"], args.norm_fit_cap)
    norm.save(run_dir / "normalization.json")

    collate = make_collate_fn(pipe.tokenizer, TEXT_MAX_LEN)
    train_ds = PTBXLECGReportDataset(parts["train"], args.ptbxl_root, norm)
    val_ds = PTBXLECGReportDataset(parts["validation"], args.ptbxl_root, norm)
    test_ds = PTBXLECGReportDataset(parts["test"], args.ptbxl_root, norm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    fusion_cfg = ECGReportFusionConfig(trained=False)
    model = ECGReportFusionClassifier(pipe.encoder, config=fusion_cfg).to(device)
    freeze_bert(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -1.0
    best_epoch = -1
    stale = 0
    history = []
    best_path = run_dir / "best_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        history.append(row)
        val_auroc = val_metrics.get("auroc", float("nan"))
        if val_auroc == val_auroc and val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            stale = 0
            model.config = ECGReportFusionConfig(trained=True)
            save_ecg_report_checkpoint(
                best_path,
                model,
                extra_metadata={
                    "epoch": epoch,
                    "selection_metric": "val_auroc",
                    "val_auroc": val_auroc,
                    "protocol_version": index_stats["protocol_version"],
                    "smoke_run": args.smoke,
                },
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    if not best_path.exists():
        print("ERROR: No checkpoint saved (validation never improved).", file=sys.stderr)
        return 4

    loaded = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded["state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    report = {
        "warning": (
            "PTB-XL in-cohort ECG + report two-modality experiment. "
            "Proxy target NORM vs MI/STTC — NOT the ECG+text+tabular GAT product."
        ),
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "test_metrics": test_metrics,
        "history": history,
        "completed_training": True,
        "smoke_run": args.smoke,
        "calibration_note": (
            "See test_metrics.calibration when computed; scores are not guaranteed calibrated "
            "without post-hoc calibration on validation data."
        ),
    }
    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    deploy_manifest = {
        "architecture_id": fusion_cfg.architecture_id,
        "experiment": "two_modality_ecg_report_ptbxl",
        "checkpoint_path": str(best_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "normalization_path": str((run_dir / "normalization.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "run_manifest": str((run_dir / "manifest.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "trained": True,
        "three_modality_gat_checkpoint": False,
    }
    deploy_dir = REPO_ROOT / "models" / "ecg_report_fusion"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    with open(deploy_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(deploy_manifest, f, indent=2)

    print(
        json.dumps(
            {"run_dir": str(run_dir), "best_val_auroc": best_auroc, "test": test_metrics},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

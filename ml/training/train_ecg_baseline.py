"""
Reproducible PTB-XL ECG baseline training entry point (Prompt 12).

Artifacts: data/processed/ecg_baseline/ (gitignored)
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

from ml.datasets.ptbxl_ecg import PTBXLECGDataset, load_ecg_mv  # noqa: E402
from ml.evaluation.calibration import CalibrationInputError, compute_calibration_metrics  # noqa: E402
from ml.evaluation.ecg_metrics import compute_binary_metrics  # noqa: E402
from ml.models.ecg_encoder import ECGBinaryClassifier  # noqa: E402
from ml.preprocessing.ecg_normalization import (  # noqa: E402
    ECGNormalizationStats,
    fit_normalization_from_signals,
)
from ml.ptbxl.labels import PROTOCOL_VERSION, build_labeled_index  # noqa: E402
from ml.ptbxl.splits import SPLIT_POLICY_V1, assign_split, verify_patient_disjoint  # noqa: E402

ARTIFACT_ROOT = REPO_ROOT / "data" / "processed" / "ecg_baseline"
PTBXL_ROOT = REPO_ROOT / "data" / "raw" / "ptb-xl"
DATASET_SHA256 = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"


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


def fit_train_normalization(ptbxl_root: Path, train_records: list[dict], max_records: int | None) -> ECGNormalizationStats:
    subset = train_records if max_records is None else train_records[:max_records]
    signals = [load_ecg_mv(ptbxl_root, r["filename_lr"]) for r in subset]
    return fit_normalization_from_signals(signals)


@torch.no_grad()
def evaluate(model: ECGBinaryClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, scores = [], []
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].cpu().numpy()
        logit, _ = model(x)
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


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logit, _ = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="PTB-XL ECG CNN-BiLSTM baseline")
    parser.add_argument("--ptbxl-root", type=Path, default=PTBXL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Tiny subset; not a full experiment")
    parser.add_argument("--norm-fit-cap", type=int, default=None, help="Max train records for norm fit (smoke uses 16)")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    if not args.ptbxl_root.exists():
        print("ERROR: PTB-XL root missing. Cannot train.", file=sys.stderr)
        return 2

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )

    records, label_stats = build_labeled_index(args.ptbxl_root)
    disjoint = verify_patient_disjoint(records)
    parts = partition_records(records)

    if args.smoke:
        args.max_train = args.max_train or 32
        args.max_val = args.max_val or 16
        args.max_test = args.max_test or 16
        args.norm_fit_cap = args.norm_fit_cap or 16
        args.epochs = min(args.epochs, 2)
        args.patience = 1

    for key in ("train", "validation", "test"):
        cap = getattr(args, f"max_{key.replace('validation', 'val')}", None)
        if cap:
            parts[key] = parts[key][:cap]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.artifact_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "split_policy": SPLIT_POLICY_V1,
        "label_stats": label_stats,
        "patient_disjoint": disjoint,
        "partition_counts": {k: len(v) for k, v in parts.items()},
        "seed": args.seed,
        "dataset_sha256": DATASET_SHA256,
        "smoke_run": args.smoke,
        "device": str(device),
        "pytorch_version": torch.__version__,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    norm = fit_train_normalization(args.ptbxl_root, parts["train"], args.norm_fit_cap)
    norm.save(run_dir / "normalization.json")

    train_ds = PTBXLECGDataset(parts["train"], args.ptbxl_root, norm)
    val_ds = PTBXLECGDataset(parts["validation"], args.ptbxl_root, norm)
    test_ds = PTBXLECGDataset(parts["test"], args.ptbxl_root, norm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ECGBinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -1.0
    best_epoch = -1
    stale = 0
    history = []

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "selection_metric": "val_auroc",
                    "val_auroc": val_auroc,
                    "protocol_version": PROTOCOL_VERSION,
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    # Load best for one-time test evaluation
    ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    report = {
        "warning": (
            "PTB-XL proxy-task metrics (NORM vs MI/STTC per DATA_CONTRACT). "
            "NOT validated clinical heart-disease diagnostic performance."
        ),
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "test_metrics": test_metrics,
        "history": history,
        "completed_training": True,
        "smoke_run": args.smoke,
    }
    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"run_dir": str(run_dir), "best_val_auroc": best_auroc, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

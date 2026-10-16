"""
Reproducible Tabular Baseline Training Entry Point.

Protocol: docs/research/TABULAR_BASELINE_PROTOCOL_v1.md
Schema: medshield.tabular.cardio_survey v1.0.0
Architecture: TabularBinaryClassifier (NeuralTabularEncoder + Linear(64, 1))

Artifacts: data/processed/tabular_baseline/runs/<timestamp>/ (gitignored)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.cardio_tabular import (  # noqa: E402
    CardioTabularDataset,
    load_and_filter_cardio_cohort,
    save_split_manifest_hashed,
    stratified_cardio_split,
)
from ml.evaluation.tabular_metrics import (  # noqa: E402
    compute_tabular_metrics,
    find_best_youden_threshold,
)
from ml.tabular.feature_schema import (  # noqa: E402
    FEATURE_NAMES,
    N_TABULAR_FEATURES,
    SCHEMA_ID,
    SCHEMA_VERSION,
)
from ml.tabular.preprocessing import TabularPreprocessor  # noqa: E402

PROTOCOL_ID = "medshield.tabular.cardio_survey.baseline"
PROTOCOL_VERSION = "1.0.0"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "datasets" / "cardio_train.csv"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "data" / "processed" / "tabular_baseline"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "cardio_patient_split_seed42.json"
EXPECTED_DATASET_SHA256 = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def verify_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return float(total_loss / max(n, 1))


def evaluate_model(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["target"].cpu().numpy()
            logits, _ = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(y)
            probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)


def run_tabular_experiment(
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    seed: int = 42,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    patience: int = 5,
    device_name: str = "auto",
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute complete Prompt 17/18 Tabular Baseline Protocol."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from ml.models.tabular_encoder import TabularBinaryClassifier
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to execute the tabular experiment. "
            "Please ensure torch is installed in your Python environment."
        ) from exc

    set_seed(seed)
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Cardio dataset not found at: {data_path}")

    actual_sha = verify_file_sha256(data_path)
    if actual_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset SHA256 mismatch. Expected {EXPECTED_DATASET_SHA256}, got {actual_sha}"
        )

    # 1. Load and filter cohort (excludes ap_lo > ap_hi)
    accepted_records, rejected_bp_ids, total_raw_rows = load_and_filter_cardio_cohort(data_path)

    # 2. Stratified split (70% train, 15% val, 15% test)
    train_ids, val_ids, test_ids = stratified_cardio_split(accepted_records, seed=seed)

    if smoke:
        # Smoke testing: cap to small disjoint subsets
        epochs = min(epochs, 2)
        patience = 1
        smoke_train_ids = set(list(train_ids)[:256])
        smoke_val_ids = set(list(val_ids)[:128])
        smoke_test_ids = set(list(test_ids)[:128])
        train_records = [r for r in accepted_records if r["record_id"] in smoke_train_ids]
        val_records = [r for r in accepted_records if r["record_id"] in smoke_val_ids]
        test_records = [r for r in accepted_records if r["record_id"] in smoke_test_ids]
    else:
        train_records = [r for r in accepted_records if r["record_id"] in train_ids]
        val_records = [r for r in accepted_records if r["record_id"] in val_ids]
        test_records = [r for r in accepted_records if r["record_id"] in test_ids]

    train_pos = sum(1 for r in train_records if r["target"] == 1)
    train_neg = len(train_records) - train_pos
    val_pos = sum(1 for r in val_records if r["target"] == 1)
    val_neg = len(val_records) - val_pos
    test_pos = sum(1 for r in test_records if r["target"] == 1)
    test_neg = len(test_records) - test_pos

    # 3. Save hashed ID manifest under data/manifests/
    save_split_manifest_hashed(
        manifest_path,
        seed=seed,
        total_raw=total_raw_rows,
        rejected_bp_ids=rejected_bp_ids,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
    )

    # 4. Training-only preprocessing
    preprocessor = TabularPreprocessor()
    preprocessor.fit([r["features"] for r in train_records])

    # Transform partitions with frozen fitted preprocessor
    train_x = preprocessor.transform_batch([r["features"] for r in train_records])
    val_x = preprocessor.transform_batch([r["features"] for r in val_records])
    test_x = preprocessor.transform_batch([r["features"] for r in test_records])

    train_y = np.array([r["target"] for r in train_records], dtype=np.float32)
    val_y = np.array([r["target"] for r in val_records], dtype=np.float32)
    test_y = np.array([r["target"] for r in test_records], dtype=np.float32)

    # 5. DataLoaders
    train_ds = CardioTabularDataset(train_x, train_y)
    val_ds = CardioTabularDataset(val_x, val_y)
    test_ds = CardioTabularDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 6. Device and model setup
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = TabularBinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifact_root / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessor state
    preprocessor.save(run_dir / "preprocessor.json")

    best_val_auroc = -1.0
    best_epoch = -1
    best_model_weights = None
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        loss_val = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_val_true, y_val_prob = evaluate_model(model, val_loader, device)
        val_metrics = compute_tabular_metrics(y_val_true, y_val_prob, threshold=0.5)

        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_val,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_brier": val_metrics["brier_score"],
        }
        history.append(epoch_record)

        current_auroc = val_metrics["auroc"]
        if not math.isnan(current_auroc) and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_epoch = epoch
            stale_epochs = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_model_weights is None:
        best_model_weights = model.state_dict()
        best_epoch = epochs

    # Load best weights
    model.load_state_dict(best_model_weights)
    model.eval()

    # 7. Select decision threshold using validation predictions ONLY
    val_y_eval, val_probs_eval = evaluate_model(model, val_loader, device)
    best_threshold, best_youden_j, threshold_grid = find_best_youden_threshold(val_y_eval, val_probs_eval)

    # 8. Save checkpoint
    checkpoint_payload = {
        "architecture_id": "medshield.tabular.cardio_survey.classifier",
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": "val_auroc",
        "best_val_auroc": best_val_auroc,
        "val_selected_threshold": best_threshold,
        "val_youden_j": best_youden_j,
        "model_state_dict": best_model_weights,
        "torch_version": torch.__version__,
    }
    torch.save(checkpoint_payload, run_dir / "best_checkpoint.pt")

    # 9. Evaluate held-out test set ONCE with frozen threshold
    test_y_eval, test_probs_eval = evaluate_model(model, test_loader, device)
    test_metrics = compute_tabular_metrics(test_y_eval, test_probs_eval, threshold=best_threshold)

    # 10. Persist manifest and report
    manifest_data = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": actual_sha,
        "seed": seed,
        "smoke_run": smoke,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "counts": {
            "total_raw_rows": total_raw_rows,
            "rejected_invalid_bp": len(rejected_bp_ids),
            "accepted_rows": len(accepted_records),
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "test_rows": len(test_records),
            "train_class_distribution": {"class_0": train_neg, "class_1": train_pos},
            "val_class_distribution": {"class_0": val_neg, "class_1": val_pos},
            "test_class_distribution": {"class_0": test_neg, "class_1": test_pos},
        },
        "training_hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "early_stopping_patience": patience,
            "loss_function": "BCEWithLogitsLoss",
        },
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    report_data = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "warning_and_limitations": (
            "Cardiovascular survey-target baseline (cardio_train.csv). "
            "Target 'cardio' is a self-reported survey/registry outcome; "
            "this model is NOT a clinically validated diagnostic tool. "
            "Production inference remains disabled."
        ),
        "cohort": {
            "total_raw_rows": total_raw_rows,
            "excluded_invalid_bp_rows": len(rejected_bp_ids),
            "accepted_cohort_rows": len(accepted_records),
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "test_rows": len(test_records),
        },
        "model_selection": {
            "best_epoch": best_epoch,
            "best_val_auroc": best_val_auroc,
            "validation_selected_threshold": best_threshold,
            "validation_youden_j": best_youden_j,
            "threshold_selection_policy": "Youden J on validation set over grid [0.05, 0.95], frozen for test",
        },
        "primary_metric": {
            "name": "test_auroc",
            "value": test_metrics["auroc"],
        },
        "test_metrics": test_metrics,
        "training_history": history,
        "completed_training": True,
        "smoke_run": smoke,
        "run_directory": str(run_dir),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "selected_threshold": best_threshold,
        "test_metrics": test_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MedShield-FL Tabular Baseline Training")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Quick run on subset for software testing")
    args = parser.parse_args()

    print(f"Starting MedShield Tabular Baseline experiment (Protocol {PROTOCOL_VERSION})...")
    try:
        results = run_tabular_experiment(
            data_path=args.data_path,
            artifact_root=args.artifact_root,
            manifest_path=args.manifest_path,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
            smoke=args.smoke,
        )
        print("Experiment completed successfully!")
        print(json.dumps(results, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: Experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

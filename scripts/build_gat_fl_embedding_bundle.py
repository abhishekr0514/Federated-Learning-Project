#!/usr/bin/env python3
"""Build frozen-embedding bundle for two-client GAT federated research demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from ml.federated.gat_ecg_report_fl import (  # noqa: E402
    DEFAULT_BUNDLE_PATH,
    DEFAULT_DEPLOY_CHECKPOINT,
    build_bundle_from_tensors,
)
from ml.ptbxl.ecg_report_index import build_ecg_report_index  # noqa: E402
from ml.ptbxl.splits import SPLIT_POLICY_V1, verify_patient_disjoint  # noqa: E402
from ml.training.train_ecg_report_fusion import PTBXL_ROOT, partition_records  # noqa: E402
from ml.training.train_gat_ecg_report import extract_embeddings, load_encoder_bundle  # noqa: E402
from ml.training.train_ecg_report_fusion import fit_train_normalization  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GAT FL embedding bundle (frozen encoder outputs)")
    parser.add_argument("--ptbxl-root", type=Path, default=PTBXL_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap on train records after split (smoke/dev only). Omit for full train split.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional cap on test records after split (smoke/dev only). Omit for full fold-10 test.",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Use the entire held-out test split (fold 10). Mutually exclusive with --max-test.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reuse-train-embeddings",
        type=Path,
        default=None,
        help="embedding_cache_train.pt from a prior centralized GAT run (skips re-encoding train rows)",
    )
    args = parser.parse_args()

    if args.full_test and args.max_test is not None:
        print("BLOCKER: use either --full-test or --max-test, not both", file=sys.stderr)
        return 2
    max_train_cap = args.max_train
    max_test_cap = None if args.full_test else args.max_test

    if not args.ptbxl_root.is_dir():
        print(f"BLOCKER: PTB-XL root missing: {args.ptbxl_root}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    records, index_stats = build_ecg_report_index(args.ptbxl_root)
    disjoint = verify_patient_disjoint(records)
    if disjoint.get("patient_disjoint") is not True:
        print("BLOCKER: patient split not disjoint", file=sys.stderr)
        return 2

    parts = partition_records(records)
    train_recs = parts["train"]
    test_recs = parts["test"]
    test_recs_full_count = len(test_recs)
    train_recs_full_count = len(train_recs)
    if max_train_cap:
        train_recs = train_recs[:max_train_cap]
    if max_test_cap:
        test_recs = test_recs[:max_test_cap]

    encoder_model, pipe, _enc_norm, enc_manifest = load_encoder_bundle(device)
    norm = fit_train_normalization(args.ptbxl_root, parts["train"], None)

    if args.reuse_train_embeddings:
        cache = torch.load(args.reuse_train_embeddings, map_location="cpu", weights_only=False)
        train_ecg, train_text, train_y = cache["train"]
        n_train = int(train_y.size(0))
        train_recs = train_recs[:n_train]
        if max_train_cap:
            train_ecg = train_ecg[:max_train_cap]
            train_text = train_text[:max_train_cap]
            train_y = train_y[:max_train_cap]
            train_recs = train_recs[:max_train_cap]
    else:
        train_ecg, train_text, train_y = extract_embeddings(
            encoder_model, pipe, norm, train_recs, args.ptbxl_root, device, args.batch_size
        )
    test_ecg, test_text, test_y = extract_embeddings(
        encoder_model, pipe, norm, test_recs, args.ptbxl_root, device, args.batch_size
    )
    train_pids = [str(r["patient_id"]) for r in train_recs]
    test_pids = [str(r["patient_id"]) for r in test_recs]

    meta = {
        "bundle_version": "1",
        "bundle_relative_path": str(args.out.relative_to(REPO_ROOT)),
        "split_policy": SPLIT_POLICY_V1,
        "index_stats": index_stats,
        "encoder_manifest": enc_manifest,
        "init_checkpoint_path": str(DEFAULT_DEPLOY_CHECKPOINT),
        "target_definition": (
            "PTB-XL proxy: NORM (0) vs MI/STTC (1). NOT validated clinical heart-disease diagnosis."
        ),
        "split_counts": {
            "train_eligible_full_split": train_recs_full_count,
            "test_eligible_full_split": test_recs_full_count,
            "train_in_bundle": len(train_recs),
            "test_in_bundle": len(test_recs),
        },
        "test_split_mode": "full" if max_test_cap is None else "capped",
        "smoke_caps": {"max_train": max_train_cap, "max_test": max_test_cap},
    }
    build_bundle_from_tensors(
        train_ecg=train_ecg,
        train_text=train_text,
        train_y=train_y,
        train_patient_ids=train_pids,
        test_ecg=test_ecg,
        test_text=test_text,
        test_y=test_y,
        test_patient_ids=test_pids,
        meta=meta,
        out_path=args.out,
    )
    print(json.dumps({"ok": True, "out": str(args.out), "train_n": int(train_y.size(0)), "test_n": int(test_y.size(0))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

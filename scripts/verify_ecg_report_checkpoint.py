"""Verify a two-modality ECG+report checkpoint (not three-modality GAT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig
from ml.models.ecg_report_fusion import ARCHITECTURE_ID, load_ecg_report_checkpoint
from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "ecg_report_fusion" / "runs",
        help="Path to .pt file or runs directory (uses latest best_checkpoint.pt)",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path.is_dir():
        runs = sorted((ckpt_path / "runs").glob("*/best_checkpoint.pt") if (ckpt_path / "runs").exists() else ckpt_path.glob("*/best_checkpoint.pt"))
        if not runs:
            print("ERROR: No best_checkpoint.pt found under directory.", file=sys.stderr)
            return 2
        ckpt_path = runs[-1]

    if not ckpt_path.is_file():
        print(f"ERROR: Missing checkpoint: {ckpt_path}", file=sys.stderr)
        return 2

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if raw.get("architecture_id") != ARCHITECTURE_ID:
        print(f"ERROR: Wrong architecture_id: {raw.get('architecture_id')}", file=sys.stderr)
        return 3
    if not raw.get("trained"):
        print("ERROR: Checkpoint not marked trained=True.", file=sys.stderr)
        return 3
    if raw.get("experiment") != "two_modality_ecg_report_ptbxl":
        print(f"ERROR: Unexpected experiment tag: {raw.get('experiment')}", file=sys.stderr)
        return 3

    cfg = ClinicalTextEncoderConfig(max_sequence_length=256)
    pipe, _ = ClinicalTextEncodingPipeline.from_pretrained(config=cfg, device="cpu")
    model = load_ecg_report_checkpoint(ckpt_path, pipe.encoder, map_location="cpu")
    model.eval()

    ecg = torch.randn(2, 12, 1000)
    tok = pipe.tokenizer(["sinus rhythm", "st elevation"], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logit, _, _ = model(ecg, tok["input_ids"], tok["attention_mask"])
    if logit.shape != (2,):
        print(f"ERROR: Bad logit shape {logit.shape}", file=sys.stderr)
        return 4

    print(json.dumps({"ok": True, "checkpoint": str(ckpt_path), "architecture_id": ARCHITECTURE_ID}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

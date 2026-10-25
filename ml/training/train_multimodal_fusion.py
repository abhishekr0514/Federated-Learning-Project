"""
MedShieldDiagnosticNet — offline multimodal fusion training entry point.

Supervised training runs ONLY when `docs/data-audit/multimodal_linkage_audit.json`
reports `three_modality_gat_training_readiness.verdict == "CONDITIONAL_GO"`.

As of Prompt 16/16B audit: cross-cohort PTB-XL + NBME + cardio_train linkage is BLOCKED.
Do not join cohorts by numeric ID coincidence. Proxy PTB-XL targets are not validated
clinical heart-disease diagnosis.

Usage (after audit allows):
  py -m ml.training.train_multimodal_fusion --config configs/fusion_train.yaml

This module intentionally does not implement a training loop until linkage passes.
"""

from __future__ import annotations

import argparse
import json
import sys

from ml.training.linkage_gate import evaluate_three_modality_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multimodal GAT fusion training (gated).")
    parser.add_argument(
        "--check-linkage-only",
        action="store_true",
        help="Print linkage gate JSON and exit without training.",
    )
    args = parser.parse_args(argv)

    gate = evaluate_three_modality_gate()
    if args.check_linkage_only:
        print(json.dumps(gate.to_dict(), indent=2))
        return 0 if gate.allowed else 2

    if not gate.allowed:
        print("Multimodal fusion training aborted: linkage gate BLOCKED.", file=sys.stderr)
        print(json.dumps(gate.to_dict(), indent=2), file=sys.stderr)
        print(
            "See docs/data-audit/MULTIMODAL_LINKAGE_AUDIT.md and "
            "docs/research/MULTIMODAL_TRAINING_PROTOCOL.md",
            file=sys.stderr,
        )
        return 2

    print("Linkage gate passed; training implementation not yet wired.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
